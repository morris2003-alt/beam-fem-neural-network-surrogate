function beam_schemeB_main()
% Self-contained Scheme B runner. Outputs are relative to this file.
projectFolder = fileparts(mfilename('fullpath'));

% BEAM_SCHEMEB_MAIN
% Physics-guided, discontinuity-aware neural-network surrogates for a
% simply supported Euler-Bernoulli beam under:
%   (1) a full-span uniformly distributed load q, and
%   (2) one downward point load P at x = a.
%
% This version addresses the weaknesses found in the original multi-output
% surrogate:
%
%   A. Separate networks are trained for deflection, bending moment, and
%      the left/right shear-force branches.
%
%   B. Essential support conditions are imposed by output transformation:
%
%        w_hat(x) = xi(1-xi) * N_w(xi,q,P,a)
%        M_hat(x) = xi(1-xi) * N_M(xi,q,P,a)
%
%      where xi = x/L. Therefore w_hat(0)=w_hat(L)=0 and
%      M_hat(0)=M_hat(L)=0 exactly, independent of network weights.
%
%   C. Shear is modeled with two branch networks:
%
%        V_left  for x < a
%        V_right for x >= a
%
%      This avoids forcing one continuous network to smooth a physical
%      point-load discontinuity.
%
% IMPORTANT:
% - This is a physics-guided data-driven surrogate, NOT a PINN.
% - The differential equation is not placed in the loss function.
% - Do not report accuracy until reading the exported metrics.
%
% Required:
%   Deep Learning Toolbox
%
% Run:
%   beam_schemeB_main
%
% Results folder:
%   beam_project_schemeB_results

clc;
close all;
rng(42, 'twister');

%% USER OPTIONS
SHOW_TRAINING_GUI = true;
EXPORT_DPI = 300;

% 'trainscg' is much faster and less memory-intensive than trainlm for the
% four separate networks. The response functions are low-dimensional, so
% scaled conjugate gradient is a defensible default.
TRAIN_ALGORITHM = 'trainscg';

MAX_EPOCHS = 2500;
MAX_VALIDATION_FAILURES = 80;
MIN_GRADIENT = 1e-9;

% Same number of cases and seed as the verified baseline, allowing a fair
% architecture comparison.
nCases = 350;

% Network sizes. These can be increased only after checking validation
% behavior; larger networks are not automatically better.
hiddenW  = [64, 64, 32];
hiddenM  = [48, 48, 24];
hiddenVL = [32, 32];
hiddenVR = [32, 32];

resultsDir = fullfile(projectFolder, 'beam_project_schemeB_results');
if ~isfolder(resultsDir)
    mkdir(resultsDir);
end

fprintf('\n============================================================\n');
fprintf('SCHEME B: PHYSICS-GUIDED BEAM SURROGATES\n');
fprintf('============================================================\n');

%% STRUCTURAL MODEL
L = 6.0;            % [m]
E = 200e9;          % [Pa]
b = 0.15;           % [m]
h = 0.30;           % [m]
I = b*h^3/12;       % [m^4]

nElem = 40;
nNode = nElem + 1;
xNode = linspace(0, L, nNode)';
xiNode = xNode/L;

qRange = [2e3, 15e3];          % [N/m]
PRange = [5e3, 45e3];          % [N]
aRange = [0.15*L, 0.85*L];     % [m]

qScale = qRange(2);
PScale = PRange(2);

% Physics-based scales, used only for nondimensionalization.
wScale = qScale*L^4/(E*I) + PScale*L^3/(E*I);
MScale = qScale*L^2 + PScale*L;
VScale = qScale*L + PScale;

%% GENERATE VERIFIED DATASET
% Feature vector:
%   [xi; q/qScale; P/PScale; alpha; r; |r|]
% where:
%   xi    = x/L
%   alpha = a/L
%   r     = xi-alpha
%
% r and |r| are redundant in a strict mathematical sense, but explicitly
% expose the point-load-relative coordinate and help the network represent
% the response change near x=a.

nSamples = nCases*nNode;

Xfull = zeros(6, nSamples);
wFull = zeros(1, nSamples);
MFull = zeros(1, nSamples);
VFull = zeros(1, nSamples);
phiFull = zeros(1, nSamples);
caseID = zeros(1, nSamples);
leftBranch = false(1, nSamples);
caseParameters = zeros(nCases, 3);

cursor = 1;

fprintf('Generating %d complete load cases...\n', nCases);

for c = 1:nCases
    q = qRange(1) + rand*(qRange(2)-qRange(1));
    P = PRange(1) + rand*(PRange(2)-PRange(1));
    a = aRange(1) + rand*(aRange(2)-aRange(1));

    sol = solveEulerBernoulliBeam(L, E, I, nElem, q, P, a);
    [M, V] = sectionForces(L, q, P, a, xNode, 'right');

    alpha = a/L;
    r = xiNode-alpha;
    phi = xiNode.*(1-xiNode);

    idx = cursor:(cursor+nNode-1);

    Xfull(:,idx) = [xiNode';
                    (q/qScale)*ones(1,nNode);
                    (P/PScale)*ones(1,nNode);
                    alpha*ones(1,nNode);
                    r';
                    abs(r)'];

    wFull(idx) = sol.w';
    MFull(idx) = M';
    VFull(idx) = V';
    phiFull(idx) = phi';
    caseID(idx) = c;
    leftBranch(idx) = xiNode' < alpha;

    caseParameters(c,:) = [q, P, a];
    cursor = cursor+nNode;
end

fprintf('Generated %d spatial samples.\n', nSamples);

%% CASE-LEVEL TRAIN / VALIDATION / TEST SPLIT
% Splitting complete load cases prevents adjacent points from one beam case
% leaking into both training and testing sets.

caseOrder = randperm(nCases);
nTrainCases = round(0.70*nCases);
nValCases = round(0.15*nCases);

trainCases = caseOrder(1:nTrainCases);
valCases = caseOrder(nTrainCases+1:nTrainCases+nValCases);
testCases = caseOrder(nTrainCases+nValCases+1:end);

isTrainCase = ismember(caseID, trainCases);
isValCase = ismember(caseID, valCases);
isTestCase = ismember(caseID, testCases);

%% PREPARE HARD-CONSTRAINED DEFLECTION AND MOMENT TARGETS
% Boundary samples are excluded from latent-target training because phi=0.
% At prediction time, multiplying by phi enforces exact support values.

interior = phiFull > 1e-12;

XW = Xfull(:,interior);
TW = (wFull(interior)/wScale)./phiFull(interior);
caseW = caseID(interior);

XM = Xfull(:,interior);
TM = (MFull(interior)/MScale)./phiFull(interior);
caseM = caseID(interior);

trainW = find(ismember(caseW,trainCases));
valW = find(ismember(caseW,valCases));
testW = find(ismember(caseW,testCases));

trainM = find(ismember(caseM,trainCases));
valM = find(ismember(caseM,valCases));
testM = find(ismember(caseM,testCases));

%% PREPARE DISCONTINUITY-AWARE SHEAR TARGETS
maskVL = leftBranch;
maskVR = ~leftBranch;

XVL = Xfull(:,maskVL);
TVL = VFull(maskVL)/VScale;
caseVL = caseID(maskVL);

XVR = Xfull(:,maskVR);
TVR = VFull(maskVR)/VScale;
caseVR = caseID(maskVR);

trainVL = find(ismember(caseVL,trainCases));
valVL = find(ismember(caseVL,valCases));
testVL = find(ismember(caseVL,testCases));

trainVR = find(ismember(caseVR,trainCases));
valVR = find(ismember(caseVR,valCases));
testVR = find(ismember(caseVR,testCases));

%% TRAIN FOUR SEPARATE NETWORKS
fprintf('\nTraining hard-constrained deflection network...\n');
[netW,trW] = trainRegressionNetwork( ...
    XW,TW,trainW,valW,testW,hiddenW,TRAIN_ALGORITHM, ...
    MAX_EPOCHS,MAX_VALIDATION_FAILURES,MIN_GRADIENT,SHOW_TRAINING_GUI);

fprintf('\nTraining hard-constrained bending-moment network...\n');
[netM,trM] = trainRegressionNetwork( ...
    XM,TM,trainM,valM,testM,hiddenM,TRAIN_ALGORITHM, ...
    MAX_EPOCHS,MAX_VALIDATION_FAILURES,MIN_GRADIENT,SHOW_TRAINING_GUI);

fprintf('\nTraining left-branch shear network...\n');
[netVL,trVL] = trainRegressionNetwork( ...
    XVL,TVL,trainVL,valVL,testVL,hiddenVL,TRAIN_ALGORITHM, ...
    MAX_EPOCHS,MAX_VALIDATION_FAILURES,MIN_GRADIENT,SHOW_TRAINING_GUI);

fprintf('\nTraining right-branch shear network...\n');
[netVR,trVR] = trainRegressionNetwork( ...
    XVR,TVR,trainVR,valVR,testVR,hiddenVR,TRAIN_ALGORITHM, ...
    MAX_EPOCHS,MAX_VALIDATION_FAILURES,MIN_GRADIENT,SHOW_TRAINING_GUI);

models.netW = netW;
models.netM = netM;
models.netVL = netVL;
models.netVR = netVR;
models.L = L;
models.qScale = qScale;
models.PScale = PScale;
models.wScale = wScale;
models.MScale = MScale;
models.VScale = VScale;

%% HELD-OUT TEST-SET METRICS IN PHYSICAL UNITS
Xtest = Xfull(:,isTestCase);
phiTest = phiFull(isTestCase);
leftTest = leftBranch(isTestCase);

wTrueTest = wFull(isTestCase);
MTrueTest = MFull(isTestCase);
VTrueTest = VFull(isTestCase);

gWTest = netW(Xtest);
gMTest = netM(Xtest);

wPredTest = phiTest.*gWTest*wScale;
MPredTest = phiTest.*gMTest*MScale;

VPredTest = zeros(size(VTrueTest));
VPredTest(leftTest) = netVL(Xtest(:,leftTest))*VScale;
VPredTest(~leftTest) = netVR(Xtest(:,~leftTest))*VScale;

% Numerically exact constraints at endpoints.
xiTest = Xtest(1,:);
atBoundary = xiTest <= 1e-14 | xiTest >= 1-1e-14;
wPredTest(atBoundary) = 0;
MPredTest(atBoundary) = 0;

metrics = responseMetricsTable( ...
    wTrueTest,wPredTest,MTrueTest,MPredTest,VTrueTest,VPredTest);

fprintf('\nSCHEME B HELD-OUT TEST METRICS\n');
disp(metrics);

writetable(metrics,fullfile(resultsDir,'schemeB_test_metrics.csv'));

%% RESERVED UNSEEN CASE
% This case is fixed and explicitly checked against accidental duplication.

qNew = 11.3e3;
PNew = 31.7e3;
aNew = 0.63*L;

duplicate = any(abs(caseParameters(:,1)-qNew)<1e-9 & ...
                abs(caseParameters(:,2)-PNew)<1e-9 & ...
                abs(caseParameters(:,3)-aNew)<1e-12);
if duplicate
    error('The reserved unseen case duplicated a generated load case.');
end

unseenSol = solveEulerBernoulliBeam(L,E,I,nElem,qNew,PNew,aNew);
[MUnseen,VUnseen] = sectionForces(L,qNew,PNew,aNew,xNode,'right');

[wUnseenPred,MUnseenPred,VUnseenPred] = predictSchemeB( ...
    models,xNode,qNew,PNew,aNew);

unseenMetrics = table( ...
    ["Deflection";"Bending moment";"Shear force"], ...
    [sqrt(mean((wUnseenPred-unseenSol.w).^2))*1e3;
     sqrt(mean((MUnseenPred-MUnseen).^2))/1e3;
     sqrt(mean((VUnseenPred-VUnseen).^2))/1e3], ...
    ["mm";"kN m";"kN"], ...
    [coefficientOfDetermination(unseenSol.w,wUnseenPred);
     coefficientOfDetermination(MUnseen,MUnseenPred);
     coefficientOfDetermination(VUnseen,VUnseenPred)], ...
    100*[relativeL2(unseenSol.w,wUnseenPred);
         relativeL2(MUnseen,MUnseenPred);
         relativeL2(VUnseen,VUnseenPred)], ...
    'VariableNames',{'Response','RMSE','RMSEUnit','R2','RelativeL2_percent'});

fprintf('\nSCHEME B RESERVED UNSEEN-CASE METRICS\n');
disp(unseenMetrics);

writetable(unseenMetrics,fullfile(resultsDir,'schemeB_unseen_metrics.csv'));

%% PHYSICAL-CONSTRAINT CHECKS
boundaryX = [0;L];
[wBoundary,MBoundary,~] = predictSchemeB(models,boundaryX,qNew,PNew,aNew);

epsilonX = min(1e-6*L, 1e-4);
jumpX = [aNew-epsilonX;aNew+epsilonX];
[~,~,VJumpPred] = predictSchemeB(models,jumpX,qNew,PNew,aNew);
predictedJump = VJumpPred(2)-VJumpPred(1);
exactJump = -PNew;
jumpErrorPercent = 100*abs(predictedJump-exactJump)/PNew;

constraintMetrics = table( ...
    max(abs(wBoundary))*1e3, ...
    max(abs(MBoundary))/1e3, ...
    predictedJump/1e3, ...
    exactJump/1e3, ...
    jumpErrorPercent, ...
    'VariableNames',{'MaxSupportDeflectionResidual_mm', ...
    'MaxSupportMomentResidual_kNm','PredictedShearJump_kN', ...
    'ExactShearJump_kN','ShearJumpError_percent'});

fprintf('\nPHYSICAL-CONSTRAINT CHECKS\n');
disp(constraintMetrics);

writetable(constraintMetrics, ...
    fullfile(resultsDir,'schemeB_constraint_checks.csv'));

%% COMPARE WITH VERIFIED BASELINE IF AVAILABLE
baselinePaths = {
    fullfile(projectFolder,'beam_project_results','nn_metrics.csv')
    fullfile(projectFolder,'nn_metrics.csv')
};

baselineFile = "";
for k = 1:numel(baselinePaths)
    if isfile(baselinePaths{k})
        baselineFile = string(baselinePaths{k});
        break;
    end
end

if strlength(baselineFile)>0
    baseline = readtable(baselineFile);

    comparison = table( ...
        string(baseline.Response), ...
        baseline.TestR2, ...
        metrics.TestR2, ...
        metrics.TestR2-baseline.TestR2, ...
        baseline.UnseenRelativeL2_percent, ...
        unseenMetrics.RelativeL2_percent, ...
        baseline.UnseenRelativeL2_percent-unseenMetrics.RelativeL2_percent, ...
        'VariableNames',{'Response','BaselineTestR2','SchemeBTestR2', ...
        'DeltaTestR2','BaselineUnseenL2_percent','SchemeBUnseenL2_percent', ...
        'ReductionInUnseenL2_points'});

    fprintf('\nCOMPARISON WITH VERIFIED BASELINE\n');
    disp(comparison);
    writetable(comparison, ...
        fullfile(resultsDir,'schemeB_vs_baseline.csv'));
else
    comparison = table;
    fprintf('\nBaseline nn_metrics.csv was not found; comparison skipped.\n');
end

%% EXPORT UNSEEN-CASE FIGURES
fig = figure('Name','Scheme B Unseen Deflection');
plot(xNode,unseenSol.w*1e3,'-','LineWidth',1.8);
hold on;
plot(xNode,wUnseenPred*1e3,'--','LineWidth',1.8);
grid on;
xlabel('Position x [m]');
ylabel('Deflection w [mm]');
title('Scheme B: Unseen-Case Deflection');
legend('FEM reference','Physics-guided NN prediction','Location','best');
saveFigure(fig,fullfile(resultsDir,'01_schemeB_unseen_deflection.png'),EXPORT_DPI);

fig = figure('Name','Scheme B Unseen Bending Moment');
plot(xNode,MUnseen/1e3,'-','LineWidth',1.8);
hold on;
plot(xNode,MUnseenPred/1e3,'--','LineWidth',1.8);
grid on;
xlabel('Position x [m]');
ylabel('Sagging bending moment M [kN m]');
title('Scheme B: Unseen-Case Bending Moment');
legend('Section-equilibrium reference','Physics-guided NN prediction', ...
    'Location','best');
saveFigure(fig,fullfile(resultsDir,'02_schemeB_unseen_bending_moment.png'),EXPORT_DPI);

% Repeat a around the jump for a visually exact discontinuity.
xShearPlot = sort(unique([xNode;aNew-epsilonX;aNew;aNew+epsilonX]));
[~,VShearTrue] = sectionForces(L,qNew,PNew,aNew,xShearPlot,'right');
[~,~,VShearPred] = predictSchemeB(models,xShearPlot,qNew,PNew,aNew);

fig = figure('Name','Scheme B Unseen Shear Force');
plot(xShearPlot,VShearTrue/1e3,'-','LineWidth',1.8);
hold on;
plot(xShearPlot,VShearPred/1e3,'--','LineWidth',1.8);
grid on;
xlabel('Position x [m]');
ylabel('Shear force V [kN]');
title('Scheme B: Discontinuity-Aware Unseen Shear Force');
legend('Section-equilibrium reference','Piecewise NN prediction', ...
    'Location','best');
saveFigure(fig,fullfile(resultsDir,'03_schemeB_unseen_shear_force.png'),EXPORT_DPI);

%% PER-OUTPUT PARITY PLOTS
makeParityPlot(wTrueTest*1e3,wPredTest*1e3, ...
    'Deflection target [mm]','Deflection prediction [mm]', ...
    'Scheme B Test Parity: Deflection', ...
    fullfile(resultsDir,'04_schemeB_test_parity_deflection.png'),EXPORT_DPI);

makeParityPlot(MTrueTest/1e3,MPredTest/1e3, ...
    'Moment target [kN m]','Moment prediction [kN m]', ...
    'Scheme B Test Parity: Bending Moment', ...
    fullfile(resultsDir,'05_schemeB_test_parity_moment.png'),EXPORT_DPI);

makeParityPlot(VTrueTest/1e3,VPredTest/1e3, ...
    'Shear target [kN]','Shear prediction [kN]', ...
    'Scheme B Test Parity: Shear Force', ...
    fullfile(resultsDir,'06_schemeB_test_parity_shear.png'),EXPORT_DPI);

%% TRAINING-PERFORMANCE FIGURES
exportTrainingPerformance(trW,'Deflection latent network', ...
    fullfile(resultsDir,'07_training_performance_deflection.png'),EXPORT_DPI);
exportTrainingPerformance(trM,'Moment latent network', ...
    fullfile(resultsDir,'08_training_performance_moment.png'),EXPORT_DPI);
exportTrainingPerformance(trVL,'Left shear branch network', ...
    fullfile(resultsDir,'09_training_performance_shear_left.png'),EXPORT_DPI);
exportTrainingPerformance(trVR,'Right shear branch network', ...
    fullfile(resultsDir,'10_training_performance_shear_right.png'),EXPORT_DPI);

%% SAVE MODELS AND REPRODUCIBILITY DATA
save(fullfile(resultsDir,'beam_schemeB_models.mat'), ...
    'models','trW','trM','trVL','trVR', ...
    'L','E','I','b','h','nElem','nCases', ...
    'qRange','PRange','aRange','caseParameters', ...
    'trainCases','valCases','testCases', ...
    'metrics','unseenMetrics','constraintMetrics','comparison', ...
    'TRAIN_ALGORITHM','hiddenW','hiddenM','hiddenVL','hiddenVR');

%% WRITE HONEST SUMMARY
summaryFile = fullfile(resultsDir,'schemeB_results_summary.txt');
fid = fopen(summaryFile,'w');

if fid<0
    warning('Could not create schemeB_results_summary.txt');
else
    fprintf(fid,'SCHEME B: PHYSICS-GUIDED BEAM SURROGATE RESULTS\n');
    fprintf(fid,'================================================\n\n');

    fprintf(fid,'METHOD\n');
    fprintf(fid,'- Separate deflection, moment, left-shear, and right-shear networks.\n');
    fprintf(fid,'- Deflection and moment support conditions imposed by output transformation.\n');
    fprintf(fid,'- Piecewise shear architecture used across the point-load discontinuity.\n');
    fprintf(fid,'- Complete load cases split into training/validation/test groups.\n');
    fprintf(fid,'- This is not a PINN; no governing-equation residual was used in the loss.\n\n');

    fprintf(fid,'TEST METRICS\n');
    for i = 1:height(metrics)
        fprintf(fid,'%s: RMSE = %.6f %s, R^2 = %.6f\n', ...
            metrics.Response(i),metrics.TestRMSE(i), ...
            metrics.RMSEUnit(i),metrics.TestR2(i));
    end

    fprintf(fid,'\nRESERVED UNSEEN CASE\n');
    fprintf(fid,'q = %.3f kN/m, P = %.3f kN, a/L = %.3f\n', ...
        qNew/1e3,PNew/1e3,aNew/L);
    for i = 1:height(unseenMetrics)
        fprintf(fid,'%s: RMSE = %.6f %s, R^2 = %.6f, relative L2 = %.6f %%\n', ...
            unseenMetrics.Response(i),unseenMetrics.RMSE(i), ...
            unseenMetrics.RMSEUnit(i),unseenMetrics.R2(i), ...
            unseenMetrics.RelativeL2_percent(i));
    end

    fprintf(fid,'\nPHYSICAL CONSTRAINTS\n');
    fprintf(fid,'Maximum support deflection residual = %.12e mm\n', ...
        constraintMetrics.MaxSupportDeflectionResidual_mm);
    fprintf(fid,'Maximum support moment residual = %.12e kN m\n', ...
        constraintMetrics.MaxSupportMomentResidual_kNm);
    fprintf(fid,'Predicted shear jump = %.6f kN\n', ...
        constraintMetrics.PredictedShearJump_kN);
    fprintf(fid,'Exact shear jump = %.6f kN\n', ...
        constraintMetrics.ExactShearJump_kN);
    fprintf(fid,'Shear jump error = %.6f %%\n', ...
        constraintMetrics.ShearJumpError_percent);

    fprintf(fid,'\nTRAINING RECORDS\n');
    fprintf(fid,'Deflection stop reason: %s; best epoch: %d\n', ...
        string(trW.stop),trW.best_epoch);
    fprintf(fid,'Moment stop reason: %s; best epoch: %d\n', ...
        string(trM.stop),trM.best_epoch);
    fprintf(fid,'Left shear stop reason: %s; best epoch: %d\n', ...
        string(trVL.stop),trVL.best_epoch);
    fprintf(fid,'Right shear stop reason: %s; best epoch: %d\n', ...
        string(trVR.stop),trVR.best_epoch);

    fclose(fid);
end

fprintf('\n============================================================\n');
fprintf('SCHEME B COMPLETED\n');
fprintf('Results exported to:\n%s\n',resultsDir);
fprintf('============================================================\n');

if ispc
    winopen(resultsDir);
end

end

%% ========================================================================
function [net,tr] = trainRegressionNetwork( ...
    X,T,trainInd,valInd,testInd,hidden,algorithm, ...
    maxEpochs,maxFail,minGrad,showGUI)

net = fitnet(hidden,algorithm);
net.performFcn = 'mse';
net.divideFcn = 'divideind';

net.divideParam.trainInd = trainInd(:)';
net.divideParam.valInd = valInd(:)';
net.divideParam.testInd = testInd(:)';

net.trainParam.epochs = maxEpochs;
net.trainParam.max_fail = maxFail;
net.trainParam.min_grad = minGrad;
net.trainParam.showWindow = showGUI;

net.inputs{1}.processFcns = {'removeconstantrows','mapminmax'};
net.outputs{end}.processFcns = {'removeconstantrows','mapminmax'};

[net,tr] = train(net,X,T);

end

%% ========================================================================
function [wPred,MPred,VPred] = predictSchemeB(models,x,q,P,a)

x = x(:);
L = models.L;
xi = x/L;
alpha = a/L;
r = xi-alpha;
phi = xi.*(1-xi);

X = [xi';
     (q/models.qScale)*ones(1,numel(x));
     (P/models.PScale)*ones(1,numel(x));
     alpha*ones(1,numel(x));
     r';
     abs(r)'];

gW = models.netW(X)';
gM = models.netM(X)';

wPred = phi.*gW*models.wScale;
MPred = phi.*gM*models.MScale;

left = xi<alpha;
VPred = zeros(size(x));

if any(left)
    VPred(left) = models.netVL(X(:,left))'*models.VScale;
end
if any(~left)
    VPred(~left) = models.netVR(X(:,~left))'*models.VScale;
end

% Enforce exact support conditions numerically as well as algebraically.
boundary = xi<=1e-14 | xi>=1-1e-14;
wPred(boundary) = 0;
MPred(boundary) = 0;

end

%% ========================================================================
function metrics = responseMetricsTable( ...
    wTrue,wPred,MTrue,MPred,VTrue,VPred)

metrics = table( ...
    ["Deflection";"Bending moment";"Shear force"], ...
    [sqrt(mean((wPred-wTrue).^2))*1e3;
     sqrt(mean((MPred-MTrue).^2))/1e3;
     sqrt(mean((VPred-VTrue).^2))/1e3], ...
    ["mm";"kN m";"kN"], ...
    [coefficientOfDetermination(wTrue,wPred);
     coefficientOfDetermination(MTrue,MPred);
     coefficientOfDetermination(VTrue,VPred)], ...
    'VariableNames',{'Response','TestRMSE','RMSEUnit','TestR2'});

end

%% ========================================================================
function sol = solveEulerBernoulliBeam(L,E,I,nElem,qDown,PDown,a)

nNode = nElem+1;
nDOF = 2*nNode;
Le = L/nElem;
xNode = linspace(0,L,nNode)';

K = zeros(nDOF,nDOF);
F = zeros(nDOF,1);
elementLoad = zeros(4,nElem);

q = -qDown;

ke = (E*I/Le^3)* ...
    [12,6*Le,-12,6*Le;
     6*Le,4*Le^2,-6*Le,2*Le^2;
     -12,-6*Le,12,-6*Le;
     6*Le,2*Le^2,-6*Le,4*Le^2];

feUDL = q*Le/2*[1;Le/6;1;-Le/6];

for e = 1:nElem
    dofs = elementDOFs(e);
    elementLoad(:,e) = feUDL;
    K(dofs,dofs) = K(dofs,dofs)+ke;
    F(dofs) = F(dofs)+feUDL;
end

if PDown>0
    if a==L
        pointElem = nElem;
        eta = 1;
    else
        pointElem = floor(a/Le)+1;
        pointElem = min(max(pointElem,1),nElem);
        xLeft = (pointElem-1)*Le;
        eta = (a-xLeft)/Le;
    end

    N = hermiteShapeFunctions(eta,Le);
    fPoint = (-PDown)*N;
    dofs = elementDOFs(pointElem);

    elementLoad(:,pointElem) = elementLoad(:,pointElem)+fPoint;
    F(dofs) = F(dofs)+fPoint;
end

fixedDOF = [1,2*nNode-1];
freeDOF = setdiff(1:nDOF,fixedDOF);

d = zeros(nDOF,1);
d(freeDOF) = K(freeDOF,freeDOF)\F(freeDOF);

reactionVector = K*d-F;

sol.xNode = xNode;
sol.d = d;
sol.w = d(1:2:end);
sol.theta = d(2:2:end);
sol.reactions = reactionVector(fixedDOF);

end

%% ========================================================================
function [M,V] = sectionForces(L,q,P,a,x,side)

x = x(:);
RA = q*L/2+P*(L-a)/L;

if strcmpi(side,'left')
    H = double(x>a);
else
    H = double(x>=a);
end

V = RA-q*x-P*H;
M = RA*x-q*x.^2/2-P*max(x-a,0);

end

%% ========================================================================
function N = hermiteShapeFunctions(eta,Le)

N1 = 1-3*eta^2+2*eta^3;
N2 = Le*(eta-2*eta^2+eta^3);
N3 = 3*eta^2-2*eta^3;
N4 = Le*(-eta^2+eta^3);

N = [N1;N2;N3;N4];

end

%% ========================================================================
function dofs = elementDOFs(e)

dofs = [2*e-1,2*e,2*(e+1)-1,2*(e+1)];

end

%% ========================================================================
function value = coefficientOfDetermination(yTrue,yPred)

yTrue = yTrue(:);
yPred = yPred(:);

ssRes = sum((yTrue-yPred).^2);
ssTot = sum((yTrue-mean(yTrue)).^2);

if ssTot<=eps
    value = NaN;
else
    value = 1-ssRes/ssTot;
end

end

%% ========================================================================
function value = relativeL2(yTrue,yPred)

value = norm(yPred(:)-yTrue(:))/max(norm(yTrue(:)),eps);

end

%% ========================================================================
function makeParityPlot(yTrue,yPred,xText,yText,titleText,filePath,dpi)

fig = figure('Name',titleText);
scatter(yTrue,yPred,18,'o');
hold on;

lower = min([yTrue(:);yPred(:)]);
upper = max([yTrue(:);yPred(:)]);

plot([lower,upper],[lower,upper],'--','LineWidth',1.5);

p = polyfit(yTrue(:),yPred(:),1);
plot([lower,upper],polyval(p,[lower,upper]),'-','LineWidth',1.5);

grid on;
xlabel(xText);
ylabel(yText);
title(sprintf('%s, R^2 = %.5f',titleText, ...
    coefficientOfDetermination(yTrue,yPred)));
legend('Test samples','Identity line','Linear fit','Location','best');

saveFigure(fig,filePath,dpi);

end

%% ========================================================================
function exportTrainingPerformance(tr,networkName,filePath,dpi)

fig = figure('Name',['Performance: ',networkName]);
plotperform(tr);
sgtitle(networkName);

saveFigure(fig,filePath,dpi);

end

%% ========================================================================
function saveFigure(fig,filePath,dpi)

try
    exportgraphics(fig,filePath,'Resolution',dpi);
catch
    print(fig,filePath,'-dpng',sprintf('-r%d',dpi));
end

end