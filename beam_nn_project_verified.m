%% beam_nn_project_verified.m
% Verified Computational Modeling of Euler-Bernoulli Beam Response
% Independent Computational Project | MATLAB
%
% Major upgrades from the earlier version:
% 1) Correct element end-force recovery:
%       s_e = k_e d_e - f_e
%    where f_e includes BOTH the distributed-load vector and the
%    consistent point-load vector for the loaded element.
%
% 2) Internal-force plotting/data recovery uses exact section equilibrium
%    for this load family (full-span UDL + one point load). This prevents
%    the concentrated-load jump from being artificially smeared by nodal
%    averaging.
%
% 3) Analytical validation against the closed-form simply supported beam
%    solution.
%
% 4) Mesh-convergence study with relative L2 errors and observed order.
%
% 5) Neural-network metrics exported automatically:
%       RMSE, R^2, and unseen-case relative L2 error.
%
% Required toolbox for neural-network training:
% Deep Learning Toolbox
%
% Output folder:
%   beam_project_results
%
% The script exports:
%   validation_summary.csv
%   mesh_convergence.csv
%   nn_metrics.csv
%   portfolio_results_summary.txt
%   publication-ready PNG figures
%   beam_response_nn_model_verified.mat

clear; clc; close all;
rng(42);

%% 0. OPTIONS
RUN_NEURAL_NETWORK = true;  % Set false to run only FEM validation/convergence
SHOW_TRAINING_GUI  = true;  % Set false for non-interactive execution
EXPORT_DPI         = 300;

projectFolder = fileparts(mfilename('fullpath'));
resultsDir = fullfile(projectFolder, 'beam_project_results');
if ~exist(resultsDir, 'dir')
    mkdir(resultsDir);
end

%% 1. STRUCTURAL MODEL
L  = 6.0;          % Beam length [m]
E  = 200e9;        % Young's modulus [Pa]
b  = 0.15;         % Rectangular section width [m]
h  = 0.30;         % Rectangular section height [m]
I  = b*h^3/12;     % Second moment of area [m^4]

nElem = 40;

% Reference loading case
q_ref = 8e3;        % Downward UDL [N/m]
P_ref = 25e3;       % Downward point load [N]
a_ref = 0.38*L;     % Point-load location [m]

fprintf('\n============================================================\n');
fprintf('VERIFIED EULER-BERNOULLI BEAM PROJECT\n');
fprintf('============================================================\n');

%% 2. REFERENCE FEM SOLUTION
ref = solveEulerBernoulliBeamVerified(L, E, I, nElem, q_ref, P_ref, a_ref);

% Dense evaluation grid
xDense = linspace(0, L, 4001)';
wFEM   = evaluateFEDisplacement(ref.xNode, ref.d, xDense);
M_FEM  = evaluateFEBendingMoment(ref.xNode, ref.d, E, I, xDense);

% Closed-form analytical solution
ana = analyticalSimplySupportedBeam(L, E, I, q_ref, P_ref, a_ref, xDense);

% Exact section-force curves used for final plotting and NN targets
[M_section, V_section] = sectionForcesSimplySupported( ...
    L, q_ref, P_ref, a_ref, xDense, 'right');

%% 3. ANALYTICAL VALIDATION
maxDefFEM = max(abs(wFEM));
maxDefAna = max(abs(ana.w));

maxMomentFEM = max(abs(M_FEM));
maxMomentAna = max(abs(ana.M));

defRelL2 = norm(wFEM - ana.w) / norm(ana.w);
momRelL2 = norm(M_FEM - ana.M) / norm(ana.M);

leftReactionError  = relativeError(ref.reactions(1), ana.RA);
rightReactionError = relativeError(ref.reactions(2), ana.RB);
maxDefError        = relativeError(maxDefFEM, maxDefAna);
maxMomentError     = relativeError(maxMomentFEM, maxMomentAna);

equilibriumResidual = abs(sum(ref.reactions) - (q_ref*L + P_ref));

validationTable = table( ...
    ["Left support reaction"; "Right support reaction"; ...
     "Maximum absolute deflection"; "Maximum absolute bending moment"], ...
    [ref.reactions(1)/1e3; ref.reactions(2)/1e3; ...
     maxDefFEM*1e3; maxMomentFEM/1e3], ...
    [ana.RA/1e3; ana.RB/1e3; ...
     maxDefAna*1e3; maxMomentAna/1e3], ...
    100*[leftReactionError; rightReactionError; maxDefError; maxMomentError], ...
    ["kN"; "kN"; "mm"; "kN m"], ...
    'VariableNames', {'Quantity','FEM','Analytical','RelativeErrorPercent','Unit'});

writetable(validationTable, fullfile(resultsDir, 'validation_summary.csv'));

fprintf('\nANALYTICAL VALIDATION\n');
disp(validationTable);
fprintf('Deflection relative L2 error = %.6f %%\n', 100*defRelL2);
fprintf('Moment relative L2 error     = %.6f %%\n', 100*momRelL2);
fprintf('Vertical equilibrium residual= %.6e N\n', equilibriumResidual);

% Validation figure: displacement
fig = figure('Name','Analytical Validation: Deflection');
plot(xDense, ana.w*1e3, '-', 'LineWidth', 1.8);
hold on;
plot(xDense, wFEM*1e3, '--', 'LineWidth', 1.6);
grid on;
xlabel('Position x [m]');
ylabel('Deflection w [mm]');
title('Analytical Validation of Beam Deflection');
legend('Closed-form analytical solution','Finite-element solution', ...
    'Location','best');
saveFigure(fig, fullfile(resultsDir,'01_analytical_validation_deflection.png'), EXPORT_DPI);

% Validation figure: bending moment
fig = figure('Name','Analytical Validation: Bending Moment');
plot(xDense, ana.M/1e3, '-', 'LineWidth', 1.8);
hold on;
plot(xDense, M_FEM/1e3, '--', 'LineWidth', 1.6);
grid on;
xlabel('Position x [m]');
ylabel('Sagging bending moment M [kN m]');
title('Analytical Validation of Bending Moment');
legend('Closed-form analytical solution','FEM curvature recovery', ...
    'Location','best');
saveFigure(fig, fullfile(resultsDir,'02_analytical_validation_moment.png'), EXPORT_DPI);

% Exact section-force plots with concentrated-load discontinuity
[xJump, MJump, VJump] = sectionForcePlotCoordinates(L, q_ref, P_ref, a_ref);

fig = figure('Name','Reference Section Forces');
plot(xJump, MJump/1e3, 'LineWidth', 1.8);
grid on;
xlabel('Position x [m]');
ylabel('Sagging bending moment M [kN m]');
title('Bending-Moment Distribution from Section Equilibrium');
saveFigure(fig, fullfile(resultsDir,'03_reference_bending_moment.png'), EXPORT_DPI);

fig = figure('Name','Reference Shear Force');
plot(xJump, VJump/1e3, 'LineWidth', 1.8);
grid on;
xlabel('Position x [m]');
ylabel('Shear force V [kN]');
title('Shear-Force Distribution with Exact Point-Load Jump');
saveFigure(fig, fullfile(resultsDir,'04_reference_shear_force.png'), EXPORT_DPI);

%% 4. MESH-CONVERGENCE STUDY
nElemList = [4, 8, 16, 32, 64, 128];
nMesh = numel(nElemList);

meshSize      = zeros(nMesh,1);
nDOF          = zeros(nMesh,1);
maxDeflection = zeros(nMesh,1);
maxDefErrPct  = zeros(nMesh,1);
defL2Pct      = zeros(nMesh,1);
momL2Pct      = zeros(nMesh,1);
observedOrder = nan(nMesh,1);

for i = 1:nMesh
    nE = nElemList(i);
    sol = solveEulerBernoulliBeamVerified(L, E, I, nE, q_ref, P_ref, a_ref);

    wMesh = evaluateFEDisplacement(sol.xNode, sol.d, xDense);
    MMesh = evaluateFEBendingMoment(sol.xNode, sol.d, E, I, xDense);

    meshSize(i)      = L/nE;
    nDOF(i)          = 2*(nE+1) - 2; % Free structural DOFs
    maxDeflection(i) = max(abs(wMesh))*1e3;
    maxDefErrPct(i)  = 100*relativeError(max(abs(wMesh)), maxDefAna);
    defL2Pct(i)      = 100*norm(wMesh-ana.w)/norm(ana.w);
    momL2Pct(i)      = 100*norm(MMesh-ana.M)/norm(ana.M);

    if i > 1 && defL2Pct(i) > 0
        observedOrder(i) = log(defL2Pct(i-1)/defL2Pct(i)) / ...
                           log(meshSize(i-1)/meshSize(i));
    end
end

meshTable = table(nElemList(:), meshSize, nDOF, maxDeflection, ...
    maxDefErrPct, defL2Pct, momL2Pct, observedOrder, ...
    'VariableNames', {'Elements','ElementLength_m','FreeDOF', ...
    'MaxDeflection_mm','MaxDeflectionError_percent', ...
    'DeflectionRelativeL2_percent','MomentRelativeL2_percent', ...
    'ObservedOrder_DeflectionL2'});

writetable(meshTable, fullfile(resultsDir, 'mesh_convergence.csv'));

fprintf('\nMESH-CONVERGENCE STUDY\n');
disp(meshTable);

fig = figure('Name','Mesh Convergence');
loglog(meshSize, defL2Pct, '-o', 'LineWidth', 1.8, 'MarkerSize', 7);
hold on;
loglog(meshSize, momL2Pct, '-s', 'LineWidth', 1.8, 'MarkerSize', 7);
grid on;
set(gca,'XDir','reverse');
xlabel('Element length h [m]');
ylabel('Relative L_2 error [%]');
title('Mesh-Convergence Study');
legend('Deflection error','Bending-moment error','Location','best');
saveFigure(fig, fullfile(resultsDir,'05_mesh_convergence.png'), EXPORT_DPI);

%% 5. PARAMETRIC DATASET
nNode = nElem + 1;
nCases = 350;

qRange = [2e3, 15e3];         % [N/m]
PRange = [5e3, 45e3];         % [N]
aRange = [0.15*L, 0.85*L];    % [m]

qScale = qRange(2);
PScale = PRange(2);

wScale = qScale*L^4/(E*I) + PScale*L^3/(E*I);
MScale = qScale*L^2 + PScale*L;
VScale = qScale*L + PScale;

nSamples = nCases*nNode;
X = zeros(4, nSamples);
T = zeros(3, nSamples);
caseParameters = zeros(nCases,3);

sampleIndex = 1;
fprintf('\nGenerating %d verified FEM load cases...\n', nCases);

for c = 1:nCases
    q = qRange(1) + rand*(qRange(2)-qRange(1));
    P = PRange(1) + rand*(PRange(2)-PRange(1));
    a = aRange(1) + rand*(aRange(2)-aRange(1));

    sol = solveEulerBernoulliBeamVerified(L, E, I, nElem, q, P, a);
    x = sol.xNode;
    w = sol.w;

    % Exact section-equilibrium recovery for M and V.
    [M, V] = sectionForcesSimplySupported(L, q, P, a, x, 'right');

    idx = sampleIndex:(sampleIndex+nNode-1);

    X(:,idx) = [x(:)'/L;
                (q/qScale)*ones(1,nNode);
                (P/PScale)*ones(1,nNode);
                (a/L)*ones(1,nNode)];

    T(:,idx) = [w(:)'/wScale;
                M(:)'/MScale;
                V(:)'/VScale];

    caseParameters(c,:) = [q,P,a];
    sampleIndex = sampleIndex+nNode;
end

fprintf('Dataset generated: %d samples.\n', size(X,2));

%% 6. TRAIN / VALIDATION / TEST SPLIT BY COMPLETE LOAD CASE
caseOrder = randperm(nCases);
nTrainCases = round(0.70*nCases);
nValCases   = round(0.15*nCases);
nTestCases  = nCases-nTrainCases-nValCases;

trainCases = caseOrder(1:nTrainCases);
valCases   = caseOrder(nTrainCases+1:nTrainCases+nValCases);
testCases  = caseOrder(nTrainCases+nValCases+1:end);

trainInd = caseNumbersToSampleIndices(trainCases,nNode);
valInd   = caseNumbersToSampleIndices(valCases,nNode);
testInd  = caseNumbersToSampleIndices(testCases,nNode);

%% 7. NEURAL-NETWORK TRAINING AND QUANTITATIVE METRICS
if RUN_NEURAL_NETWORK
    hiddenLayerSizes = [64,64,32];
    net = fitnet(hiddenLayerSizes,'trainlm');

    net.performFcn = 'mse';
    net.divideFcn = 'divideind';
    net.divideParam.trainInd = trainInd;
    net.divideParam.valInd = valInd;
    net.divideParam.testInd = testInd;

    net.trainParam.epochs = 1000;
    net.trainParam.max_fail = 25;
    net.trainParam.min_grad = 1e-10;
    net.trainParam.showWindow = SHOW_TRAINING_GUI;

    net.inputs{1}.processFcns = {'removeconstantrows','mapminmax'};
    net.outputs{end}.processFcns = {'removeconstantrows','mapminmax'};

    fprintf('\nTraining neural-network surrogate...\n');
    [net,tr] = train(net,X,T);

    Y = net(X);

    trainMSE = perform(net,T(:,trainInd),Y(:,trainInd));
    valMSE   = perform(net,T(:,valInd),Y(:,valInd));
    testMSE  = perform(net,T(:,testInd),Y(:,testInd));

    wTrueTest = T(1,testInd)*wScale;
    mTrueTest = T(2,testInd)*MScale;
    vTrueTest = T(3,testInd)*VScale;

    wPredTest = Y(1,testInd)*wScale;
    mPredTest = Y(2,testInd)*MScale;
    vPredTest = Y(3,testInd)*VScale;

    rmseW = sqrt(mean((wPredTest-wTrueTest).^2));
    rmseM = sqrt(mean((mPredTest-mTrueTest).^2));
    rmseV = sqrt(mean((vPredTest-vTrueTest).^2));

    r2W = coefficientOfDetermination(wTrueTest,wPredTest);
    r2M = coefficientOfDetermination(mTrueTest,mPredTest);
    r2V = coefficientOfDetermination(vTrueTest,vPredTest);

    % Explicitly reserved unseen case
    q_new = 11.3e3;
    P_new = 31.7e3;
    a_new = 0.63*L;

    % Verify it is not numerically duplicated in the random dataset.
    duplicateTol = [1e-9,1e-9,1e-12];
    duplicated = any( ...
        abs(caseParameters(:,1)-q_new) <= duplicateTol(1) & ...
        abs(caseParameters(:,2)-P_new) <= duplicateTol(2) & ...
        abs(caseParameters(:,3)-a_new) <= duplicateTol(3));

    if duplicated
        error('The reserved unseen case unexpectedly duplicates a generated case.');
    end

    unseen = solveEulerBernoulliBeamVerified(L,E,I,nElem,q_new,P_new,a_new);
    xNew = unseen.xNode;
    wTrue = unseen.w;
    [MTrue,VTrue] = sectionForcesSimplySupported(L,q_new,P_new,a_new,xNew,'right');

    Xnew = [xNew(:)'/L;
            (q_new/qScale)*ones(1,nNode);
            (P_new/PScale)*ones(1,nNode);
            (a_new/L)*ones(1,nNode)];

    Ynew = net(Xnew);
    wPred = Ynew(1,:)'*wScale;
    MPred = Ynew(2,:)'*MScale;
    VPred = Ynew(3,:)'*VScale;

    unseenL2W = norm(wPred-wTrue)/norm(wTrue);
    unseenL2M = norm(MPred-MTrue)/norm(MTrue);
    unseenL2V = norm(VPred-VTrue)/norm(VTrue);

    nnMetrics = table( ...
        ["Deflection";"Bending moment";"Shear force"], ...
        [rmseW*1e3;rmseM/1e3;rmseV/1e3], ...
        ["mm";"kN m";"kN"], ...
        [r2W;r2M;r2V], ...
        100*[unseenL2W;unseenL2M;unseenL2V], ...
        'VariableNames', {'Response','TestRMSE','RMSEUnit','TestR2', ...
        'UnseenRelativeL2_percent'});

    writetable(nnMetrics, fullfile(resultsDir,'nn_metrics.csv'));

    fprintf('\nNEURAL-NETWORK QUANTITATIVE RESULTS\n');
    fprintf('Training MSE   = %.6e\n',trainMSE);
    fprintf('Validation MSE = %.6e\n',valMSE);
    fprintf('Test MSE       = %.6e\n',testMSE);
    disp(nnMetrics);

    % Unseen comparison figures
    fig = figure('Name','Unseen Deflection Comparison');
    plot(xNew,wTrue*1e3,'-','LineWidth',1.8);
    hold on;
    plot(xNew,wPred*1e3,'--','LineWidth',1.8);
    grid on;
    xlabel('Position x [m]');
    ylabel('Deflection w [mm]');
    title('Unseen Case: Deflection');
    legend('FEM reference','Neural-network prediction','Location','best');
    saveFigure(fig,fullfile(resultsDir,'06_unseen_deflection.png'),EXPORT_DPI);

    fig = figure('Name','Unseen Moment Comparison');
    plot(xNew,MTrue/1e3,'-','LineWidth',1.8);
    hold on;
    plot(xNew,MPred/1e3,'--','LineWidth',1.8);
    grid on;
    xlabel('Position x [m]');
    ylabel('Sagging bending moment M [kN m]');
    title('Unseen Case: Bending Moment');
    legend('Section-equilibrium reference','Neural-network prediction', ...
        'Location','best');
    saveFigure(fig,fullfile(resultsDir,'07_unseen_bending_moment.png'),EXPORT_DPI);

    fig = figure('Name','Unseen Shear Comparison');
    plot(xNew,VTrue/1e3,'-','LineWidth',1.8);
    hold on;
    plot(xNew,VPred/1e3,'--','LineWidth',1.8);
    grid on;
    xlabel('Position x [m]');
    ylabel('Shear force V [kN]');
    title('Unseen Case: Shear Force');
    legend('Section-equilibrium reference','Neural-network prediction', ...
        'Location','best');
    saveFigure(fig,fullfile(resultsDir,'08_unseen_shear_force.png'),EXPORT_DPI);

    % Training performance and regression plots
    fig = figure('Name','Network Performance');
    plotperform(tr);
    saveFigure(fig,fullfile(resultsDir,'09_training_performance.png'),EXPORT_DPI);

    fig = figure('Name','Test Regression');
    plotregression(T(:,testInd),Y(:,testInd),'Test set');
    saveFigure(fig,fullfile(resultsDir,'10_test_regression.png'),EXPORT_DPI);

    save(fullfile(resultsDir,'beam_response_nn_model_verified.mat'), ...
        'net','tr','L','E','I','b','h','nElem','qScale','PScale', ...
        'wScale','MScale','VScale','qRange','PRange','aRange', ...
        'caseParameters','trainCases','valCases','testCases', ...
        'validationTable','meshTable','nnMetrics');
else
    trainMSE = NaN;
    valMSE = NaN;
    testMSE = NaN;
    nnMetrics = table;
end

%% 8. TEXT SUMMARY FOR THE PORTFOLIO
summaryPath = fullfile(resultsDir,'portfolio_results_summary.txt');
fid = fopen(summaryPath,'w');

if fid < 0
    warning('Could not create summary text file.');
else
    fprintf(fid,'VERIFIED EULER-BERNOULLI BEAM PROJECT RESULTS\n');
    fprintf(fid,'=================================================\n\n');

    fprintf(fid,'REFERENCE MODEL\n');
    fprintf(fid,'L = %.3f m\n',L);
    fprintf(fid,'E = %.3f GPa\n',E/1e9);
    fprintf(fid,'I = %.6e m^4\n',I);
    fprintf(fid,'q = %.3f kN/m\n',q_ref/1e3);
    fprintf(fid,'P = %.3f kN\n',P_ref/1e3);
    fprintf(fid,'a/L = %.3f\n\n',a_ref/L);

    fprintf(fid,'ANALYTICAL VALIDATION\n');
    fprintf(fid,'FEM left reaction = %.6f kN\n',ref.reactions(1)/1e3);
    fprintf(fid,'Exact left reaction = %.6f kN\n',ana.RA/1e3);
    fprintf(fid,'FEM right reaction = %.6f kN\n',ref.reactions(2)/1e3);
    fprintf(fid,'Exact right reaction = %.6f kN\n',ana.RB/1e3);
    fprintf(fid,'FEM maximum deflection = %.6f mm\n',maxDefFEM*1e3);
    fprintf(fid,'Exact maximum deflection = %.6f mm\n',maxDefAna*1e3);
    fprintf(fid,'Deflection relative L2 error = %.6f %%\n',100*defRelL2);
    fprintf(fid,'Moment relative L2 error = %.6f %%\n',100*momRelL2);
    fprintf(fid,'Vertical equilibrium residual = %.6e N\n\n',equilibriumResidual);

    fprintf(fid,'MESH CONVERGENCE\n');
    fprintf(fid,'See mesh_convergence.csv and 05_mesh_convergence.png\n\n');

    if RUN_NEURAL_NETWORK
        fprintf(fid,'NEURAL-NETWORK METRICS\n');
        fprintf(fid,'Training MSE = %.6e\n',trainMSE);
        fprintf(fid,'Validation MSE = %.6e\n',valMSE);
        fprintf(fid,'Test MSE = %.6e\n\n',testMSE);

        for i = 1:height(nnMetrics)
            fprintf(fid,'%s: Test RMSE = %.6f %s, R^2 = %.6f, unseen relative L2 = %.6f %%\n', ...
                nnMetrics.Response(i),nnMetrics.TestRMSE(i), ...
                nnMetrics.RMSEUnit(i),nnMetrics.TestR2(i), ...
                nnMetrics.UnseenRelativeL2_percent(i));
        end
    end

    fclose(fid);
end

fprintf('\n============================================================\n');
fprintf('COMPLETED\n');
fprintf('Results exported to:\n%s\n',resultsDir);
fprintf('============================================================\n');

%% LOCAL FUNCTIONS

function sol = solveEulerBernoulliBeamVerified(L,E,I,nElem,qDown,PDown,a)
% Two-node Euler-Bernoulli beam FEM.
% The element load vector is stored separately for every element.
% Therefore, element end-force recovery correctly subtracts both the UDL
% and the consistent point-load vector in the loaded element.

    validateattributes(L,{'numeric'},{'scalar','positive'});
    validateattributes(E,{'numeric'},{'scalar','positive'});
    validateattributes(I,{'numeric'},{'scalar','positive'});
    validateattributes(nElem,{'numeric'},{'scalar','integer','>=',2});
    validateattributes(qDown,{'numeric'},{'scalar','nonnegative'});
    validateattributes(PDown,{'numeric'},{'scalar','nonnegative'});
    validateattributes(a,{'numeric'},{'scalar','>=',0,'<=',L});

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

    if PDown > 0
        if a == L
            pointElem = nElem;
            xi = 1;
        else
            pointElem = floor(a/Le)+1;
            pointElem = min(max(pointElem,1),nElem);
            xLeft = (pointElem-1)*Le;
            xi = (a-xLeft)/Le;
        end

        N = hermiteShapeFunctions(xi,Le);
        fPoint = (-PDown)*N;
        dofs = elementDOFs(pointElem);

        elementLoad(:,pointElem) = elementLoad(:,pointElem)+fPoint;
        F(dofs) = F(dofs)+fPoint;
    else
        pointElem = NaN;
    end

    fixedDOF = [1,2*nNode-1];
    freeDOF = setdiff(1:nDOF,fixedDOF);

    d = zeros(nDOF,1);
    d(freeDOF) = K(freeDOF,freeDOF)\F(freeDOF);

    reactionVector = K*d-F;
    reactions = [reactionVector(fixedDOF(1));reactionVector(fixedDOF(2))];

    elementEndForce = zeros(4,nElem);
    for e = 1:nElem
        dofs = elementDOFs(e);
        de = d(dofs);

        % Correct recovery: subtract the complete element load vector.
        elementEndForce(:,e) = ke*de-elementLoad(:,e);
    end

    sol.xNode = xNode;
    sol.d = d;
    sol.w = d(1:2:end);
    sol.theta = d(2:2:end);
    sol.reactions = reactions;
    sol.elementLoad = elementLoad;
    sol.elementEndForce = elementEndForce;
    sol.pointElement = pointElem;
end

function wEval = evaluateFEDisplacement(xNode,d,xEval)
% Cubic Hermite interpolation of FEM displacement.

    nElem = numel(xNode)-1;
    Le = xNode(2)-xNode(1);
    wEval = zeros(size(xEval));

    for k = 1:numel(xEval)
        x = xEval(k);

        if x >= xNode(end)
            e = nElem;
            xi = 1;
        else
            e = floor((x-xNode(1))/Le)+1;
            e = min(max(e,1),nElem);
            xi = (x-xNode(e))/Le;
        end

        N = hermiteShapeFunctions(xi,Le);
        de = d(elementDOFs(e));
        wEval(k) = N'*de;
    end
end

function MEval = evaluateFEBendingMoment(xNode,d,E,I,xEval)
% Bending moment recovered from FEM curvature:
% M = E I d^2w/dx^2, using the positive-sagging convention.

    nElem = numel(xNode)-1;
    Le = xNode(2)-xNode(1);
    MEval = zeros(size(xEval));

    for k = 1:numel(xEval)
        x = xEval(k);

        if x >= xNode(end)
            e = nElem;
            xi = 1;
        else
            e = floor((x-xNode(1))/Le)+1;
            e = min(max(e,1),nElem);
            xi = (x-xNode(e))/Le;
        end

        B2 = [(-6+12*xi)/Le^2;
              (-4+6*xi)/Le;
              ( 6-12*xi)/Le^2;
              (-2+6*xi)/Le];

        de = d(elementDOFs(e));
        MEval(k) = E*I*(B2'*de);
    end
end

function ana = analyticalSimplySupportedBeam(L,E,I,q,P,a,x)
% Closed-form solution for a simply supported prismatic beam under:
% - full-span UDL q
% - one point load P at x = a
%
% Downward deflection is negative.
% Bending moment uses positive sagging convention.

    b = L-a;
    RA = q*L/2 + P*b/L;
    RB = q*L/2 + P*a/L;

    wUDL = -q*x.*(L^3-2*L*x.^2+x.^3)/(24*E*I);

    wPoint = zeros(size(x));
    left = x <= a;
    right = ~left;

    wPoint(left) = -P*b*x(left).* ...
        (L^2-b^2-x(left).^2)/(6*L*E*I);

    xr = L-x(right);
    wPoint(right) = -P*a*xr.* ...
        (L^2-a^2-xr.^2)/(6*L*E*I);

    M = RA*x-q*x.^2/2-P*max(x-a,0);

    ana.w = wUDL+wPoint;
    ana.M = M;
    ana.RA = RA;
    ana.RB = RB;
end

function [M,V] = sectionForcesSimplySupported(L,q,P,a,x,side)
% Exact section-equilibrium internal forces.
% M is positive in sagging.
%
% At x=a, shear is discontinuous. side='left' returns the left limit;
% side='right' returns the right limit.

    RA = q*L/2 + P*(L-a)/L;

    if strcmpi(side,'left')
        H = double(x > a);
    else
        H = double(x >= a);
    end

    V = RA-q*x-P*H;
    M = RA*x-q*x.^2/2-P*max(x-a,0);
end

function [xPlot,MPlot,VPlot] = sectionForcePlotCoordinates(L,q,P,a)
% Repeats x=a twice so the point-load-induced shear jump is plotted
% vertically rather than being smeared across an element.

    xLeft = linspace(0,a,600)';
    xRight = linspace(a,L,600)';

    [MLeft,VLeft] = sectionForcesSimplySupported(L,q,P,a,xLeft,'left');
    [MRight,VRight] = sectionForcesSimplySupported(L,q,P,a,xRight,'right');

    xPlot = [xLeft;xRight];
    MPlot = [MLeft;MRight];
    VPlot = [VLeft;VRight];
end

function N = hermiteShapeFunctions(xi,Le)
    N1 = 1-3*xi^2+2*xi^3;
    N2 = Le*(xi-2*xi^2+xi^3);
    N3 = 3*xi^2-2*xi^3;
    N4 = Le*(-xi^2+xi^3);
    N = [N1;N2;N3;N4];
end

function dofs = elementDOFs(e)
    dofs = [2*e-1,2*e,2*(e+1)-1,2*(e+1)];
end

function err = relativeError(value,reference)
    err = abs(value-reference)/max(abs(reference),eps);
end

function indices = caseNumbersToSampleIndices(caseNumbers,nNode)
    indices = zeros(1,numel(caseNumbers)*nNode);
    cursor = 1;

    for k = 1:numel(caseNumbers)
        firstIndex = (caseNumbers(k)-1)*nNode+1;
        block = firstIndex:(firstIndex+nNode-1);
        indices(cursor:cursor+nNode-1) = block;
        cursor = cursor+nNode;
    end
end

function r2 = coefficientOfDetermination(yTrue,yPred)
    yTrue = yTrue(:);
    yPred = yPred(:);

    ssRes = sum((yTrue-yPred).^2);
    ssTot = sum((yTrue-mean(yTrue)).^2);

    if ssTot <= eps
        r2 = NaN;
    else
        r2 = 1-ssRes/ssTot;
    end
end

function saveFigure(fig,filePath,dpi)
% Export a clean figure. Uses exportgraphics when available.

    try
        exportgraphics(fig,filePath,'Resolution',dpi);
    catch
        print(fig,filePath,'-dpng',sprintf('-r%d',dpi));
    end
end
