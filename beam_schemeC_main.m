function beam_schemeC_main()
% BEAM_SCHEMEC_MAIN
% Scheme C: a physics-structured surrogate that builds linear superposition
% into the model, for the same simply supported Euler-Bernoulli beam as
% Schemes A (baseline) and B:
%   (1) a full-span uniformly distributed load q, and
%   (2) one downward point load P at x = a.
%
% IDEA
%   For a linear elastic beam the deflection is linear in the loads:
%
%       w(x; q, P, a) = q * f(x) + P * g(x, a)
%
%   f(x)    = deflection per unit UDL      (one FEM solve, no network needed)
%   g(x, a) = deflection per unit point load at x = a  (the deflection
%             influence surface; this is the only part a network learns)
%
%   g is learned with the same hard support constraint used in Scheme B:
%       g_hat = xi(1-xi) * (L^3/EI) * N_g(xi, alpha, r, |r|)
%
%   The beam is statically determinate, so the bending moment and shear force
%   follow exactly from section equilibrium once the loads are known. Scheme C
%   therefore uses equilibrium for M and V (including the exact moment kink and
%   shear jump at x = a) and uses the network only where the FE solution is
%   actually needed: the deflection.
%
% WHAT IS COMPARED
%   1. Accuracy on Scheme B's 52 held-out test cases and the reserved case
%      (the same 350 cases and split are regenerated with Scheme B's seed).
%   2. Data efficiency: accuracy versus the number of FEM solves.
%   3. Extrapolation in load magnitude (outside the training ranges), where
%      superposition should keep Scheme C accurate.
%   4. Extrapolation in load position (a/L outside 0.15-0.85), where neither
%      scheme is expected to be reliable.
%
% This is a physics-structured data-driven surrogate, NOT a PINN.
%
% Run:  beam_schemeC_main
% Needs: Deep Learning Toolbox. For the comparison with Scheme B, run
%        beam_schemeB_main first (it saves beam_schemeB_models.mat).
% Output folder: beam_project_schemeC_results

projectFolder = fileparts(mfilename('fullpath'));
clc;
close all;

%% USER OPTIONS
SHOW_TRAINING_GUI = false;
EXPORT_DPI = 300;
TRAIN_ALGORITHM = 'trainscg';
MAX_EPOCHS = 3000;
MAX_VALIDATION_FAILURES = 100;
MIN_GRADIENT = 1e-10;
hiddenG = [32, 32];
nAlphaList = [5, 10, 20, 40];     % number of point-load positions used for training
nAlphaFinal = 20;                 % model used for the detailed comparison
SEEDS = [1, 2, 3];                % random initial weights for each setting

resultsDir = fullfile(projectFolder, 'beam_project_schemeC_results');
if ~isfolder(resultsDir)
    mkdir(resultsDir);
end
% files written by an earlier single-seed version of this script
staleFiles = {'01_schemeC_data_efficiency.png', 'schemeC_data_efficiency.csv'};
for k = 1:numel(staleFiles)
    f = fullfile(resultsDir, staleFiles{k});
    if isfile(f)
        delete(f);
    end
end

fprintf('\n============================================================\n');
fprintf('SCHEME C: SUPERPOSITION-STRUCTURED BEAM SURROGATE\n');
fprintf('============================================================\n');

%% STRUCTURAL MODEL (identical to Scheme B)
L = 6.0;  E = 200e9;  b = 0.15;  h = 0.30;  I = b*h^3/12;  EI = E*I;
nElem = 40;  nNode = nElem + 1;
xNode = linspace(0, L, nNode)';
xiNode = xNode/L;
phiNode = xiNode.*(1 - xiNode);
interior = phiNode > 1e-12;

qRange = [2e3, 15e3];
PRange = [5e3, 45e3];
aRange = [0.15*L, 0.85*L];
gScale = L^3/EI;                  % deflection scale per unit point load [m/N]

%% 1. REGENERATE SCHEME B'S DATASET AND SPLIT
% Same seed and the same sequence of random calls as beam_schemeB_main.
rng(42, 'twister');
nCases = 350;
caseParameters = zeros(nCases, 3);
for c = 1:nCases
    q = qRange(1) + rand*(qRange(2)-qRange(1));
    P = PRange(1) + rand*(PRange(2)-PRange(1));
    a = aRange(1) + rand*(aRange(2)-aRange(1));
    caseParameters(c,:) = [q, P, a];
end
caseOrder = randperm(nCases);
nTrainCases = round(0.70*nCases);
nValCases = round(0.15*nCases);
testCases = caseOrder(nTrainCases+nValCases+1:end);
fprintf('Regenerated Scheme B split: %d test cases.\n', numel(testCases));

% FEM reference for the test cases
nTest = numel(testCases);
wTest = zeros(nNode, nTest);
for k = 1:nTest
    p = caseParameters(testCases(k),:);
    s = solveEulerBernoulliBeam(L, E, I, nElem, p(1), p(2), p(3));
    wTest(:,k) = s.w;
end

%% 2. UNIT-UDL SOLUTION f(x): one FEM solve
solQ = solveEulerBernoulliBeam(L, E, I, nElem, 1.0, 0.0, L/2);
fW = solQ.w;                      % deflection per unit q [m per N/m]

% Superposition check: q*f + P*g must equal the full FEM solution
pChk = caseParameters(testCases(1),:);
sFull = solveEulerBernoulliBeam(L, E, I, nElem, pChk(1), pChk(2), pChk(3));
sUnitP = solveEulerBernoulliBeam(L, E, I, nElem, 0.0, 1.0, pChk(3));
supErr = max(abs(pChk(1)*fW + pChk(2)*sUnitP.w - sFull.w))/max(abs(sFull.w));
fprintf('Superposition check (should be round-off): %.2e\n', supErr);

%% 3. RESERVED CASE AND EXTRAPOLATION CASES
evalNames = ["Reserved case (in range)"; ...
             "P = 60 kN (above range)"; ...
             "q = 20 kN/m, P = 80 kN (above range)"; ...
             "a/L = 0.05 (outside position range)"; ...
             "a/L = 0.95 (outside position range)"];
evalLoads = [11.3e3, 31.7e3, 0.63*L;
             11.3e3, 60.0e3, 0.63*L;
             20.0e3, 80.0e3, 0.63*L;
             11.3e3, 31.7e3, 0.05*L;
             11.3e3, 31.7e3, 0.95*L];
nEval = size(evalLoads, 1);
wEvalTrue = zeros(nNode, nEval);
for k = 1:nEval
    s = solveEulerBernoulliBeam(L, E, I, nElem, evalLoads(k,1), evalLoads(k,2), evalLoads(k,3));
    wEvalTrue(:,k) = s.w;
end

%% 4. DATA-EFFICIENCY STUDY: Scheme C vs the Scheme B architecture
% Both surrogates get the same number of FEM solves, and each setting is
% trained with several random initial weights.
%   Scheme C:  nA positions -> 1 (unit UDL) + nA (training) + nA-1 (validation)
%              = 2*nA FEM solves
%   Scheme B architecture (deflection network only, same inputs, hard support
%   constraint and size as Scheme B): trained on the same number of complete
%   load cases, taken from Scheme B's training/validation pool (never from
%   its test cases).
budgets = 2*nAlphaList;                      % FEM solves: 10, 20, 40, 80
poolCases = caseOrder(1:nTrainCases+nValCases);
qScale = qRange(2);  PScale = PRange(2);
wScaleB = qScale*L^4/EI + PScale*L^3/EI;     % same deflection scale as Scheme B
hiddenB = [64, 64, 32];                      % same size as Scheme B's deflection network

nRuns = numel(budgets)*numel(SEEDS)*2;
effTable = table('Size', [nRuns, 7], ...
    'VariableTypes', {'string','double','double','double','double','double','string'}, ...
    'VariableNames', {'Method','FEMSolves','Seed','TestR2_Deflection', ...
    'TestRMSE_Deflection_mm','ReservedL2_Deflection_percent','StopReason'});
modelsC = cell(numel(budgets), numel(SEEDS));
row = 0;

for s = 1:numel(budgets)
    nA = nAlphaList(s);
    alphaTrain = linspace(aRange(1)/L, aRange(2)/L, nA);
    alphaVal = (alphaTrain(1:end-1) + alphaTrain(2:end))/2;   % positions between the training ones
    [XC, TC, iTrC, iVaC] = influenceDataset(L, E, I, nElem, xiNode, phiNode, interior, ...
                                            gScale, alphaTrain, alphaVal);
    nB = budgets(s);
    casesB = poolCases(1:nB);
    nBval = max(1, round(0.15*nB));
    [XB, TB, iTrB, iVaB] = schemeBDataset(L, E, I, nElem, xiNode, phiNode, interior, ...
        caseParameters(casesB,:), qScale, PScale, wScaleB, nB - nBval);

    for k = 1:numel(SEEDS)
        seed = SEEDS(k);

        % --- Scheme C
        rng(seed, 'twister');
        fprintf('\nScheme C: %d positions (%d FEM solves), seed %d\n', nA, budgets(s), seed);
        [netG, trG] = trainRegressionNetwork(XC, TC, iTrC, iVaC, hiddenG, TRAIN_ALGORITHM, ...
            MAX_EPOCHS, MAX_VALIDATION_FAILURES, MIN_GRADIENT, SHOW_TRAINING_GUI);
        model = struct('netG', netG, 'fW', fW, 'L', L, 'gScale', gScale, 'xiNode', xiNode, 'tr', trG);
        modelsC{s,k} = model;
        predictFn = @(q, P, a) predictDeflectionC(model, q, P, a);
        row = row + 1;
        effTable(row,:) = evaluateRun("Scheme C", budgets(s), seed, predictFn, caseParameters, ...
            testCases, wTest, evalLoads(1,:), wEvalTrue(:,1), string(trG.stop));

        % --- Scheme B architecture with the same number of FEM solves
        rng(seed, 'twister');
        fprintf('Scheme B architecture: %d load cases (%d FEM solves), seed %d\n', nB, nB, seed);
        [netB, trB] = trainRegressionNetwork(XB, TB, iTrB, iVaB, hiddenB, TRAIN_ALGORITHM, ...
            MAX_EPOCHS, MAX_VALIDATION_FAILURES, MIN_GRADIENT, SHOW_TRAINING_GUI);
        predictFn = @(q, P, a) predictDeflectionBArch(netB, xiNode, L, q, P, a, qScale, PScale, wScaleB);
        row = row + 1;
        effTable(row,:) = evaluateRun("Scheme B architecture", nB, seed, predictFn, caseParameters, ...
            testCases, wTest, evalLoads(1,:), wEvalTrue(:,1), string(trB.stop));
    end
end

fprintf('\nDATA EFFICIENCY, ALL RUNS (deflection)\n');
disp(effTable);
writetable(effTable, fullfile(resultsDir, 'schemeC_data_efficiency_runs.csv'));

% mean, min and max over the seeds
methodNames = ["Scheme C"; "Scheme B architecture"];
effSummary = table('Size', [numel(methodNames)*numel(budgets), 8], ...
    'VariableTypes', {'string','double','double','double','double','double','double','double'}, ...
    'VariableNames', {'Method','FEMSolves','ReservedL2_mean_percent','ReservedL2_min_percent', ...
    'ReservedL2_max_percent','TestR2_mean','TestR2_min','TestR2_max'});
row = 0;
for m = 1:numel(methodNames)
    for s = 1:numel(budgets)
        sel = effTable.Method == methodNames(m) & effTable.FEMSolves == budgets(s);
        row = row + 1;
        e = effTable.ReservedL2_Deflection_percent(sel);
        r2 = effTable.TestR2_Deflection(sel);
        effSummary(row,:) = {methodNames(m), budgets(s), mean(e), min(e), max(e), mean(r2), min(r2), max(r2)};
    end
end
fprintf('\nDATA EFFICIENCY SUMMARY (%d seeds per setting)\n', numel(SEEDS));
disp(effSummary);
writetable(effSummary, fullfile(resultsDir, 'schemeC_data_efficiency_summary.csv'));

%% 5. DETAILED COMPARISON WITH SCHEME B (final Scheme C model)
final = modelsC{nAlphaList == nAlphaFinal, 1};      % first seed

schemeBFile = fullfile(projectFolder, 'beam_project_schemeB_results', 'beam_schemeB_models.mat');
haveB = isfile(schemeBFile);
if haveB
    SB = load(schemeBFile, 'models', 'caseParameters', 'testCases');
    modelsB = SB.models;
    fprintf('\nLoaded Scheme B models from %s\n', schemeBFile);
    % The regenerated dataset and split must be identical to Scheme B's
    sameCases = isequal(size(SB.caseParameters), size(caseParameters)) && ...
                max(abs(SB.caseParameters(:) - caseParameters(:))) == 0;
    sameSplit = isequal(sort(SB.testCases(:)), sort(testCases(:)));
    fprintf('Same 350 load cases as Scheme B: %d;  same test split: %d\n', sameCases, sameSplit);
    if ~(sameCases && sameSplit)
        warning('Dataset or split differs from Scheme B; the comparison is not like-for-like.');
    end
else
    fprintf('\nScheme B models not found (%s); Scheme B columns will be NaN.\n', schemeBFile);
end

cmp = table('Size', [nEval, 7], ...
    'VariableTypes', {'string','double','double','double','double','double','double'}, ...
    'VariableNames', {'Case','q_kN_per_m','P_kN','a_over_L', ...
    'SchemeB_DeflectionL2_percent','SchemeC_DeflectionL2_percent','SchemeB_MomentL2_percent'});
wEvalC = zeros(nNode, nEval);
wEvalB = nan(nNode, nEval);
for k = 1:nEval
    q = evalLoads(k,1); P = evalLoads(k,2); a = evalLoads(k,3);
    wEvalC(:,k) = predictDeflectionC(final, q, P, a);
    cmp.Case(k) = evalNames(k);
    cmp.q_kN_per_m(k) = q/1e3; cmp.P_kN(k) = P/1e3; cmp.a_over_L(k) = a/L;
    cmp.SchemeC_DeflectionL2_percent(k) = 100*relativeL2(wEvalTrue(:,k), wEvalC(:,k));
    if haveB
        [wB, MB, ~] = predictSchemeB(modelsB, xNode, q, P, a);
        wEvalB(:,k) = wB;
        [MTrue, ~] = sectionForces(L, q, P, a, xNode, 'right');
        cmp.SchemeB_DeflectionL2_percent(k) = 100*relativeL2(wEvalTrue(:,k), wB);
        cmp.SchemeB_MomentL2_percent(k) = 100*relativeL2(MTrue, MB);
    else
        cmp.SchemeB_DeflectionL2_percent(k) = NaN;
        cmp.SchemeB_MomentL2_percent(k) = NaN;
    end
end
fprintf('\nRESERVED AND EXTRAPOLATION CASES (relative L2 error, %%)\n');
fprintf('Scheme C moment and shear come from section equilibrium, so their error is zero by construction.\n');
disp(cmp);
writetable(cmp, fullfile(resultsDir, 'schemeC_vs_schemeB.csv'));

%% 6. FIGURES
% (a) data efficiency: mean over seeds, bars = min to max
fig = figure('Name', 'Data efficiency');
mk = {'-o', '-s'};
for m = 1:numel(methodNames)
    sel = effSummary.Method == methodNames(m);
    x = effSummary.FEMSolves(sel);
    y = effSummary.ReservedL2_mean_percent(sel);
    lo = y - effSummary.ReservedL2_min_percent(sel);
    hi = effSummary.ReservedL2_max_percent(sel) - y;
    errorbar(x, y, lo, hi, mk{m}, 'LineWidth', 1.8, 'MarkerSize', 7);
    hold on;
end
if haveB
    resB = cmp.SchemeB_DeflectionL2_percent(1);
    plot(350, resB, 'p', 'MarkerSize', 14, 'Color', [0.47 0.67 0.19], ...
         'MarkerFaceColor', [0.47 0.67 0.19]);
    legendText = [methodNames; "Scheme B (350 load cases)"];
else
    legendText = methodNames;
end
set(gca, 'XScale', 'log', 'YScale', 'log');
xlim([7, 600]);
grid on;
xlabel('Number of FEM solves used to build the surrogate');
ylabel('Reserved-case deflection error, relative L_2 [%]');
title(sprintf('Accuracy versus training data (%d seeds; bars = min to max)', numel(SEEDS)));
legend(legendText, 'Location', 'best');
saveFigure(fig, fullfile(resultsDir, '01_data_efficiency.png'), EXPORT_DPI);

% (b) extrapolation in load magnitude: deflection for q = 20 kN/m, P = 80 kN
k = 3;
fig = figure('Name', 'Extrapolation in load magnitude');
plot(xNode, wEvalTrue(:,k)*1e3, '-', 'LineWidth', 2.0);
hold on;
plot(xNode, wEvalC(:,k)*1e3, '--', 'LineWidth', 1.8);
if haveB
    plot(xNode, wEvalB(:,k)*1e3, ':', 'LineWidth', 2.0);
    legend('FEM reference', 'Scheme C (superposition)', 'Scheme B', 'Location', 'best');
else
    legend('FEM reference', 'Scheme C (superposition)', 'Location', 'best');
end
grid on;
xlabel('Position x [m]');
ylabel('Deflection w [mm]');
title('Outside the training range: q = 20 kN/m, P = 80 kN, a/L = 0.63');
saveFigure(fig, fullfile(resultsDir, '02_extrapolation_load_magnitude.png'), EXPORT_DPI);

% (c) bending moment at the reserved case: Scheme B vs exact equilibrium
q = evalLoads(1,1); P = evalLoads(1,2); a = evalLoads(1,3);
xDense = sort(unique([linspace(0, L, 601)'; a]));
[MExact, ~] = sectionForces(L, q, P, a, xDense, 'right');
fig = figure('Name', 'Moment kink at the point load');
plot(xDense, MExact/1e3, '-', 'LineWidth', 2.0);
hold on;
if haveB
    [~, MBd, ~] = predictSchemeB(modelsB, xDense, q, P, a);
    plot(xDense, MBd/1e3, '--', 'LineWidth', 1.8);
    legend('Section equilibrium (used by Scheme C)', 'Scheme B network', 'Location', 'best');
else
    legend('Section equilibrium (used by Scheme C)', 'Location', 'best');
end
xline(a, ':', 'HandleVisibility', 'off');
grid on;
xlabel('Position x [m]');
ylabel('Sagging bending moment M [kN m]');
title('Bending moment at the reserved case (kink at x = a)');
saveFigure(fig, fullfile(resultsDir, '03_moment_kink.png'), EXPORT_DPI);

% (d) training performance of the final network
fig = figure('Name', 'Influence-surface network training');
plotperform(final.tr);
saveFigure(fig, fullfile(resultsDir, '04_training_performance_influence_network.png'), EXPORT_DPI);

%% 7. SAVE AND SUMMARIZE
save(fullfile(resultsDir, 'beam_schemeC_models.mat'), 'modelsC', 'nAlphaList', 'nAlphaFinal', 'SEEDS', ...
    'effTable', 'effSummary', 'cmp', 'supErr', 'L', 'E', 'I', 'nElem', 'hiddenG', 'TRAIN_ALGORITHM');

fid = fopen(fullfile(resultsDir, 'schemeC_results_summary.txt'), 'w');
if fid < 0
    warning('Could not create schemeC_results_summary.txt');
else
    fprintf(fid, 'SCHEME C: SUPERPOSITION-STRUCTURED BEAM SURROGATE\n');
    fprintf(fid, '=================================================\n\n');
    fprintf(fid, 'w(x; q, P, a) = q f(x) + P g(x, a); only g is learned (hard support constraint).\n');
    fprintf(fid, 'M and V from section equilibrium (statically determinate beam).\n');
    fprintf(fid, 'This is not a PINN; no governing-equation residual is used in the loss.\n');
    fprintf(fid, 'Superposition check (relative): %.3e\n\n', supErr);
    fprintf(fid, 'DATA EFFICIENCY (deflection on Scheme B''s test cases and reserved case, %d seeds)\n', numel(SEEDS));
    for r = 1:height(effSummary)
        fprintf(fid, '%-22s %3d FEM solves: reserved L2 mean %.4f %% (min %.4f, max %.4f), test R^2 mean %.6f (min %.6f)\n', ...
            effSummary.Method(r), effSummary.FEMSolves(r), effSummary.ReservedL2_mean_percent(r), ...
            effSummary.ReservedL2_min_percent(r), effSummary.ReservedL2_max_percent(r), ...
            effSummary.TestR2_mean(r), effSummary.TestR2_min(r));
    end
    fprintf(fid, '\nRESERVED AND EXTRAPOLATION CASES (Scheme C: %d positions, seed %d; Scheme B: full model, 350 cases)\n', nAlphaFinal, SEEDS(1));
    for k = 1:height(cmp)
        fprintf(fid, '%s: Scheme C deflection L2 = %.4f %%, Scheme B deflection L2 = %.4f %%, Scheme B moment L2 = %.4f %%\n', ...
            cmp.Case(k), cmp.SchemeC_DeflectionL2_percent(k), cmp.SchemeB_DeflectionL2_percent(k), ...
            cmp.SchemeB_MomentL2_percent(k));
    end
    fclose(fid);
end

fprintf('\n============================================================\n');
fprintf('SCHEME C COMPLETED\nResults exported to:\n%s\n', resultsDir);
fprintf('============================================================\n');
end

%% ========================================================================
function [X, T, iTrain, iVal] = influenceDataset(L, E, I, nElem, xiNode, phiNode, interior, ...
                                                  gScale, alphaTrain, alphaVal)
% Unit point-load FEM solutions at the given positions, as samples of the
% latent influence function N_g = g / (gScale * xi(1-xi)).
alphaAll = [alphaTrain(:); alphaVal(:)];
isVal = [false(numel(alphaTrain),1); true(numel(alphaVal),1)];
nIn = nnz(interior);
X = zeros(4, nIn*numel(alphaAll));
T = zeros(1, nIn*numel(alphaAll));
valMask = false(1, nIn*numel(alphaAll));
xi = xiNode(interior);
cursor = 1;
for k = 1:numel(alphaAll)
    alpha = alphaAll(k);
    s = solveEulerBernoulliBeam(L, E, I, nElem, 0.0, 1.0, alpha*L);
    g = s.w(interior);
    r = xi - alpha;
    idx = cursor:(cursor+nIn-1);
    X(:,idx) = [xi'; alpha*ones(1,nIn); r'; abs(r)'];
    T(idx) = (g ./ (gScale*phiNode(interior)))';
    valMask(idx) = isVal(k);
    cursor = cursor + nIn;
end
iTrain = find(~valMask);
iVal = find(valMask);
end

%% ========================================================================
function [X, T, iTrain, iVal] = schemeBDataset(L, E, I, nElem, xiNode, phiNode, interior, ...
                                               params, qScale, PScale, wScale, nTrain)
% Scheme B's deflection inputs and hard-constrained target for the given load
% cases; the first nTrain cases are used for training, the rest for validation.
nIn = nnz(interior);
nC = size(params, 1);
X = zeros(6, nIn*nC);
T = zeros(1, nIn*nC);
isVal = false(1, nIn*nC);
xi = xiNode(interior);
for c = 1:nC
    q = params(c,1); P = params(c,2); a = params(c,3);
    s = solveEulerBernoulliBeam(L, E, I, nElem, q, P, a);
    alpha = a/L;
    r = xi - alpha;
    idx = (c-1)*nIn + (1:nIn);
    X(:,idx) = [xi'; (q/qScale)*ones(1,nIn); (P/PScale)*ones(1,nIn); alpha*ones(1,nIn); r'; abs(r)'];
    T(idx) = (s.w(interior) ./ (wScale*phiNode(interior)))';
    isVal(idx) = c > nTrain;
end
iTrain = find(~isVal);
iVal = find(isVal);
end

%% ========================================================================
function w = predictDeflectionBArch(net, xiNode, L, q, P, a, qScale, PScale, wScale)
xi = xiNode;
alpha = a/L;
r = xi - alpha;
n = numel(xi);
X = [xi'; (q/qScale)*ones(1,n); (P/PScale)*ones(1,n); alpha*ones(1,n); r'; abs(r)'];
w = (xi.*(1-xi)) .* net(X)' * wScale;
w([1 end]) = 0;
end

%% ========================================================================
function row = evaluateRun(method, femSolves, seed, predictFn, caseParameters, testCases, ...
                           wTest, reservedLoad, wReservedTrue, stopReason)
nTest = numel(testCases);
wPred = zeros(size(wTest));
for k = 1:nTest
    p = caseParameters(testCases(k),:);
    wPred(:,k) = predictFn(p(1), p(2), p(3));
end
wRes = predictFn(reservedLoad(1), reservedLoad(2), reservedLoad(3));
row = table(method, femSolves, seed, coefficientOfDetermination(wTest(:), wPred(:)), ...
    sqrt(mean((wPred(:) - wTest(:)).^2))*1e3, 100*relativeL2(wReservedTrue, wRes), stopReason, ...
    'VariableNames', {'Method','FEMSolves','Seed','TestR2_Deflection', ...
    'TestRMSE_Deflection_mm','ReservedL2_Deflection_percent','StopReason'});
end

%% ========================================================================
function w = predictDeflectionC(model, q, P, a)
xi = model.xiNode;
alpha = a/model.L;
r = xi - alpha;
X = [xi'; alpha*ones(1,numel(xi)); r'; abs(r)'];
g = (xi.*(1-xi)) .* model.netG(X)' * model.gScale;
w = q*model.fW + P*g;
w([1 end]) = 0;                    % exact support conditions
end

%% ========================================================================
function [net, tr] = trainRegressionNetwork(X, T, trainInd, valInd, hidden, algorithm, ...
                                            maxEpochs, maxFail, minGrad, showGUI)
net = fitnet(hidden, algorithm);
net.performFcn = 'mse';
net.divideFcn = 'divideind';
net.divideParam.trainInd = trainInd(:)';
net.divideParam.valInd = valInd(:)';
net.divideParam.testInd = [];
net.trainParam.epochs = maxEpochs;
net.trainParam.max_fail = maxFail;
net.trainParam.min_grad = minGrad;
net.trainParam.showWindow = showGUI;
net.inputs{1}.processFcns = {'removeconstantrows', 'mapminmax'};
net.outputs{end}.processFcns = {'removeconstantrows', 'mapminmax'};
[net, tr] = train(net, X, T);
end

%% ========================================================================
function [wPred, MPred, VPred] = predictSchemeB(models, x, q, P, a)
% Same prediction function as beam_schemeB_main.m
x = x(:);
L = models.L;
xi = x/L;
alpha = a/L;
r = xi - alpha;
phi = xi.*(1 - xi);
X = [xi'; (q/models.qScale)*ones(1,numel(x)); (P/models.PScale)*ones(1,numel(x)); ...
     alpha*ones(1,numel(x)); r'; abs(r)'];
gW = models.netW(X)';
gM = models.netM(X)';
wPred = phi.*gW*models.wScale;
MPred = phi.*gM*models.MScale;
left = xi < alpha;
VPred = zeros(size(x));
if any(left)
    VPred(left) = models.netVL(X(:,left))'*models.VScale;
end
if any(~left)
    VPred(~left) = models.netVR(X(:,~left))'*models.VScale;
end
boundary = xi <= 1e-14 | xi >= 1 - 1e-14;
wPred(boundary) = 0;
MPred(boundary) = 0;
end

%% ========================================================================
function sol = solveEulerBernoulliBeam(L, E, I, nElem, qDown, PDown, a)
% Same two-node Hermite beam FEM as beam_schemeB_main.m
nNode = nElem + 1;
nDOF = 2*nNode;
Le = L/nElem;
xNode = linspace(0, L, nNode)';
K = zeros(nDOF, nDOF);
F = zeros(nDOF, 1);
q = -qDown;
ke = (E*I/Le^3) * [12, 6*Le, -12, 6*Le;
                   6*Le, 4*Le^2, -6*Le, 2*Le^2;
                   -12, -6*Le, 12, -6*Le;
                   6*Le, 2*Le^2, -6*Le, 4*Le^2];
feUDL = q*Le/2*[1; Le/6; 1; -Le/6];
for e = 1:nElem
    dofs = elementDOFs(e);
    K(dofs,dofs) = K(dofs,dofs) + ke;
    F(dofs) = F(dofs) + feUDL;
end
if PDown > 0
    if a == L
        pointElem = nElem;
        eta = 1;
    else
        pointElem = floor(a/Le) + 1;
        pointElem = min(max(pointElem, 1), nElem);
        eta = (a - (pointElem-1)*Le)/Le;
    end
    N = hermiteShapeFunctions(eta, Le);
    dofs = elementDOFs(pointElem);
    F(dofs) = F(dofs) + (-PDown)*N;
end
fixedDOF = [1, 2*nNode-1];
freeDOF = setdiff(1:nDOF, fixedDOF);
d = zeros(nDOF, 1);
d(freeDOF) = K(freeDOF,freeDOF) \ F(freeDOF);
sol.xNode = xNode;
sol.d = d;
sol.w = d(1:2:end);
end

%% ========================================================================
function [M, V] = sectionForces(L, q, P, a, x, side)
x = x(:);
RA = q*L/2 + P*(L-a)/L;
if strcmpi(side, 'left')
    H = double(x > a);
else
    H = double(x >= a);
end
V = RA - q*x - P*H;
M = RA*x - q*x.^2/2 - P*max(x-a, 0);
end

%% ========================================================================
function N = hermiteShapeFunctions(eta, Le)
N = [1 - 3*eta^2 + 2*eta^3;
     Le*(eta - 2*eta^2 + eta^3);
     3*eta^2 - 2*eta^3;
     Le*(-eta^2 + eta^3)];
end

function dofs = elementDOFs(e)
dofs = [2*e-1, 2*e, 2*(e+1)-1, 2*(e+1)];
end

function value = coefficientOfDetermination(yTrue, yPred)
yTrue = yTrue(:); yPred = yPred(:);
ssRes = sum((yTrue - yPred).^2);
ssTot = sum((yTrue - mean(yTrue)).^2);
if ssTot <= eps
    value = NaN;
else
    value = 1 - ssRes/ssTot;
end
end

function value = relativeL2(yTrue, yPred)
value = norm(yPred(:) - yTrue(:)) / max(norm(yTrue(:)), eps);
end

function saveFigure(fig, filePath, dpi)
try
    exportgraphics(fig, filePath, 'Resolution', dpi);
catch
    print(fig, filePath, '-dpng', sprintf('-r%d', dpi));
end
end
