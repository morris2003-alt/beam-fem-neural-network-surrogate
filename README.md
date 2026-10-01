# Computational beam FEM and physics-guided surrogate

MATLAB implementation of a simply supported Euler–Bernoulli beam finite-element model and a physics-guided neural-network surrogate for deflection, bending moment, and shear force. The beam has a full-span uniformly distributed load and one downward point load.

The [project portfolio](docs/Computational_Beam_Project_Portfolio.pdf.pdf) presents the beam model, dataset, neural-network design, prediction results, and training performance.

## Run

Open MATLAB, set **Current Folder** to this repository, and run:

```matlab
beam_nn_project_verified      % FEM verification, mesh study, and initial NN baseline
beam_schemeB_main             % final Scheme B surrogate and comparison to baseline
```

Run the scripts in this order to produce `schemeB_vs_baseline.csv`. Scheme B can also run on its own; it skips that comparison if the baseline CSV is absent. The baseline has a `RUN_NEURAL_NETWORK` switch near the top: set it to `false` for FEM verification only, but then no baseline NN metrics are generated. Both files have a `SHOW_TRAINING_GUI` option. Training four networks can take time and memory; leave the numerical settings unchanged when attempting to compare with the portfolio.

Requires MATLAB with Deep Learning Toolbox (formerly Neural Network Toolbox) for `fitnet`, `train`, and `plotperform`. Both MATLAB files include a `SHOW_TRAINING_GUI` option to control the training windows.

## Files and outputs

| File | Role |
| --- | --- |
| `beam_nn_project_verified.m` | Baseline script: FEM, closed-form validation, mesh convergence, and initial multi-output NN. |
| `beam_schemeB_main.m` | Final Scheme B function: four response-specific networks and an explicitly reserved unseen case. |
| `docs/Computational_Beam_Project_Portfolio.pdf.pdf` | Project portfolio: model, dataset, network design, prediction results, and training performance. |

Each entry point writes to its own generated directory beside the source: `beam_project_results/` or `beam_project_schemeB_results/`. These directories contain validation or test metrics (`.csv`), figures (`.png`), a text summary, and trained models (`.mat`). In particular, inspect `validation_summary.csv`, `mesh_convergence.csv`, `nn_metrics.csv`, `schemeB_test_metrics.csv`, `schemeB_unseen_metrics.csv`, and `schemeB_constraint_checks.csv`. Scheme B writes `schemeB_vs_baseline.csv` only when it finds baseline metrics.

## Model and evaluation

- Fixed geometry and material: length 6 m, `E = 200 GPa`, rectangular section `b = 0.15 m`, `h = 0.30 m`, and 40 two-node cubic Hermite beam elements in the reference model.
- Parametric data: 350 complete load cases, 41 nodes per case, with `q = 2–15 kN/m`, `P = 5–45 kN`, and `a/L = 0.15–0.85`. The split operates on entire loading cases (245 train, 53 validation, 52 held-out test), avoiding shared points from the same load case across partitions.
- The baseline validates displacement and bending moment against a closed-form solution and checks mesh convergence. Moment and shear response targets use section equilibrium, retaining the point-load jump.
- Scheme B uses separate deflection and moment networks with `xi(1-xi)` output transformations (`xi = x/L`) to satisfy simple-support values. Separate left and right shear networks represent the discontinuity at the point load.
- The reserved unseen loading case uses `q = 11.3 kN/m`, `P = 31.7 kN`, and `a/L = 0.630`; the script checks that it was not duplicated in the randomly generated cases.

The hard transforms enforce the support values for deflection and moment, while the shear jump is evaluated and reported rather than imposed exactly. The study uses deterministic simulated data and the stated load, boundary, geometry, and material family. Extrapolation, experimental noise, and other support or load types were not evaluated.

## Results

The following results are reported in the project portfolio.

### Held-out test set: 52 loading cases

| Response | R² | RMSE |
| --- | --- | --- |
| Deflection | 0.999915 | 0.013080 mm |
| Bending moment | 0.999559 | 0.552811 kN·m |
| Shear force | 0.999987 | 0.099024 kN |

### Additional loading case

This case uses q = 11.3 kN/m, P = 31.7 kN, and a/L = 0.630. It lies within the sampled loading ranges and is excluded from the 350-case dataset.

| Response | Relative L2 error |
| --- | --- |
| Deflection | 0.247% |
| Bending moment | 1.086% |
| Shear force | 0.182% |

The largest relative L2 error among the three responses for this case is 1.086%, for bending moment. The shear-jump error is 0.084%.

Results may vary with MATLAB version, toolbox version, and numerical environment.
