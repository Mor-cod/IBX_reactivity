# Machine Learning Analysis of Reactivity and Substituent Effects in Hypervalent Iodine(V) Compounds

## Overview

This repository contains Python code and computational data used to investigate the hypervalent twisting transition-state (TS) barrier and substituent effects in IBX (2-iodoxybenzoic acid) derivatives, using machine learning models trained on DFT-derived molecular descriptors. Five regression models — Linear Regression, Random Forest, Gradient Boosting, Support Vector Regression, and a Neural Network (MLP) — are trained and evaluated to predict the twisting TS barrier, followed by hyperparameter tuning, residual diagnostics, dimensionality reduction, feature importance, Partial Dependence Plots (PDP), and SHAP analysis.

## Research context

This work was conducted in the Chemistry discipline at the University of Tasmania (UTAS), Hobart, Australia.

## Dataset

**The full DFT descriptor dataset and row-level prediction outputs are not currently distributed in this public repository and will be made available upon publication. The repository currently provides the analysis code and aggregate results required to document the computational workflow.**

For reference, the dataset (when present at `data/DFT_data.csv`) consists of **284 IBX derivatives** with eight DFT-derived descriptors and the target TS barrier — one row per compound:

- **Target variable**: `TS_Barrier` — the DFT-computed transition-state barrier (kcal·mol⁻¹) for the hypervalent twisting step in methanol oxidation.
- **Descriptors**:
  - `HOMO-O (Hartree)`, `LUMO-O (Hartree)` — frontier molecular orbital energies
  - `Dipole Moment X (Debye)`, `Dipole Moment Total (Debye)` — molecular dipole moment and its x-component
  - `D16 value` — the twisting dihedral angle Θ
  - `I-O Bond`, `I=O`, `I-OH bond` — iodine–oxygen bond length metrics

### Provenance note

`archive_original/` retains, for provenance, the original working script (`Mori2_original.py`), a notebook version of the analysis (`Mori2_notebook_original.ipynb`), the manuscript source, and a small, unrelated wide-layout table (`DFT_data_original.csv`, 18×17, only 2 of the 8 descriptors, a different compound set) that the analysis code never reads. The 284-compound dataset itself, any file found to be an equivalent copy of it, and generated outputs that expose individual-compound values (per-molecule predictions, PCA coordinates, and cluster assignments — see **Reproducibility** below) have been removed from this repository and its Git history. Aggregate results computed from the dataset (model-performance metrics, feature importance, correlation coefficients, descriptive statistics, and publication-style figures) remain, as they do not expose individual observations.

## Machine-learning methods

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- Support Vector Regression (RBF/linear kernel, tuned)
- Neural Network (MLP Regressor)

For each model: an 80/20 train-test split (`random_state=42`), evaluation with default hyperparameters, then hyperparameter tuning (`GridSearchCV` for Random Forest and Gradient Boosting, `RandomizedSearchCV` for SVR and the Neural Network, both with 5-fold cross-validation), residual and QQ-plot diagnostics, learning curves, feature importance (tree-based models), Partial Dependence Plots, and a SHAP analysis of the best-performing tuned model. All hyperparameter grids, the random seed, and the train/test split strategy are unchanged from the original script (`archive_original/Mori2_original.py`).

## Repository structure

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── README.md                # Explains where to place DFT_data.csv locally
│   └── DFT_data.csv             # NOT included — place your own copy here to run the analysis (gitignored)
│
├── src/
│   └── ibx_ml_analysis.py      # Cleaned analysis script (entry point)
│
├── results/
│   ├── figures/                # Generated plots (EDA, residuals, SHAP, PDP, ...)
│   ├── tables/                 # Aggregate metrics/importance/stats (row-level outputs gitignored)
│   └── models/                 # Persisted best model (.joblib, gitignored)
│
└── archive_original/            # Untouched originals kept for provenance (dataset files excluded)
    ├── DFT_data_original.csv                # Unrelated to the ML pipeline (see Provenance note)
    ├── Mori2_original.py                    # Original working script (was Mori2.txt)
    ├── Mori2_notebook_original.ipynb        # Notebook version of the analysis
    └── manuscript_source_text4.txt
```

## Installation

```bash
git clone https://github.com/Mor-cod/IBX_reactivity.git
cd IBX_reactivity

python -m venv .venv
```

Activate the environment:

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

**This public repository does not, by itself, let you reproduce the numerical results** — the dataset is required and is not included (see **Dataset** above). Once you have obtained `data/DFT_data.csv` (available upon publication) and placed it locally:

```bash
python src/ibx_ml_analysis.py
```

This loads `data/DFT_data.csv`, validates it, runs the exploratory analysis (correlation heatmap, pair plot, box plots, VIF, PCA, K-Means clustering), trains and evaluates all five models with default hyperparameters, tunes four of them (5-fold CV), generates residual/QQ/learning-curve diagnostics, feature importance, Partial Dependence Plots and a SHAP analysis of the best tuned model, and writes every figure and table into `results/`. Without the dataset present, the script will raise a `FileNotFoundError` at startup. Note that some generated per-compound outputs (individual predictions, PCA coordinates, cluster assignments) are excluded from version control by `.gitignore` even when regenerated locally, consistent with this repository's pre-publication data policy.

## Reproducibility

- Random seed `42` is used for the train/test split, all stochastic model estimators (Random Forest, Gradient Boosting, Neural Network), K-Means clustering, and `RandomizedSearchCV`.
- Package versions used for this reproducibility run are pinned in `requirements.txt`.
- The metrics below were obtained using the full 284-compound, 8-descriptor DFT dataset. **Neither the dataset nor the row-level prediction/PCA/cluster outputs used to compute these metrics are included in this public repository** (see **Dataset** above); only the aggregate metrics themselves are reported. Running `src/ibx_ml_analysis.py` without the dataset present will not reproduce them.

### Reproduced metrics vs. manuscript reference (default hyperparameters, DFT descriptors)

| Model | MAE (Reproduced) | MSE (Reproduced) | R² (Reproduced) | MAE (Manuscript) | R² (Manuscript) |
|---|---|---|---|---|---|
| Linear Regression | 0.9506 | 2.1652 | 0.8308 | 0.9506 | 0.8308 |
| Random Forest | 0.6776 | 0.8108 | 0.9366 | 0.6777 | 0.9366 |
| Gradient Boosting | 0.6833 | 0.7463 | 0.9417 | 0.6833 | 0.9417 |
| Support Vector Regression | 0.4397 | 0.3324 | 0.9740 | 0.4397 | 0.9740 |
| Neural Network | 0.4606 | 0.3220 | 0.9748 | 0.4606 | 0.9748 |

**The default-hyperparameter results reproduce the manuscript's Table 1 to four decimal places**, using the unmodified original code, random seed, and 80/20 split against the full 284-compound dataset described above (not included in this public repository).

After hyperparameter tuning (`GridSearchCV`/`RandomizedSearchCV`, 5-fold CV, seed 42):

| Model | MAE (Tuned) | R² (Tuned) |
|---|---|---|
| Random Forest | 0.706 | 0.931 |
| Gradient Boosting | 0.672 | 0.945 |
| Support Vector Regression | 0.776 | 0.909 |
| Neural Network | 0.727 | 0.910 |

The manuscript's Table 1 reports **default**-hyperparameter metrics; it does not tabulate exact tuned-model numbers (tuning results are shown only as diagnostic plots in the manuscript's Figure 6F–I). Note that tuned SVR/NN performed *worse* on the held-out test set here than their default counterparts — the grid/randomized search selects hyperparameters by 5-fold CV score on the training split, which does not guarantee improvement on this particular test split; this is expected search-variance behaviour, not a bug, and Gradient Boosting (tuned) was selected as the best model (R²=0.945).

### Key scientific conclusions — reproduced

- **Θ (the twisting dihedral angle, `D16 value`) is the dominant descriptor**: it accounts for ~88–90% of feature importance in both Random Forest and Gradient Boosting (`results/tables/random_forest_feature_importances.csv`, `results/tables/gradient_boosting_feature_importances.csv`), and is the top feature in the SHAP and Partial Dependence analyses.
- **Inverse relationship between Θ and the TS barrier**: confirmed — `D16 value` has a Pearson correlation of **-0.80** with `TS_Barrier` (`results/tables/correlation_matrix.csv`), the strongest of any descriptor.
- **PCA**: PC1 explains **92.48%** of variance and PC2 explains **5.28%** (`results/tables/pca_explained_variance.csv`) — an exact match to the manuscript's reported values.
- **SVR/NN outperforming linear/ensemble models**: confirmed at default hyperparameters (R² 0.974–0.975 vs. 0.83–0.94), matching the manuscript. Not preserved after this run's tuning (see above).
- **Model-specific SHAP-ranked PDPs**: `results/figures/pdp_shap_top3_models.png` (and `.pdf`) shows each model's own top-3 descriptors by mean |SHAP value|, with Partial Dependence curves computed via `method="brute"` (the default `"recursion"` method for gradient-boosted trees omits the ensemble's initial estimator and offsets its curves from the true predicted-response scale; `"brute"` keeps every panel on the actual TS-barrier scale, ~13–21 kcal/mol here). Θ dominates every model; the runner-up descriptors are μx (dipole moment X) and I–OH bond length for RF/GB/SVR/NN, and I–OH bond/I=O bond length for LR.

  Random Forest, Gradient Boosting and Linear Regression use the same default-hyperparameter fit reported in Table 1 (these algorithms are scale-invariant, so this is unambiguous). SVR and the Neural Network use their **tuned, `StandardScaler`-scaled pipeline** rather than the Table-1 default fit — this is a deliberate, single, consistently-applied choice for these two models, not an accidental mix of default/tuned analyses, and mirrors how SVR/NN are already treated in this script's own hyperparameter-tuning step and in the manuscript's stated methodology ("for models sensitive to feature scaling (SVR and NN), tuning was integrated within a pipeline that included a standard scaler"). Computing SHAP on SVR/NN's *unscaled* default fit instead produces a materially different top-3 ranking (dipole-moment-total displaces I–OH bond) — not because I–OH bond's values happen to occupy a narrow numerical range, but because SVR and MLP are trained on the *unscaled* features in that configuration, so the model itself weighs each descriptor differently; SHAP attribution reflects what the trained model actually learned; a different trained model yields a different attribution.

## Citation

Citation information will be added upon publication.

## License

This code is released under the MIT License. See [LICENSE](LICENSE) for details.
