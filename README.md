# Machine Learning Analysis of Reactivity and Substituent Effects in Hypervalent Iodine(V) Compounds

## Overview

This repository contains Python code and computational data used to investigate the hypervalent twisting transition-state (TS) barrier and substituent effects in IBX (2-iodoxybenzoic acid) derivatives, using machine learning models trained on DFT-derived molecular descriptors. Five regression models — Linear Regression, Random Forest, Gradient Boosting, Support Vector Regression, and a Neural Network (MLP) — are trained and evaluated to predict the twisting TS barrier, followed by hyperparameter tuning, residual diagnostics, dimensionality reduction, feature importance, Partial Dependence Plots (PDP), and SHAP analysis.

## Research context

This work was conducted in the Chemistry discipline at the University of Tasmania (UTAS), Hobart, Australia.

## Dataset

`data/DFT_data.csv` contains **284 IBX derivatives** with eight DFT-derived descriptors and the target TS barrier — one row per compound, already analysis-ready. This matches the manuscript's stated computational set ("These computations encompassed 284 IBX derivatives") and its reported descriptive statistics (e.g. TS barrier mean 18.45 kcal·mol⁻¹, range ~9–23; `D16 value` mean 7.30 / std 7.77, matching the manuscript's reported Θ distribution exactly). Dataset validation found **no missing values, no duplicate rows, no unnamed columns, and no numeric-conversion issues** (see `results/tables/dataset_validation_report.csv`).

- **Observations**: individual IBX derivatives (no compound-name column is present in this file — compounds are identified only by row/descriptor values).
- **Target variable**: `TS_Barrier` — the DFT-computed transition-state barrier (kcal·mol⁻¹) for the hypervalent twisting step in methanol oxidation.
- **Descriptors**:
  - `HOMO-O (Hartree)`, `LUMO-O (Hartree)` — frontier molecular orbital energies
  - `Dipole Moment X (Debye)`, `Dipole Moment Total (Debye)` — molecular dipole moment and its x-component
  - `D16 value` — the twisting dihedral angle Θ (see note above on how this identification was made)
  - `I-O Bond`, `I=O`, `I-OH bond` — iodine–oxygen bond length metrics

### Provenance note

During cleanup, the only file initially present in the project folder (`archive_original/DFT_data_original.csv`) was a small, wide-layout table (72 compounds, single descriptor `D16 value` + `TS_Barrier`, with compound-name columns) that turned out to be a *different*, unrelated file — the analysis script never reads it, and it is preserved in `archive_original/` for provenance only, not used by the pipeline. The actual training dataset (`new3_sample-orginal.csv`, 284 rows × 8 descriptors) and a Jupyter notebook version of the analysis (`Mori2.ipynb`) were supplied afterward and are archived unmodified as `archive_original/new3_sample-orginal_original.csv` and `archive_original/Mori2_notebook_original.ipynb`. The notebook's final, most complete code cell was verified to be **byte-identical** to `archive_original/Mori2_original.py`, confirming `src/ibx_ml_analysis.py` was refactored from the correct, final version of the methodology.

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
│   └── DFT_data.csv            # 284 IBX derivatives x 8 DFT descriptors + TS_Barrier
│
├── src/
│   └── ibx_ml_analysis.py      # Cleaned analysis script (entry point)
│
├── results/
│   ├── figures/                # Generated plots (EDA, residuals, SHAP, PDP, ...)
│   ├── tables/                 # Generated CSV tables, metrics, logs
│   └── models/                 # Persisted best model (.joblib, gitignored)
│
└── archive_original/            # Untouched originals kept for provenance
    ├── DFT_data_original.csv                # Superseded; unrelated to the ML pipeline (see Provenance note)
    ├── Mori2_original.py                    # Original working script (was Mori2.txt)
    ├── Mori2_notebook_original.ipynb        # Notebook version of the analysis
    ├── new3_sample-orginal_original.csv     # Original name of data/DFT_data.csv
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

Run the full analysis from the repository root:

```bash
python src/ibx_ml_analysis.py
```

This loads `data/DFT_data.csv`, validates it, runs the exploratory analysis (correlation heatmap, pair plot, box plots, VIF, PCA, K-Means clustering), trains and evaluates all five models with default hyperparameters, tunes four of them (5-fold CV), generates residual/QQ/learning-curve diagnostics, feature importance, Partial Dependence Plots and a SHAP analysis of the best tuned model, and writes every figure and table into `results/`. No manual intervention is required.

## Reproducibility

- Random seed `42` is used for the train/test split, all stochastic model estimators (Random Forest, Gradient Boosting, Neural Network), K-Means clustering, and `RandomizedSearchCV`.
- All outputs are written to `results/figures/` and `results/tables/`; the best tuned model is persisted to `results/models/`.
- Package versions used for this reproducibility run are pinned in `requirements.txt`.

### Reproduced metrics vs. manuscript reference (default hyperparameters, DFT descriptors)

| Model | MAE (Reproduced) | MSE (Reproduced) | R² (Reproduced) | MAE (Manuscript) | R² (Manuscript) |
|---|---|---|---|---|---|
| Linear Regression | 0.9506 | 2.1652 | 0.8308 | 0.9506 | 0.8308 |
| Random Forest | 0.6776 | 0.8108 | 0.9366 | 0.6777 | 0.9366 |
| Gradient Boosting | 0.6833 | 0.7463 | 0.9417 | 0.6833 | 0.9417 |
| Support Vector Regression | 0.4397 | 0.3324 | 0.9740 | 0.4397 | 0.9740 |
| Neural Network | 0.4606 | 0.3220 | 0.9748 | 0.4606 | 0.9748 |

**The default-hyperparameter results reproduce the manuscript's Table 1 to four decimal places**, using the unmodified original code, random seed, and 80/20 split against the recovered dataset (`archive_original/new3_sample-orginal_original.csv`).

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

## Citation

Citation information will be added upon publication.

## License

This code is released under the MIT License. See [LICENSE](LICENSE) for details.
