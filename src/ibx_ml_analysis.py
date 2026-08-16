"""
Machine learning analysis of the hypervalent twisting transition-state (TS)
barrier of IBX (2-iodoxybenzoic acid) derivatives.

This script reproduces the ML workflow described in the accompanying
manuscript ("Machine Learning and Computational Chemistry for Understanding
Reactivity and Substituent Effects in Hypervalent Iodine Compounds"): five
regression models (Linear Regression, Random Forest, Gradient Boosting,
Support Vector Regression, Neural Network / MLP) are trained to predict the
DFT-derived TS barrier from molecular descriptors, followed by hyperparameter
tuning, residual diagnostics, PCA/clustering, feature importance, Partial
Dependence Plots and SHAP analysis.

Dataset
-------
``data/DFT_data.csv`` contains 284 IBX derivatives (matching the manuscript's
DFT computational set) with eight descriptors per compound: HOMO, LUMO,
dipole moment (X-component and total), the "D16 value" (the twisting
dihedral angle Theta, inferred from its summary statistics matching the
manuscript's reported Theta distribution), and three iodine-oxygen bond
metrics (I-O, I=O, I-OH). The table is already tidy (one row per compound,
no reshaping required).
"""

import os
import logging
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend: avoids Tk crashes with parallel (n_jobs>1) model fitting.
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
from scipy import stats

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, learning_curve
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.inspection import PartialDependenceDisplay
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Prevent a known KMeans memory-leak warning on Windows with MKL.
os.environ.setdefault("OMP_NUM_THREADS", "2")

# -----------------------------------------------------------------------
# Project paths (relative to this file, so the script is portable)
# -----------------------------------------------------------------------
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")

RAW_DATA_FILE = os.path.join(DATA_DIR, "DFT_data.csv")
TARGET_COLUMN = "TS_Barrier"

# Random seed used throughout for reproducibility (matches the original script).
RANDOM_STATE = 42

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger("ibx_ml_analysis")


def setup_logging():
    """Configure console + file logging into results/tables/script.log."""
    os.makedirs(TABLES_DIR, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(TABLES_DIR, "script.log"),
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(message)s",
    )


def ensure_output_dirs():
    for directory in (DATA_DIR, RESULTS_DIR, FIGURES_DIR, TABLES_DIR, MODELS_DIR):
        os.makedirs(directory, exist_ok=True)


# -----------------------------------------------------------------------
# 1. Data loading and validation
# -----------------------------------------------------------------------
def load_dataset(file_path=RAW_DATA_FILE):
    """Load DFT_data.csv (already tidy: one row per compound)."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    data = pd.read_csv(file_path)
    logger.info("Dataset loaded: %s (%d rows, %d columns)", file_path, *data.shape)
    return data


def validate_dataset(df):
    """
    Check the dataset for common data-quality issues and log a report.
    Returns the validation report as a dict (also saved to results/tables/).
    """
    report = {}

    report["n_rows"] = len(df)
    report["missing_values_per_column"] = df.isnull().sum().to_dict()
    report["n_duplicate_rows"] = int(df.duplicated().sum())
    report["n_unnamed_columns"] = int(sum(str(c).startswith("Unnamed") for c in df.columns))
    non_numeric = df.apply(lambda col: pd.to_numeric(col, errors="coerce").isnull().sum())
    report["n_numeric_conversion_failures"] = int(non_numeric.sum())

    report_df = pd.DataFrame([report])
    report_df.to_csv(os.path.join(TABLES_DIR, "dataset_validation_report.csv"), index=False)

    for key, value in report.items():
        logger.info("Dataset validation - %s: %s", key, value)
    print("Dataset validation report:")
    for key, value in report.items():
        print(f"  {key}: {value}")

    return report


def load_and_prepare_data(file_path=RAW_DATA_FILE):
    """Load and validate the dataset; return the DataFrame."""
    df = load_dataset(file_path)
    validate_dataset(df)
    return df


# -----------------------------------------------------------------------
# 2. Exploratory data analysis
# -----------------------------------------------------------------------
def save_descriptive_statistics(tidy_df):
    stats_df = tidy_df.describe()
    stats_df.to_csv(os.path.join(TABLES_DIR, "descriptive_statistics.csv"))
    logger.info("Descriptive statistics saved.")


def plot_target_distribution(y):
    plt.figure(figsize=(6, 4))
    sns.histplot(y, kde=True, color="steelblue")
    plt.title("Distribution of TS_Barrier")
    plt.xlabel("TS_Barrier")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "distribution_ts_barrier.png"), dpi=300)
    plt.close()
    logger.info("Distribution of TS_Barrier plot saved.")


def plot_correlation_heatmap(numeric_df):
    plt.figure(figsize=(8, 6))
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "correlation_heatmap.png"), dpi=300)
    plt.close()
    corr.to_csv(os.path.join(TABLES_DIR, "correlation_matrix.csv"))
    logger.info("Correlation heatmap and matrix saved.")


def plot_pairplot(numeric_df):
    sns.pairplot(numeric_df)
    plt.suptitle("Pair Plot of Numeric Features", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "pair_plot_numeric_features.png"), dpi=300)
    plt.close()
    logger.info("Pair plot saved.")


def plot_boxplots(X):
    plt.figure(figsize=(8, 5))
    x_melted = X.melt(var_name="Feature", value_name="Value")
    sns.boxplot(x="Feature", y="Value", data=x_melted)
    plt.title("Box Plots of All Features")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "box_plots_all_features.png"), dpi=300)
    plt.close()
    x_melted.to_csv(os.path.join(TABLES_DIR, "box_plots_data.csv"), index=False)
    logger.info("Box plots saved.")


def plot_missing_values_heatmap(X):
    plt.figure(figsize=(8, 4))
    sns.heatmap(X.isnull(), cbar=False, cmap="viridis")
    plt.title("Missing Values Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "missing_values_heatmap.png"), dpi=300)
    plt.close()

    missing_values = X.isnull().sum().reset_index()
    missing_values.columns = ["Feature", "Missing_Count"]
    missing_values["Missing_Percentage"] = (missing_values["Missing_Count"] / len(X)) * 100
    missing_values.to_csv(os.path.join(TABLES_DIR, "missing_values.csv"), index=False)
    logger.info("Missing values analysis saved.")


def compute_vif(X):
    """Variance Inflation Factor for multicollinearity among features."""
    x_with_const = X.copy()
    x_with_const["const"] = 1
    vif_data = pd.DataFrame()
    vif_data["Feature"] = x_with_const.columns
    vif_data["VIF"] = [
        variance_inflation_factor(x_with_const.values, i) for i in range(x_with_const.shape[1])
    ]
    vif_data = vif_data[vif_data["Feature"] != "const"]
    vif_data.to_csv(os.path.join(TABLES_DIR, "variance_inflation_factor.csv"), index=False)
    logger.info("VIF data saved.")
    return vif_data


def run_pca_and_clustering(X):
    """
    PCA (2 components) followed by K-Means clustering with silhouette-based
    model selection, as in the original workflow.
    """
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    pca_df = pd.DataFrame(data=X_pca, columns=["PC1", "PC2"])
    pca_df.to_csv(os.path.join(TABLES_DIR, "pca_transformed_data.csv"), index=False)

    plt.figure(figsize=(6, 5))
    sns.scatterplot(x="PC1", y="PC2", data=pca_df, alpha=0.7)
    plt.title("PCA - First Two Principal Components")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.2f}% Variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.2f}% Variance)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "pca_scatter_plot.png"), dpi=300)
    plt.close()

    pca_variance = pd.DataFrame({
        "Principal_Component": [f"PC{i + 1}" for i in range(len(pca.explained_variance_ratio_))],
        "Explained_Variance_Ratio": pca.explained_variance_ratio_,
    })
    pca_variance.to_csv(os.path.join(TABLES_DIR, "pca_explained_variance.csv"), index=False)
    logger.info("PCA analysis saved.")

    # K-Means with silhouette-based selection of the number of clusters.
    silhouette_scores = {}
    for n_clusters in range(2, 7):
        clusterer = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE)
        preds = clusterer.fit_predict(X_pca)
        silhouette_scores[n_clusters] = silhouette_score(X_pca, preds)

    plt.figure(figsize=(6, 5))
    sns.lineplot(x=list(silhouette_scores.keys()), y=list(silhouette_scores.values()), marker="o")
    plt.title("Silhouette Scores for Various Number of Clusters")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "silhouette_scores.png"), dpi=300)
    plt.close()

    optimal_clusters = max(silhouette_scores, key=silhouette_scores.get)
    logger.info("Optimal number of clusters: %d", optimal_clusters)

    kmeans = KMeans(n_clusters=optimal_clusters, random_state=RANDOM_STATE)
    cluster_labels = kmeans.fit_predict(X_pca)
    pca_df["Cluster"] = cluster_labels
    pca_df.to_csv(os.path.join(TABLES_DIR, "cluster_assignments.csv"), index=False)

    plt.figure(figsize=(6, 5))
    sns.scatterplot(x="PC1", y="PC2", hue="Cluster", palette="Set1", data=pca_df, alpha=0.7)
    plt.title(f"K-Means Clustering with {optimal_clusters} Clusters")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "kmeans_clusters_pca.png"), dpi=300)
    plt.close()
    logger.info("Cluster analysis saved.")

    return pca_df


def run_eda(tidy_df, X, y):
    save_descriptive_statistics(tidy_df)
    plot_target_distribution(y)
    numeric_df = tidy_df.select_dtypes(include=[float, int])
    plot_correlation_heatmap(numeric_df)
    plot_pairplot(numeric_df)
    plot_boxplots(X)
    plot_missing_values_heatmap(X)
    compute_vif(X)
    run_pca_and_clustering(X)


# -----------------------------------------------------------------------
# 3. Model definitions (unchanged from the original script)
# -----------------------------------------------------------------------
def get_default_models():
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
        "Support Vector Regression": SVR(),
        "Neural Network": MLPRegressor(random_state=RANDOM_STATE, max_iter=1000),
    }


def get_hyperparameter_grids():
    return {
        "Linear Regression": {},  # No hyperparameters to tune
        "Random Forest": {
            "regressor__n_estimators": [100, 200, 300],
            "regressor__max_depth": [None, 10, 20, 30],
            "regressor__min_samples_split": [2, 5, 10],
            "regressor__min_samples_leaf": [1, 2, 4],
            "regressor__bootstrap": [True, False],
        },
        "Gradient Boosting": {
            "regressor__n_estimators": [100, 200, 300],
            "regressor__learning_rate": [0.01, 0.05, 0.1],
            "regressor__max_depth": [3, 5, 7],
            "regressor__min_samples_split": [2, 5, 10],
            "regressor__min_samples_leaf": [1, 2, 4],
        },
        "Support Vector Regression": {
            "regressor__C": [0.5, 1, 5, 10],
            "regressor__epsilon": [0.05, 0.1, 0.2],
            "regressor__kernel": ["linear", "rbf"],
        },
        "Neural Network": {
            "regressor__hidden_layer_sizes": [(100,), (100, 50), (100, 100, 50)],
            "regressor__activation": ["relu", "tanh"],
            "regressor__solver": ["adam", "lbfgs"],
            "regressor__alpha": [0.0001, 0.001, 0.01],
            "regressor__learning_rate": ["constant", "adaptive"],
        },
    }


# -----------------------------------------------------------------------
# 4. Default-hyperparameter training and evaluation
# -----------------------------------------------------------------------
def compute_metrics(y_test, y_pred):
    return {
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "MAPE (%)": mean_absolute_percentage_error(y_test, y_pred) * 100,
        "R²": r2_score(y_test, y_pred),
    }


def plot_residuals(y_pred, residuals, model_name, tag):
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)
    plt.axhline(0, color="red", linestyle="--")
    plt.title(f"Residuals Plot for {model_name} ({tag})")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.tight_layout()
    fname = f"residuals_{tag.lower()}_{model_name.replace(' ', '_')}.png"
    plt.savefig(os.path.join(FIGURES_DIR, fname), dpi=300)
    plt.close()


def plot_residual_diagnostics(residuals, model_name, tag):
    plt.figure(figsize=(6, 4))
    sns.histplot(residuals, kde=True, color="purple")
    plt.title(f"Histogram of Residuals for {model_name} ({tag})")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.tight_layout()
    fname = f"histogram_residuals_{tag.lower()}_{model_name.replace(' ', '_')}.png"
    plt.savefig(os.path.join(FIGURES_DIR, fname), dpi=300)
    plt.close()

    plt.figure(figsize=(6, 4))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title(f"QQ Plot of Residuals for {model_name} ({tag})")
    plt.tight_layout()
    fname = f"qq_plot_residuals_{tag.lower()}_{model_name.replace(' ', '_')}.png"
    plt.savefig(os.path.join(FIGURES_DIR, fname), dpi=300)
    plt.close()


def train_and_evaluate_default_models(X_train, X_test, y_train, y_test):
    """Train each model with its default (untuned) hyperparameters.

    Returns the results table and the dict of fitted models (so callers can
    reuse the already-trained estimators for interpretability analyses
    instead of refitting).
    """
    default_models = get_default_models()
    default_results = []

    print("\n--- Training and Evaluating Models with Default Hyperparameters ---\n")
    logger.info("--- Training and Evaluating Models with Default Hyperparameters ---")

    for model_name, model in default_models.items():
        print(f"Training {model_name} with default parameters...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = compute_metrics(y_test, y_pred)
        default_results.append({"Model": model_name, **metrics})

        pd.DataFrame({"Actual": y_test, "Predicted": y_pred}).to_csv(
            os.path.join(TABLES_DIR, f"predictions_default_{model_name.replace(' ', '_')}.csv"),
            index=False,
        )

        residuals = y_test - y_pred
        plot_residuals(y_pred, residuals, model_name, "Default")
        plot_residual_diagnostics(residuals, model_name, "Default")

    default_results_df = pd.DataFrame(default_results)
    print("\nDefault Model Evaluation Results:\n", default_results_df)
    default_results_df.to_csv(os.path.join(TABLES_DIR, "default_model_evaluation_results.csv"), index=False)
    return default_results_df, default_models


# -----------------------------------------------------------------------
# 5. Hyperparameter tuning
# -----------------------------------------------------------------------
def tune_models(X_train, X_test, y_train, y_test):
    default_models = get_default_models()
    hyperparameter_grids = get_hyperparameter_grids()

    tuned_results = []
    best_estimators = {}

    print("\n--- Hyperparameter Tuning for Each Model ---\n")
    logger.info("--- Hyperparameter Tuning for Each Model ---")

    for model_name, params in hyperparameter_grids.items():
        if model_name == "Linear Regression":
            print(f"\n{model_name} has no hyperparameters to tune.\n")
            continue

        print(f"Starting hyperparameter tuning for {model_name}...")

        if model_name in ("Support Vector Regression", "Neural Network"):
            pipeline = Pipeline([("scaler", StandardScaler()), ("regressor", default_models[model_name])])
        else:
            pipeline = Pipeline([("regressor", default_models[model_name])])

        if model_name in ("Random Forest", "Gradient Boosting"):
            search = GridSearchCV(
                estimator=pipeline, param_grid=params, cv=5, scoring="r2", n_jobs=4, verbose=0,
            )
        else:
            total_params = int(np.prod([len(v) for v in params.values()]))
            n_iter = min(50, total_params)
            search = RandomizedSearchCV(
                estimator=pipeline, param_distributions=params, n_iter=n_iter, cv=5,
                scoring="r2", random_state=RANDOM_STATE, n_jobs=4, verbose=0,
            )

        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        best_estimators[model_name] = best_model
        print(f"Best parameters for {model_name}: {search.best_params_}")
        logger.info("Best parameters for %s: %s", model_name, search.best_params_)

        y_pred_tuned = best_model.predict(X_test)
        metrics = compute_metrics(y_test, y_pred_tuned)
        tuned_results.append({"Model": model_name, **metrics})

        pd.DataFrame({"Actual": y_test, "Predicted": y_pred_tuned}).to_csv(
            os.path.join(TABLES_DIR, f"predictions_tuned_{model_name.replace(' ', '_')}.csv"), index=False,
        )

        residuals_tuned = y_test - y_pred_tuned
        plot_residuals(y_pred_tuned, residuals_tuned, model_name, "Tuned")

        train_sizes, train_scores, test_scores = learning_curve(
            best_model, X_train, y_train, cv=5, scoring="r2", n_jobs=4,
            train_sizes=np.linspace(0.1, 1.0, 10),
        )
        plt.figure(figsize=(6, 4))
        plt.plot(train_sizes, train_scores.mean(axis=1), "o-", color="blue", label="Training score")
        plt.plot(train_sizes, test_scores.mean(axis=1), "o-", color="green", label="Cross-validation score")
        plt.title(f"Learning Curve for {model_name} (Tuned)")
        plt.xlabel("Training Set Size")
        plt.ylabel("R² Score")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f"learning_curve_tuned_{model_name.replace(' ', '_')}.png"), dpi=300)
        plt.close()

    tuned_results_df = pd.DataFrame(tuned_results)
    print("\nTuned Model Evaluation Results:\n", tuned_results_df)
    if not tuned_results_df.empty:
        tuned_results_df.to_csv(os.path.join(TABLES_DIR, "tuned_model_evaluation_results.csv"), index=False)

    return tuned_results_df, best_estimators


def save_comparison(default_results_df, tuned_results_df):
    if tuned_results_df.empty:
        return None
    comparison_df = pd.merge(
        default_results_df, tuned_results_df, on="Model", how="left", suffixes=("_Default", "_Tuned"),
    )
    comparison_df.to_csv(os.path.join(TABLES_DIR, "models_comparison.csv"), index=False)

    for metric in ["MAE", "MSE", "RMSE", "MAPE (%)", "R²"]:
        plt.figure(figsize=(8, 5))
        sns.barplot(x="Model", y=f"{metric}_Default", data=comparison_df, color="skyblue", label="Default")
        sns.barplot(x="Model", y=f"{metric}_Tuned", data=comparison_df, color="salmon", alpha=0.7, label="Tuned")
        plt.title(f"Model Comparison by {metric}")
        plt.xticks(rotation=45, ha="right")
        plt.legend()
        plt.tight_layout()
        safe_metric = metric.replace(" ", "_").replace("(", "").replace(")", "").replace("%", "pct")
        plt.savefig(os.path.join(FIGURES_DIR, f"model_comparison_{safe_metric}.png"), dpi=300)
        plt.close()

    return comparison_df


def select_best_model(default_results_df, tuned_results_df, default_models, best_estimators):
    if not tuned_results_df.empty and "Model" in tuned_results_df.columns:
        best_row = tuned_results_df.sort_values(by="R²", ascending=False).iloc[0]
        best_model_name = best_row["Model"]
        best_model = best_estimators[best_model_name]
        print(f"\nBest model based on Tuned R² is: {best_model_name}\n")
    else:
        best_row = default_results_df.sort_values(by="R²", ascending=False).iloc[0]
        best_model_name = best_row["Model"]
        best_model = default_models[best_model_name]
        print(f"\nBest model based on Default R² is: {best_model_name} (Default)\n")

    logger.info("Best model: %s", best_model_name)
    return best_model_name, best_model


# -----------------------------------------------------------------------
# 6. Interpretability: feature importance, PDP, SHAP
# -----------------------------------------------------------------------
def plot_feature_importances(best_estimators, X):
    for model_name in ("Random Forest", "Gradient Boosting"):
        if model_name not in best_estimators:
            continue
        model = best_estimators[model_name]
        regressor = model.named_steps["regressor"] if hasattr(model, "named_steps") else model
        if not hasattr(regressor, "feature_importances_"):
            continue

        importances = pd.Series(regressor.feature_importances_, index=X.columns).sort_values(ascending=False)
        plt.figure(figsize=(8, 5))
        sns.barplot(x=importances, y=importances.index, palette="magma")
        plt.title(f"{model_name} Feature Importances")
        plt.xlabel("Importance Score")
        plt.tight_layout()
        fname = f"{model_name.lower().replace(' ', '_')}_feature_importances.png"
        plt.savefig(os.path.join(FIGURES_DIR, fname), dpi=300)
        plt.close()
        importances.to_csv(os.path.join(TABLES_DIR, f"{model_name.lower().replace(' ', '_')}_feature_importances.csv"))


def plot_partial_dependence(best_model_name, best_model, best_estimators, X, X_train):
    if best_model_name == "Linear Regression":
        print("Best model is Linear Regression. Skipping Partial Dependence Plots.")
        return []

    if best_model_name in ("Random Forest", "Gradient Boosting"):
        importances = best_estimators[best_model_name].named_steps["regressor"].feature_importances_
        feature_importances = pd.Series(importances, index=X.columns).sort_values(ascending=False)
        top_features = feature_importances.head(3).index.tolist()
    else:
        top_features = X.columns[: min(3, len(X.columns))].tolist()

    print(f"Generating Partial Dependence Plots for top features: {top_features}")
    fig, ax = plt.subplots(figsize=(6 * len(top_features), 4))
    PartialDependenceDisplay.from_estimator(best_model, X_train, features=top_features, ax=ax, kind="average")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "pdp_best_model.png"), dpi=300)
    plt.close()
    return top_features


def run_shap_analysis(best_model_name, best_model, X_train, top_features):
    print("\n--- Performing SHAP Analysis for the Best Model ---\n")
    try:
        if best_model_name in ("Support Vector Regression", "Neural Network"):
            regressor = best_model.named_steps["regressor"]
            scaler = best_model.named_steps["scaler"]
            X_train_input = scaler.transform(X_train)
        else:
            regressor = best_model.named_steps["regressor"] if hasattr(best_model, "named_steps") else best_model
            X_train_input = X_train

        explainer = shap.Explainer(regressor, X_train_input)
        shap_values = explainer(X_train_input)

        plt.figure()
        shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
        plt.title(f"SHAP Summary Plot for {best_model_name} (Global Importance)")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f"shap_summary_global_{best_model_name.replace(' ', '_')}.png"), dpi=300)
        plt.close()

        plt.figure()
        shap.summary_plot(shap_values, X_train, show=False)
        plt.title(f"SHAP Summary Plot for {best_model_name} (Detailed)")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f"shap_summary_detailed_{best_model_name.replace(' ', '_')}.png"), dpi=300)
        plt.close()

        top_feature = top_features[0] if top_features else X_train.columns[0]
        plt.figure()
        shap.dependence_plot(top_feature, shap_values.values, X_train, show=False)
        plt.title(f"SHAP Dependence Plot for {top_feature} in {best_model_name}")
        plt.tight_layout()
        fname = f"shap_dependence_{top_feature}_{best_model_name.replace(' ', '_')}.png".replace(" ", "_")
        plt.savefig(os.path.join(FIGURES_DIR, fname), dpi=300)
        plt.close()
        logger.info("SHAP analysis completed for %s.", best_model_name)
    except Exception as exc:  # SHAP explainer choice is model-dependent and can fail for some estimators.
        print(f"Error during SHAP analysis for {best_model_name}: {exc}")
        logger.error("Error during SHAP analysis for %s: %s", best_model_name, exc, exc_info=True)


# Display labels for descriptor columns, used in the SHAP-ranked PDP grid.
FEATURE_DISPLAY_LABELS = {
    "D16 value": "Θ (dihedral angle)",
    "Dipole Moment X (Debye)": "μx (dipole moment, X)",
    "Dipole Moment Total (Debye)": "μ (dipole moment, total)",
    "I-OH bond": "I–OH bond length",
    "I=O": "I=O bond length",
    "I-O Bond": "I–O bond length",
    "HOMO-O (Hartree)": "HOMO",
    "LUMO-O (Hartree)": "LUMO",
}


def compute_shap_top_features(fitted_model, X_train, top_n=3):
    """
    Compute SHAP values for a fitted model and return its top-N descriptors
    ranked by mean absolute SHAP value, plus the full ranking.

    Works with either a plain estimator or a scaling Pipeline
    (StandardScaler + regressor, as used elsewhere in this script for the
    tuned SVR/Neural Network models), so it is reusable for any model in
    the pipeline without modification.
    """
    if hasattr(fitted_model, "named_steps"):
        regressor = fitted_model.named_steps["regressor"]
        scaler = fitted_model.named_steps.get("scaler")
        X_input = scaler.transform(X_train) if scaler is not None else X_train.values
        X_input = pd.DataFrame(X_input, columns=X_train.columns, index=X_train.index)
    else:
        regressor = fitted_model
        X_input = X_train

    try:
        # shap.Explainer auto-detects fast, exact explainers (Linear/Tree) for
        # model types it recognises directly.
        explainer = shap.Explainer(regressor, X_input)
    except TypeError:
        # Model types SHAP doesn't special-case (e.g. SVR, MLPRegressor) must
        # be passed as a callable predict function, which selects a
        # model-agnostic (Permutation) explainer instead.
        explainer = shap.Explainer(regressor.predict, X_input)
    shap_values = explainer(X_input)

    mean_abs_shap = pd.Series(
        np.abs(shap_values.values).mean(axis=0), index=X_train.columns
    ).sort_values(ascending=False)

    return mean_abs_shap.head(top_n).index.tolist(), mean_abs_shap


def plot_pdp_grid(model_entries, X_train, png_path, pdf_path, feature_labels=None):
    """
    Reusable routine: plot a grid of Partial Dependence panels, one row per
    model and one column per selected feature for that model, with
    lettered panel labels (A, B, C, ...).

    Parameters
    ----------
    model_entries : list of dict
        Each entry needs 'name' (row label), 'model' (fitted estimator or
        scaling Pipeline), and 'features' (ordered list of column names to
        plot for that model).
    X_train : pd.DataFrame
        Training features used to compute the PDP curves.
    feature_labels : dict, optional
        Maps internal column names to publication-style display labels.
    """
    feature_labels = feature_labels or {}
    n_rows = len(model_entries)
    n_cols = max(len(entry["features"]) for entry in model_entries)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.atleast_2d(axes)

    letters = [chr(ord("A") + i) for i in range(n_rows * n_cols)]
    letter_idx = 0

    for row, entry in enumerate(model_entries):
        for col in range(n_cols):
            ax = axes[row, col]
            if col >= len(entry["features"]):
                ax.axis("off")
                letter_idx += 1
                continue

            feature = entry["features"][col]
            # method="brute" computes PDP curves by actually averaging model
            # predictions over the grid (true response scale for every model
            # type). The default "recursion" method for gradient-boosted
            # trees omits the ensemble's initial estimator, which shifts its
            # curves by a constant offset relative to the actual predicted
            # scale -- "brute" avoids that and keeps all panels comparable.
            PartialDependenceDisplay.from_estimator(
                entry["model"], X_train, features=[feature], ax=ax, kind="average", method="brute",
            )
            label = feature_labels.get(feature, feature)
            ax.set_title(f"{letters[letter_idx]}  {entry['name']} — {label}", fontsize=10, loc="left")
            ax.set_xlabel(label)
            ax.set_ylabel("Partial dependence\n(TS barrier, kcal/mol)" if col == 0 else "")
            letter_idx += 1

    plt.tight_layout()
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)


def select_shap_pdp_models(fitted_default_models, tuned_estimators):
    """
    Choose which fitted estimator represents each model for SHAP/PDP
    analysis. Linear Regression, Random Forest and Gradient Boosting are
    scale-invariant, so their plain default-hyperparameter fit is used.
    SVR and the Neural Network are scale-sensitive: computing SHAP on their
    unscaled default fit distorts attribution towards large-magnitude
    features (e.g. total dipole moment) and away from small-range ones
    (e.g. I-OH bond length), so their tuned StandardScaler pipeline is used
    instead, matching how they are actually deployed elsewhere in this
    script (see tune_models()).
    """
    return {
        "Linear Regression": fitted_default_models["Linear Regression"],
        "Random Forest": fitted_default_models["Random Forest"],
        "Gradient Boosting": fitted_default_models["Gradient Boosting"],
        "Support Vector Regression": tuned_estimators["Support Vector Regression"],
        "Neural Network": tuned_estimators["Neural Network"],
    }


def generate_shap_ranked_pdp_grid(models_by_name, X_train, top_n=3):
    """
    For each of the five models, rank descriptors by mean |SHAP value| and
    plot the top-N as a composite Partial Dependence grid (one row per
    model, panels lettered A-O). Saves results/figures/pdp_shap_top3_models
    as both PNG and PDF. Returns {model_name: (top_features, full_ranking)}
    for verification/logging.
    """
    model_order = [
        "Linear Regression", "Random Forest", "Gradient Boosting",
        "Support Vector Regression", "Neural Network",
    ]
    ranking_summary = {}
    model_entries = []

    print("\n--- Computing SHAP-ranked descriptors for all five models ---\n")
    for model_name in model_order:
        model = models_by_name[model_name]
        print(f"Computing SHAP feature ranking for {model_name}...")
        top_features, full_ranking = compute_shap_top_features(model, X_train, top_n=top_n)
        ranking_summary[model_name] = (top_features, full_ranking)
        model_entries.append({"name": model_name, "model": model, "features": top_features})
        logger.info("SHAP top-%d for %s: %s", top_n, model_name, top_features)

    png_path = os.path.join(FIGURES_DIR, "pdp_shap_top3_models.png")
    pdf_path = os.path.join(FIGURES_DIR, "pdp_shap_top3_models.pdf")
    plot_pdp_grid(model_entries, X_train, png_path, pdf_path, feature_labels=FEATURE_DISPLAY_LABELS)
    print(f"\nSaved SHAP-ranked PDP grid: {png_path}\n                            {pdf_path}")

    return ranking_summary, png_path, pdf_path


def persist_best_model(best_model_name, best_model):
    try:
        pd.Series(best_model.get_params()).to_csv(
            os.path.join(TABLES_DIR, f"best_model_{best_model_name.replace(' ', '_')}_parameters.csv")
        )
    except Exception as exc:
        logger.error("Error saving parameters for %s: %s", best_model_name, exc, exc_info=True)

    joblib.dump(best_model, os.path.join(MODELS_DIR, f"best_model_{best_model_name.replace(' ', '_')}.joblib"))
    print(f"Best model ({best_model_name}) saved using joblib.")


# -----------------------------------------------------------------------
# 7. Illustrative prediction for a new (hypothetical) structure
# -----------------------------------------------------------------------
def predict_new_structure(best_model_name, best_model, X):
    """
    Predict the TS barrier for one illustrative set of descriptor values,
    exactly as in the original script (archive_original/Mori2_original.py).
    """
    new_structure_data = {
        "HOMO-O (Hartree)": [-0.32608],
        "LUMO-O (Hartree)": [-0.03573],
        "Dipole Moment X (Debye)": [3.0385],
        "Dipole Moment Total (Debye)": [10.4733],
        "D16 value": [3.88178],
        "I-O Bond": [2.173799731],
        "I=O": [1.819830622],
        "I-OH bond": [2.002986092],
    }
    new_df = pd.DataFrame(new_structure_data)

    missing_cols = set(X.columns) - set(new_df.columns)
    for col in missing_cols:
        new_df[col] = X[col].mean()
    new_df = new_df[X.columns]

    ts_pred = best_model.predict(new_df)
    print("--- Prediction for New Structure(s) ---")
    print("Input Features:\n", new_df)
    print("Predicted TS_Barrier:", ts_pred)

    prediction_df = new_df.copy()
    prediction_df["Predicted_TS_Barrier"] = ts_pred
    prediction_df.to_csv(os.path.join(TABLES_DIR, "new_structure_prediction.csv"), index=False)
    logger.info("Prediction for new structure (%s) saved.", best_model_name)


# -----------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------
def main():
    ensure_output_dirs()
    setup_logging()

    df = load_and_prepare_data()

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")

    X = df.drop(columns=[TARGET_COLUMN]).select_dtypes(include=[float, int])
    y = df[TARGET_COLUMN]

    # Drop rows with missing values in features or target (none expected, but explicit for safety).
    valid_rows = X.notnull().all(axis=1) & y.notnull()
    X, y = X.loc[valid_rows], y.loc[valid_rows]

    run_eda(df, X, y)

    # 80/20 train-test split (unchanged from the original script).
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    print("Train-test split completed.")

    default_results_df, fitted_default_models = train_and_evaluate_default_models(X_train, X_test, y_train, y_test)
    tuned_results_df, best_estimators = tune_models(X_train, X_test, y_train, y_test)
    save_comparison(default_results_df, tuned_results_df)

    best_model_name, best_model = select_best_model(
        default_results_df, tuned_results_df, fitted_default_models, best_estimators
    )

    plot_feature_importances(best_estimators, X)
    top_features = plot_partial_dependence(best_model_name, best_model, best_estimators, X, X_train)
    run_shap_analysis(best_model_name, best_model, X_train, top_features)
    persist_best_model(best_model_name, best_model)
    predict_new_structure(best_model_name, best_model, X)

    shap_pdp_models = select_shap_pdp_models(fitted_default_models, best_estimators)
    generate_shap_ranked_pdp_grid(shap_pdp_models, X_train, top_n=3)

    print("\nWorkflow complete. See results/figures and results/tables for outputs.\n")


if __name__ == "__main__":
    main()
