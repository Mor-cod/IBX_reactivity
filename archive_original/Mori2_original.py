# -----------------------------
# 1. Setup Environment Variables and Imports
# -----------------------------

import os  # Import os first to set environment variables
# Set environment variable to prevent KMeans memory leak warnings on Windows with MKL
os.environ["OMP_NUM_THREADS"] = "2"

import pandas as pd
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
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
import joblib  # For model persistence
import logging
import shap  # For SHAP analysis

# -----------------------------
# 2. Configure Logging
# -----------------------------
# Ensure a results directory exists
results_dir = 'Mori2Results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Configure logging
logging.basicConfig(
    filename=os.path.join(results_dir, 'script.log'),
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

# -----------------------------
# 3. Suppress Specific Warnings for Cleaner Output
# -----------------------------
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)  # To suppress KMeans warnings after setting OMP_NUM_THREADS

# -----------------------------
# 4. Load the Dataset
# -----------------------------
def load_csv(file_path):
    """
    Load a CSV file into a pandas DataFrame.

    Parameters:
    - file_path (str): Path to the CSV file.

    Returns:
    - pd.DataFrame or None: Loaded DataFrame or None if loading fails.
    """
    if os.path.exists(file_path):
        try:
            data = pd.read_csv(file_path)
            print("File loaded successfully!")
            logging.info("File loaded successfully!")
            return data
        except Exception as e:
            print(f"Error loading file: {e}")
            logging.error(f"Error loading file: {e}", exc_info=True)
            return None
    else:
        print("File not found. Please check the path and try again.")
        logging.error("File not found. Please check the path and try again.")
        return None

# Specify your CSV file name here
csv_file = 'new3_sample-orginal1.csv'  # Ensure this file is in the same directory as the script
data = load_csv(csv_file)

# -----------------------------
# 5. Exploratory Data Analysis (EDA)
# -----------------------------
if data is not None:
    # Verify the target column exists
    if 'TS_Barrier' in data.columns:
        # -----------------------------
        # 5.1 Separate Target from Features
        # -----------------------------
        X = data.drop(columns=['TS_Barrier'])
        # Keep only numeric columns
        X = X.select_dtypes(include=[float, int])
        y = data['TS_Barrier']

        # -----------------------------
        # 5.2 Save Raw Data Statistics
        # -----------------------------
        descriptive_stats = data.describe()
        descriptive_stats.to_csv(os.path.join(results_dir, 'descriptive_statistics.csv'))
        print("Descriptive statistics saved.")
        logging.info("Descriptive statistics saved.")

        # -----------------------------
        # 5.3 Exploratory Plots
        # -----------------------------
        # 5.3.1 Distribution of TS_Barrier
        plt.figure(figsize=(6, 4))
        sns.histplot(y, kde=True, color='steelblue')
        plt.title('Distribution of TS_Barrier')
        plt.xlabel('TS_Barrier')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'distribution_ts_barrier.png'), dpi=300)
        plt.show()
        plt.close()
        logging.info("Distribution of TS_Barrier plot saved.")

        # 5.3.2 Correlation Heatmap (numeric columns only)
        plt.figure(figsize=(12, 10))
        numeric_data = data.select_dtypes(include=[float, int])  # only numeric
        corr = numeric_data.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'correlation_heatmap.png'), dpi=300)
        plt.show()
        plt.close()
        corr.to_csv(os.path.join(results_dir, 'correlation_matrix.csv'))
        print("Correlation matrix saved.")
        logging.info("Correlation matrix saved.")

        # 5.3.3 Pair Plot for all numeric features
        sns.pairplot(numeric_data)
        plt.suptitle('Pair Plot of Numeric Features', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'pair_plot_numeric_features.png'), dpi=300)
        plt.show()
        plt.close()
        logging.info("Pair plot of numeric features saved.")

        # 5.3.4 Box Plots for all features
        plt.figure(figsize=(15, 10))
        X_melted = X.melt(var_name='Feature', value_name='Value')
        sns.boxplot(x='Feature', y='Value', data=X_melted)
        plt.title('Box Plots of All Features')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'box_plots_all_features.png'), dpi=300)
        plt.show()
        plt.close()
        X_melted.to_csv(os.path.join(results_dir, 'box_plots_data.csv'), index=False)
        print("Box plots data saved.")
        logging.info("Box plots data saved.")

        # 5.3.5 Individual Feature Distributions
        for feature in X.columns:
            plt.figure(figsize=(6, 4))
            sns.histplot(X[feature], kde=True, color='green')
            plt.title(f'Distribution of {feature}')
            plt.xlabel(feature)
            plt.ylabel('Frequency')
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f'distribution_{feature}.png'), dpi=300)
            plt.show()
            plt.close()
            logging.info(f"Distribution plot for {feature} saved.")

        # -----------------------------
        # 5.4 Additional EDA Analyses
        # -----------------------------
        # 5.4.1 Missing Value Analysis
        plt.figure(figsize=(12, 6))
        sns.heatmap(X.isnull(), cbar=False, cmap='viridis')
        plt.title('Missing Values Heatmap')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'missing_values_heatmap.png'), dpi=300)
        plt.show()
        plt.close()

        # Save Missing Values Data
        missing_values = X.isnull().sum().reset_index()
        missing_values.columns = ['Feature', 'Missing_Count']
        missing_values['Missing_Percentage'] = (missing_values['Missing_Count'] / len(X)) * 100
        missing_values.to_csv(os.path.join(results_dir, 'missing_values.csv'), index=False)
        print("Missing values analysis saved.")
        logging.info("Missing values analysis saved.")

        # 5.4.2 Variance Inflation Factor (VIF) for Multicollinearity
        # Adding a constant for VIF calculation
        X_with_const = X.copy()
        X_with_const['const'] = 1
        vif_data = pd.DataFrame()
        vif_data['Feature'] = X_with_const.columns
        vif_data['VIF'] = [variance_inflation_factor(X_with_const.values, i) for i in range(X_with_const.shape[1])]
        # Remove the constant
        vif_data = vif_data[vif_data['Feature'] != 'const']
        vif_data.to_csv(os.path.join(results_dir, 'variance_inflation_factor.csv'), index=False)
        print("Variance Inflation Factor (VIF) data saved.")
        logging.info("Variance Inflation Factor (VIF) data saved.")

        # 5.4.3 Principal Component Analysis (PCA)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
        pca_df.to_csv(os.path.join(results_dir, 'pca_transformed_data.csv'), index=False)

        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='PC1', y='PC2', data=pca_df, alpha=0.7)
        plt.title('PCA - First Two Principal Components')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}% Variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}% Variance)')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'pca_scatter_plot.png'), dpi=300)
        plt.show()
        plt.close()

        # Save PCA Explained Variance
        pca_variance = pd.DataFrame({
            'Principal_Component': [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
            'Explained_Variance_Ratio': pca.explained_variance_ratio_
        })
        pca_variance.to_csv(os.path.join(results_dir, 'pca_explained_variance.csv'), index=False)
        print("PCA analysis saved.")
        logging.info("PCA analysis saved.")

        # 5.4.4 Cluster Analysis (K-Means)
        silhouette_scores = {}
        range_n_clusters = list(range(2, 7))
        for n_clusters in range_n_clusters:
            try:
                clusterer = KMeans(n_clusters=n_clusters, random_state=42)
                preds = clusterer.fit_predict(X_pca)
                silhouette_avg = silhouette_score(X_pca, preds)
                silhouette_scores[n_clusters] = silhouette_avg
                logging.info(f"Silhouette score for {n_clusters} clusters: {silhouette_avg}")
            except Exception as e:
                print(f"Error during KMeans clustering with {n_clusters} clusters: {e}")
                logging.error(f"Error during KMeans clustering with {n_clusters} clusters: {e}", exc_info=True)

        # Plot Silhouette Scores
        plt.figure(figsize=(8, 6))
        sns.lineplot(x=list(silhouette_scores.keys()), y=list(silhouette_scores.values()), marker='o')
        plt.title('Silhouette Scores for Various Number of Clusters')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.xticks(range_n_clusters)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'silhouette_scores.png'), dpi=300)
        plt.show()
        plt.close()

        # Choose number of clusters with highest silhouette score
        if silhouette_scores:
            optimal_clusters = max(silhouette_scores, key=silhouette_scores.get)
            print(f"Optimal number of clusters based on silhouette score: {optimal_clusters}")
            logging.info(f"Optimal number of clusters based on silhouette score: {optimal_clusters}")
        else:
            optimal_clusters = 2  # Default to 2 if silhouette_scores is empty
            print("Could not determine optimal number of clusters. Defaulting to 2.")
            logging.warning("Could not determine optimal number of clusters. Defaulting to 2.")

        # Apply K-Means with optimal clusters
        try:
            kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(X_pca)
            pca_df['Cluster'] = cluster_labels
            pca_df.to_csv(os.path.join(results_dir, 'cluster_assignments.csv'), index=False)

            # Plot Clusters
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x='PC1', y='PC2', hue='Cluster', palette='Set1', data=pca_df, alpha=0.7)
            plt.title(f'K-Means Clustering with {optimal_clusters} Clusters')
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.legend(title='Cluster')
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'kmeans_clusters_pca.png'), dpi=300)
            plt.show()
            plt.close()

            print("Cluster analysis saved.")
            logging.info("Cluster analysis saved.")
        except Exception as e:
            print(f"Error during final KMeans clustering: {e}")
            logging.error(f"Error during final KMeans clustering: {e}", exc_info=True)

        # -----------------------------
        # 6. Train-Test Split
        # -----------------------------
        # 80% train, 20% test
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            print("Train-test split completed.")
            logging.info("Train-test split completed.")
        except Exception as e:
            print(f"Error during train-test split: {e}")
            logging.error(f"Error during train-test split: {e}", exc_info=True)
            raise  # Cannot proceed without train-test split

        # -----------------------------
        # 7. Initialize Models with Default Hyperparameters
        # -----------------------------
        default_models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
            "Support Vector Regression": SVR(),
            "Neural Network": MLPRegressor(random_state=42, max_iter=1000)
        }

        # For storing default metrics and predictions
        default_results = []
        default_predictions = {}
        tuned_results = []
        tuned_predictions = {}
        best_estimators = {}

        # -----------------------------
        # 8. Train & Evaluate Each Model with Default Hyperparameters
        # -----------------------------
        print("\n--- Training and Evaluating Models with Default Hyperparameters ---\n")
        logging.info("--- Training and Evaluating Models with Default Hyperparameters ---")

        for model_name, model in default_models.items():
            try:
                print(f"Training {model_name} with default parameters...")
                logging.info(f"Training {model_name} with default parameters.")
                model.fit(X_train, y_train)  # Training on unscaled data

                # Predict on Test Set
                y_pred = model.predict(X_test)
                default_predictions[model_name] = y_pred

                # Calculate Metrics
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mse)
                mape = mean_absolute_percentage_error(y_test, y_pred) * 100  # Percentage

                default_results.append({
                    "Model": model_name,
                    "MAE": mae,
                    "MSE": mse,
                    "RMSE": rmse,
                    "MAPE (%)": mape,
                    "R²": r2
                })

                # Save Predictions
                pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).to_csv(
                    os.path.join(results_dir, f'predictions_default_{model_name.replace(" ", "_")}.csv'),
                    index=False
                )
                logging.info(f"Saved default predictions for {model_name}.")

                # -----------------------------
                # Plot Residuals
                # -----------------------------
                residuals = y_test - y_pred
                plt.figure(figsize=(6, 4))
                sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)
                plt.axhline(0, color='red', linestyle='--')
                plt.title(f'Residuals Plot for {model_name} (Default)')
                plt.xlabel('Predicted Values')
                plt.ylabel('Residuals')
                plt.tight_layout()
                plt.savefig(os.path.join(results_dir, f'residuals_default_{model_name.replace(" ", "_")}.png'), dpi=300)
                plt.show()
                plt.close()
                logging.info(f"Saved residuals plot for {model_name}.")

                # -----------------------------
                # Enhanced Residual Analysis
                # -----------------------------
                # 1. QQ Plot
                plt.figure(figsize=(6, 4))
                sns.histplot(residuals, kde=True, color='purple')
                plt.title(f'Histogram of Residuals for {model_name} (Default)')
                plt.xlabel('Residuals')
                plt.ylabel('Frequency')
                plt.tight_layout()
                plt.savefig(os.path.join(results_dir, f'histogram_residuals_default_{model_name.replace(" ", "_")}.png'), dpi=300)
                plt.show()
                plt.close()
                logging.info(f"Saved histogram of residuals for {model_name}.")

                # 2. QQ Plot
                from scipy import stats
                plt.figure(figsize=(6, 4))
                stats.probplot(residuals, dist="norm", plot=plt)
                plt.title(f'QQ Plot of Residuals for {model_name} (Default)')
                plt.tight_layout()
                plt.savefig(os.path.join(results_dir, f'qq_plot_residuals_default_{model_name.replace(" ", "_")}.png'), dpi=300)
                plt.show()
                plt.close()
                logging.info(f"Saved QQ plot of residuals for {model_name}.")

            except Exception as e:
                print(f"Error training {model_name}: {e}")
                logging.error(f"Error training {model_name}: {e}", exc_info=True)
                continue  # Proceed to the next model

        # -----------------------------
        # 9. Save Default Results DataFrame
        # -----------------------------
        default_results_df = pd.DataFrame(default_results)
        print("\nDefault Model Evaluation Results:\n")
        print(default_results_df)
        default_results_df.to_csv(os.path.join(results_dir, 'default_model_evaluation_results.csv'), index=False)
        print("Default model evaluation results saved.")
        logging.info("Default model evaluation results saved.")

        # -----------------------------
        # 10. Hyperparameter Tuning for Each Model
        # -----------------------------
        print("\n--- Hyperparameter Tuning for Each Model ---\n")
        logging.info("--- Hyperparameter Tuning for Each Model ---")

        hyperparameter_grids = {
            "Linear Regression": {},  # No hyperparameters to tune
            "Random Forest": {
                'regressor__n_estimators': [100, 200, 300],
                'regressor__max_depth': [None, 10, 20, 30],
                'regressor__min_samples_split': [2, 5, 10],
                'regressor__min_samples_leaf': [1, 2, 4],
                'regressor__bootstrap': [True, False]
            },
            "Gradient Boosting": {
                'regressor__n_estimators': [100, 200, 300],
                'regressor__learning_rate': [0.01, 0.05, 0.1],
                'regressor__max_depth': [3, 5, 7],
                'regressor__min_samples_split': [2, 5, 10],
                'regressor__min_samples_leaf': [1, 2, 4]
            },
            "Support Vector Regression": {
                'regressor__C': [0.5, 1, 5, 10],
                'regressor__epsilon': [0.05, 0.1, 0.2],
                'regressor__kernel': ['linear', 'rbf']
            },
            "Neural Network": {
                'regressor__hidden_layer_sizes': [(100,), (100, 50), (100, 100, 50)],
                'regressor__activation': ['relu', 'tanh'],  # Excluded 'logistic' to prevent saturation
                'regressor__solver': ['adam', 'lbfgs'],      # Excluded 'sgd' for better convergence
                'regressor__alpha': [0.0001, 0.001, 0.01],
                'regressor__learning_rate': ['constant', 'adaptive']
            }
        }

        for model_name, params in hyperparameter_grids.items():
            if model_name == "Linear Regression":
                print(f"\n{model_name} has no hyperparameters to tune.\n")
                logging.info(f"{model_name} has no hyperparameters to tune.")
                continue  # Skip tuning for Linear Regression

            print(f"Starting hyperparameter tuning for {model_name}...")
            logging.info(f"Starting hyperparameter tuning for {model_name}.")

            if model_name in ["Support Vector Regression", "Neural Network"]:
                # Create a pipeline with scaling for these models
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('regressor', default_models[model_name])
                ])
                search_params = params  # Already prefixed with 'regressor__'
            else:
                # No scaling for ensemble models
                pipeline = Pipeline([
                    ('regressor', default_models[model_name])
                ])
                search_params = params  # Already prefixed with 'regressor__'

            # Define GridSearchCV or RandomizedSearchCV based on model
            if model_name in ["Random Forest", "Gradient Boosting"]:
                # Use GridSearchCV for ensemble models
                search = GridSearchCV(
                    estimator=pipeline,
                    param_grid=search_params,
                    cv=5,
                    scoring='r2',
                    n_jobs=4,  # Limit to 4 parallel jobs to prevent system overload
                    verbose=0
                )
            else:
                # Use RandomizedSearchCV for models with larger hyperparameter spaces
                total_params = np.prod([len(v) for v in params.values()])
                n_iter = min(50, total_params)  # Adjust n_iter based on parameter space
                search = RandomizedSearchCV(
                    estimator=pipeline,
                    param_distributions=search_params,
                    n_iter=n_iter,
                    cv=5,
                    scoring='r2',
                    random_state=42,
                    n_jobs=4,  # Limit to 4 parallel jobs
                    verbose=0
                )

            # Perform Hyperparameter Tuning inside a try-except to catch potential errors
            try:
                search.fit(X_train, y_train)  # Pipeline handles scaling internally
                best_model = search.best_estimator_
                best_estimators[model_name] = best_model
                print(f"Best parameters for {model_name}: {search.best_params_}")
                logging.info(f"Best parameters for {model_name}: {search.best_params_}")
            except Exception as e:
                print(f"Error during hyperparameter tuning for {model_name}: {e}")
                logging.error(f"Error during hyperparameter tuning for {model_name}: {e}", exc_info=True)
                continue  # Skip to the next model

            # Predict on Test Set with Tuned Model
            try:
                y_pred_tuned = best_model.predict(X_test)
                tuned_predictions[model_name] = y_pred_tuned
            except Exception as e:
                print(f"Error during prediction with tuned {model_name}: {e}")
                logging.error(f"Error during prediction with tuned {model_name}: {e}", exc_info=True)
                continue  # Skip to the next model

            # Calculate Metrics
            try:
                mae_tuned = mean_absolute_error(y_test, y_pred_tuned)
                mse_tuned = mean_squared_error(y_test, y_pred_tuned)
                r2_tuned = r2_score(y_test, y_pred_tuned)
                rmse_tuned = np.sqrt(mse_tuned)
                mape_tuned = mean_absolute_percentage_error(y_test, y_pred_tuned) * 100  # Percentage
            except Exception as e:
                print(f"Error calculating metrics for tuned {model_name}: {e}")
                logging.error(f"Error calculating metrics for tuned {model_name}: {e}", exc_info=True)
                continue  # Skip to the next model

            tuned_results.append({
                "Model": model_name,
                "MAE": mae_tuned,
                "MSE": mse_tuned,
                "RMSE": rmse_tuned,
                "MAPE (%)": mape_tuned,
                "R²": r2_tuned
            })

            # Save Tuned Predictions
            pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_tuned}).to_csv(
                os.path.join(results_dir, f'predictions_tuned_{model_name.replace(" ", "_")}.csv'),
                index=False
            )
            logging.info(f"Saved tuned predictions for {model_name}.")

            # -----------------------------
            # Plot Residuals for Tuned Model
            # -----------------------------
            try:
                residuals_tuned = y_test - y_pred_tuned
                plt.figure(figsize=(6, 4))
                sns.scatterplot(x=y_pred_tuned, y=residuals_tuned, alpha=0.6)
                plt.axhline(0, color='red', linestyle='--')
                plt.title(f'Residuals Plot for {model_name} (Tuned)')
                plt.xlabel('Predicted Values')
                plt.ylabel('Residuals')
                plt.tight_layout()
                plt.savefig(os.path.join(results_dir, f'residuals_tuned_{model_name.replace(" ", "_")}.png'), dpi=300)
                plt.show()
                plt.close()
                logging.info(f"Saved residuals plot for tuned {model_name}.")
            except Exception as e:
                print(f"Error plotting residuals for tuned {model_name}: {e}")
                logging.error(f"Error plotting residuals for tuned {model_name}: {e}", exc_info=True)

            # -----------------------------
            # Plot Learning Curves for Tuned Model
            # -----------------------------
            try:
                train_sizes, train_scores, test_scores = learning_curve(
                    best_model,
                    X_train,
                    y_train,
                    cv=5,
                    scoring='r2',
                    n_jobs=4,
                    train_sizes=np.linspace(0.1, 1.0, 10)
                )
                train_mean = np.mean(train_scores, axis=1)
                train_std = np.std(train_scores, axis=1)
                test_mean = np.mean(test_scores, axis=1)
                test_std = np.std(test_scores, axis=1)

                plt.figure(figsize=(6, 4))
                plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
                plt.plot(train_sizes, test_mean, 'o-', color='green', label='Cross-validation score')
                plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
                plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='green')
                plt.title(f'Learning Curve for {model_name} (Tuned)')
                plt.xlabel('Training Set Size')
                plt.ylabel('R² Score')
                plt.legend(loc='best')
                plt.tight_layout()
                plt.savefig(os.path.join(results_dir, f'learning_curve_tuned_{model_name.replace(" ", "_")}.png'), dpi=300)
                plt.show()
                plt.close()
                logging.info(f"Saved learning curve for tuned {model_name}.")
            except Exception as e:
                print(f"Error plotting learning curve for tuned {model_name}: {e}")
                logging.error(f"Error plotting learning curve for tuned {model_name}: {e}", exc_info=True)

        # -----------------------------
        # 11. Save Tuned Results DataFrame
        # -----------------------------
        tuned_results_df = pd.DataFrame(tuned_results)
        print("\nTuned Model Evaluation Results:\n")
        print(tuned_results_df)
        if not tuned_results_df.empty:
            tuned_results_df.to_csv(os.path.join(results_dir, 'tuned_model_evaluation_results.csv'), index=False)
            print("Tuned model evaluation results saved.")
            logging.info("Tuned model evaluation results saved.")
        else:
            print("No tuned model results to save.")
            logging.warning("No tuned model results to save.")

        # -----------------------------
        # 12. Combine Default and Tuned Results for Comparison
        # -----------------------------
        if not tuned_results_df.empty:
            comparison_df = pd.merge(
                default_results_df,
                tuned_results_df,
                on='Model',
                how='left',
                suffixes=('_Default', '_Tuned')
            )
            comparison_df.to_csv(os.path.join(results_dir, 'models_comparison.csv'), index=False)
            print("Models comparison results saved.")
            logging.info("Models comparison results saved.")
        else:
            print("No tuned models to compare. Skipping comparison DataFrame creation.")
            logging.warning("No tuned models to compare. Skipping comparison DataFrame creation.")

        # -----------------------------
        # 13. Choose Best Model based on Tuned R²
        # -----------------------------
        if not tuned_results_df.empty:
            # Check if 'Model' column exists in tuned_results_df
            if 'Model' in tuned_results_df.columns:
                # Identify the best model based on Tuned R²
                best_model_row = tuned_results_df.sort_values(by='R²', ascending=False).iloc[0]
                best_model_name = best_model_row['Model']
                best_model = best_estimators[best_model_name]
                print(f"\nBest model based on Tuned R² is: {best_model_name}\n")
                logging.info(f"Best model based on Tuned R² is: {best_model_name}")
            else:
                print("\nError: 'Model' column not found in tuned_results_df. Selecting best model from default results.\n")
                logging.error("Error: 'Model' column not found in tuned_results_df. Selecting best model from default results.")
                best_model_row = default_results_df.sort_values(by='R²', ascending=False).iloc[0]
                best_model_name = best_model_row['Model']
                best_model = default_models[best_model_name]
                print(f"Best model based on Default R² is: {best_model_name} (Default)\n")
                logging.info(f"Best model based on Default R² is: {best_model_name} (Default)")
        else:
            # If no models were tuned, select the best from default models
            best_model_row = default_results_df.sort_values(by='R²', ascending=False).iloc[0]
            best_model_name = best_model_row['Model']
            best_model = default_models[best_model_name]
            print(f"\nBest model based on Default R² is: {best_model_name} (Default)\n")
            logging.info(f"Best model based on Default R² is: {best_model_name} (Default)")

        # -----------------------------
        # 14. Plot Metrics Comparison
        # -----------------------------
        # Only plot comparison if there are tuned results
        if not tuned_results_df.empty:
            metrics = ['MAE', 'MSE', 'RMSE', 'MAPE (%)', 'R²']
            for metric in metrics:
                plt.figure(figsize=(10, 6))
                # Plot Default Metrics
                sns.barplot(
                    x='Model',
                    y=f'{metric}_Default',
                    data=comparison_df,
                    color='skyblue',
                    label='Default'
                )
                # Plot Tuned Metrics
                sns.barplot(
                    x='Model',
                    y=f'{metric}_Tuned',
                    data=comparison_df,
                    color='salmon',
                    alpha=0.7,
                    label='Tuned'
                )
                plt.title(f'Model Comparison by {metric}')
                plt.xlabel('Model')
                plt.ylabel(metric)
                plt.xticks(rotation=45, ha='right')
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(results_dir, f'model_comparison_{metric}.png'), dpi=300)
                plt.show()
                plt.close()
                logging.info(f"Saved model comparison plot for {metric}.")
        else:
            print("No tuned models available for metrics comparison plots.")
            logging.warning("No tuned models available for metrics comparison plots.")

        # -----------------------------
        # 15. Feature Importances: Applicable Models
        # -----------------------------
        for model_name in ["Random Forest", "Gradient Boosting"]:
            if model_name in best_estimators:
                model = best_estimators[model_name]
                try:
                    if hasattr(model, 'named_steps'):
                        regressor = model.named_steps['regressor']
                    else:
                        regressor = model
                    if hasattr(regressor, 'feature_importances_'):
                        importances = regressor.feature_importances_
                    else:
                        print(f"{model_name} does not have feature_importances_ attribute.")
                        logging.warning(f"{model_name} does not have feature_importances_ attribute.")
                        continue

                    feature_importances = pd.Series(importances, index=X.columns).sort_values(ascending=False)

                    plt.figure(figsize=(10, 8))
                    sns.barplot(x=feature_importances, y=feature_importances.index, palette='magma', dodge=False)
                    plt.title(f'{model_name} Feature Importances')
                    plt.xlabel('Importance Score')
                    plt.ylabel('Feature')
                    plt.tight_layout()
                    plt.savefig(os.path.join(results_dir, f'{model_name.lower().replace(" ", "_")}_feature_importances.png'), dpi=300)
                    plt.show()
                    plt.close()

                    # Save Feature Importances
                    feature_importances.to_csv(os.path.join(results_dir, f'{model_name.lower().replace(" ", "_")}_feature_importances.csv'))
                    print(f"{model_name} feature importances saved.")
                    logging.info(f"{model_name} feature importances saved.")
                except Exception as e:
                    print(f"Error extracting feature importances for {model_name}: {e}")
                    logging.error(f"Error extracting feature importances for {model_name}: {e}", exc_info=True)

        # -----------------------------
        # 16. Partial Dependence Plots (PDP) for the Best Model
        # -----------------------------
        # Only generate PDP if the best model is not Linear Regression
        if best_model_name not in ["Linear Regression"]:
            # Identify top 3 important features
            if best_model_name in ["Random Forest", "Gradient Boosting"]:
                importances = best_estimators[best_model_name].named_steps['regressor'].feature_importances_
                feature_importances = pd.Series(importances, index=X.columns).sort_values(ascending=False)
            else:
                # For models without feature_importances_, skip PDP
                feature_importances = pd.Series(dtype=float)

            top_features = feature_importances.head(3).index.tolist() if not feature_importances.empty else X.columns[:3].tolist()
            print(f"Generating Partial Dependence Plots for top features: {top_features}")
            logging.info(f"Generating Partial Dependence Plots for top features: {top_features}")

            # Generate PDP
            try:
                fig, ax = plt.subplots(figsize=(12, 4))
                display = PartialDependenceDisplay.from_estimator(
                    best_model,
                    X_train,
                    features=top_features,
                    ax=ax,
                    kind='average'
                )
                plt.tight_layout()
                plt.savefig(os.path.join(results_dir, f'pdp_best_model.png'), dpi=300)
                plt.show()
                plt.close()

                # Note: The 'partial_dependence' function may not return 'values' in newer versions.
                # Using PartialDependenceDisplay handles plotting internally.
                # Therefore, no need to manually extract 'values'.

                print("Partial Dependence Plots saved.")
                logging.info("Partial Dependence Plots saved.")
            except Exception as e:
                print(f"Error generating Partial Dependence Plots for {best_model_name}: {e}")
                logging.error(f"Error generating Partial Dependence Plots for {best_model_name}: {e}", exc_info=True)
        else:
            print("Best model is Linear Regression. Skipping Partial Dependence Plots.")
            logging.info("Best model is Linear Regression. Skipping Partial Dependence Plots.")

        # -----------------------------
        # 17. SHAP Analysis for the Best Model
        # -----------------------------
        print("\n--- Performing SHAP Analysis for the Best Model ---\n")
        logging.info("--- Performing SHAP Analysis for the Best Model ---")

        try:
            # Initialize SHAP explainer
            if best_model_name in ["Support Vector Regression", "Neural Network"]:
                # Models within a pipeline require accessing the regressor
                regressor = best_model.named_steps['regressor']
                scaler = best_model.named_steps['scaler']
                explainer = shap.Explainer(regressor, scaler.transform(X_train))
                shap_values = explainer(scaler.transform(X_train))
            else:
                regressor = best_model.named_steps['regressor'] if hasattr(best_model, 'named_steps') else best_model
                explainer = shap.Explainer(regressor, X_train)
                shap_values = explainer(X_train)

            # Global SHAP Summary Plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
            plt.title(f'SHAP Summary Plot for {best_model_name} (Global Importance)')
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f'shap_summary_global_{best_model_name.replace(" ", "_")}.png'), dpi=300)
            plt.show()
            plt.close()
            logging.info(f"SHAP summary plot saved for {best_model_name}.")

            # SHAP Summary Plot (Detailed)
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_train, show=False)
            plt.title(f'SHAP Summary Plot for {best_model_name} (Detailed)')
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f'shap_summary_detailed_{best_model_name.replace(" ", "_")}.png'), dpi=300)
            plt.show()
            plt.close()
            logging.info(f"SHAP detailed summary plot saved for {best_model_name}.")

            # SHAP Dependence Plot for Top Feature
            top_feature = top_features[0] if top_features else X.columns[0]
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(top_feature, shap_values.values, X_train, show=False)
            plt.title(f'SHAP Dependence Plot for {top_feature} in {best_model_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f'shap_dependence_{top_feature}_{best_model_name.replace(" ", "_")}.png'), dpi=300)
            plt.show()
            plt.close()
            logging.info(f"SHAP dependence plot saved for feature {top_feature} in {best_model_name}.")

        except Exception as e:
            print(f"Error during SHAP analysis for {best_model_name}: {e}")
            logging.error(f"Error during SHAP analysis for {best_model_name}: {e}", exc_info=True)

        # -----------------------------
        # 18. Save Best Model Parameters
        # -----------------------------
        try:
            best_model_params = best_model.get_params()
            pd.Series(best_model_params).to_csv(os.path.join(results_dir, f'best_model_{best_model_name.replace(" ", "_")}_parameters.csv'))
            print(f"Best model ({best_model_name}) parameters saved.")
            logging.info(f"Best model ({best_model_name}) parameters saved.")
        except Exception as e:
            print(f"Error saving parameters for {best_model_name}: {e}")
            logging.error(f"Error saving parameters for {best_model_name}: {e}", exc_info=True)

        # -----------------------------
        # 19. Persist the Best Model
        # -----------------------------
        try:
            joblib.dump(best_model, os.path.join(results_dir, f'best_model_{best_model_name.replace(" ", "_")}.joblib'))
            print(f"Best model ({best_model_name}) saved using joblib.")
            logging.info(f"Best model ({best_model_name}) saved using joblib.")
        except Exception as e:
            print(f"Error saving best model ({best_model_name}): {e}")
            logging.error(f"Error saving best model ({best_model_name}): {e}", exc_info=True)

        # -----------------------------
        # 20. Predict on a New Structure
        # -----------------------------
        # Define your new structure data here with exact feature names as in X.columns
        new_structure_data = {
            "HOMO-O (Hartree)": [-0.32608],
            "LUMO-O (Hartree)": [-0.03573],
            "Dipole Moment X (Debye)": [3.0385],
            "Dipole Moment Total (Debye)": [10.4733],
            "D16 value": [3.88178],
            "I-O Bond": [2.173799731],
            "I=O": [1.819830622],
            "I-OH bond": [2.002986092]
        }

        # Only run this if you have at least these columns in X
        if len(new_structure_data) > 0:
            new_df = pd.DataFrame(new_structure_data)
            # Ensure new_df's columns match the order/names in X
            missing_cols = set(X.columns) - set(new_df.columns)
            extra_cols = set(new_df.columns) - set(X.columns)
            if missing_cols:
                print("Warning! Missing columns:", missing_cols)
                logging.warning(f"Missing columns in new structure data: {missing_cols}")
            if extra_cols:
                print("Warning! Extra columns not in training set:", extra_cols)
                logging.warning(f"Extra columns in new structure data not in training set: {extra_cols}")

            # Handle missing columns by filling with mean or other strategy
            for col in missing_cols:
                new_df[col] = X[col].mean()

            # Ensure the order of columns
            new_df = new_df[X.columns]

            # Predict with the best model
            try:
                if best_model_name in ["Support Vector Regression", "Neural Network"]:
                    # Use the pipeline for these models
                    ts_pred = best_model.predict(new_df)
                else:
                    # Direct prediction for ensemble models
                    ts_pred = best_model.predict(new_df)

                print("--- Prediction for New Structure(s) ---")
                print("Input Features:\n", new_df)
                print("Predicted TS_Barrier:", ts_pred)
                print("----------------------------------------\n")
                logging.info("Prediction for new structure completed.")

                # Save Prediction
                prediction_df = new_df.copy()
                prediction_df['Predicted_TS_Barrier'] = ts_pred
                prediction_df.to_csv(os.path.join(results_dir, 'new_structure_prediction.csv'), index=False)
                print("Prediction for new structure saved.")
                logging.info("Prediction for new structure saved.")
            except Exception as e:
                print(f"Error making prediction for new structure: {e}")
                logging.error(f"Error making prediction for new structure: {e}", exc_info=True)
        else:
            print("No manual feature dictionary provided. Please fill 'new_structure_data' accordingly.")
            logging.warning("No manual feature dictionary provided. Please fill 'new_structure_data' accordingly.")
    else:
        print("No data was loaded.")
        logging.error("No data was loaded.")
