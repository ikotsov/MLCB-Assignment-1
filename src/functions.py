from pathlib import Path
import joblib
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet, BayesianRidge
from sklearn.svm import SVR
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.utils import resample
from sklearn.model_selection import RandomizedSearchCV


class BMIPredictor:
    """
    A class to implement baseline BMI prediction models using metagenomic data.
    Only handles training with default parameters, assumes data is preprocessed.
    """

    def __init__(self, model_registry=None):
        """
        Initialize the BMI predictor.

        Args:
            model_registry: Optional ModelRegistry instance. Creates default one if None.
        """
        self.registry = model_registry if model_registry else ModelRegistry()

    def fit(self, X_train, y_train):
        """
        Train all baseline models with default parameters.

        Args:
            X_train (array-like): Training features
            y_train (array-like): Training targets

        Returns:
            dict: Dictionary of trained models
        """

        trained_models = {}
        for model_name in self.registry.available_models:
            model = self.registry.get_model(model_name)
            model.fit(X_train, y_train)
            trained_models[model_name] = model

        return trained_models


class ModelEvaluator:
    """Evaluates a single model with bootstrapped metrics."""

    def __init__(self, model, model_name=""):
        """
        Args:
            model: Pre-trained model object
            model_name (str): Identifier for reports
        """
        self.model = model
        self.model_name = model_name
        self.metrics = {
            'RMSE': [],
            'MAE': [],
            'R2': []
        }
        self.stats = None
        self.ci = 95

    def evaluate(self, X, y, n_iterations=500, random_state=42):
        """
        Run bootstrapped evaluation (stores results internally).

        Args:
            X: Evaluation features
            y: True labels
            n_iterations: Bootstrap samples (200-1000 recommended)
            random_state: Reproducibility seed
        """
        np.random.seed(random_state)
        for _ in range(n_iterations):
            X_bs, y_bs = resample(X, y)
            y_pred = self.model.predict(X_bs)

            self.metrics['RMSE'].append(root_mean_squared_error(y_bs, y_pred))
            self.metrics['MAE'].append(mean_absolute_error(y_bs, y_pred))
            self.metrics['R2'].append(r2_score(y_bs, y_pred))

        self.stats = self.__compute_statistics()

    def __compute_statistics(self):
        """Calculate and store statistics."""
        ci_low = (100 - self.ci) / 2
        return {
            metric: {
                'mean': np.mean(values),
                'median': np.median(values),
                f'CI_{self.ci}': (np.percentile(values, ci_low),
                                  np.percentile(values, 100 - ci_low)),
                'std': np.std(values)
            }
            for metric, values in self.metrics.items()
        }

    def generate_report(self):
        """Generate a DataFrame report with all metrics (RMSE, MAE, R2)."""
        if not self.stats:
            raise ValueError("Run evaluate() first")

        report_df = pd.DataFrame.from_dict(self.stats, orient='index')

        report_df[['CI_low', 'CI_high']] = pd.DataFrame(
            report_df[f'CI_{95}'].tolist(),
            index=report_df.index
        )

        report_df = report_df.round(3)

        return report_df[['mean', 'std', 'median', 'CI_low', 'CI_high']]


class ModelTuner:
    """
    Handles hyperparameter tuning for BMI prediction models using selected features.
    """

    def __init__(self, scoring_metric='neg_root_mean_squared_error', cv=5, model_registry=None):
        """
        Args:
            scoring_metric: The scoring metric to use for evaluation (default: 'neg_root_mean_squared_error')
            cv: Number of cross-validation folds
        """
        self.registry = model_registry if model_registry else ModelRegistry()
        self.scoring_metric = scoring_metric
        self.cv = cv

        # Define hyperparameter grids for each model
        self.param_grids = {
            'ElasticNet': {
                'alpha': [0.1, 0.5, 1.0, 2.0],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]  # L1 vs L2 mix
            },
            'SVR': {
                'C': [0.1, 1, 10, 100],
                'epsilon': [0.01, 0.1, 0.5],
                'kernel': ['linear', 'rbf']
            },
            'BayesianRidge': {
                'alpha_1': [1e-6, 1e-5, 1e-4],
                'alpha_2': [1e-6, 1e-5, 1e-4],
                'lambda_1': [1e-6, 1e-5, 1e-4],
                'lambda_2': [1e-6, 1e-5, 1e-4]
            }
        }

        self.best_params_ = {}
        self.best_scores_ = {}

    def tune(self, model_name, X_train, y_train, n_iterations=50, random_state=42):
        """
        Perform hyperparameter tuning using randomized search.

        Args:
            model_name: One of ['ElasticNet', 'SVR', 'BayesianRidge']
            X_train: Training features
            y_train: Target values
            n_iterations: Number of parameter combinations to try
            random_state: Random seed

        Returns:
            Trained model with best parameters
        """
        model = self.registry.get_model(model_name)

        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=self.param_grids[model_name],
            n_iter=n_iterations,
            cv=self.cv,
            scoring=self.scoring_metric,
            random_state=random_state,
            n_jobs=-1  # Use all processors
        )

        search.fit(X_train, y_train)

        self.best_params_[model_name] = search.best_params_
        self.best_scores_[model_name] = - \
            search.best_score_  # Convert to positive RMSE

        return search.best_estimator_


class ModelRegistry:
    """
    Central registry for model definitions.
    """

    def __init__(self):
        self._model_classes = {
            'ElasticNet': ElasticNet,
            'SVR': SVR,
            'BayesianRidge': BayesianRidge
        }

    @property
    def available_models(self):
        """Return list of registered model names."""
        return list(self._model_classes.keys())

    def get_model(self, model_name):
        """
        Get a model instance.

        Args:
            model_name: Name of the model to retrieve

        Returns:
            Model instance (new instance each call)
        """
        if model_name not in self._model_classes:
            raise ValueError(
                f"Unknown model: {model_name}. Available: {self.available_models}")

        return self._model_classes[model_name]()


class ModelIO:
    """
    Handles saving and loading of trained models with consistent naming conventions.
    """

    def __init__(self, models_dir):
        """
        Initialize with directory for model storage.

        Args:
            models_dir (str): Directory to save/load models
        """
        if models_dir is None:
            raise ValueError("models_dir must be provided, cannot be None")

        self.models_dir = Path(models_dir)
        os.makedirs(self.models_dir, exist_ok=True)

    def save_model(self, model, model_name, suffix=''):
        """
        Save a trained model to disk with standardized naming.

        Args:
            model: Trained model object
            model_name (str): Name of the model (e.g., 'ElasticNet')
            suffix (str): Optional suffix for filename (e.g., 'baseline', 'tuned')
        """
        filename = self.__generate_filename(model_name, suffix)
        joblib.dump(model, filename)

    def load_model(self, model_name, suffix=''):
        """
        Load a saved model from disk.

        Args:
            model_name (str): Name of the model
            suffix (str): Optional suffix used when saving

        Returns:
            The loaded model object
        """
        filename = self.__generate_filename(model_name, suffix)
        return joblib.load(filename)

    def __generate_filename(self, model_name, suffix):
        """
        Generate consistent filenames for models.

        Args:
            model_name (str): Name of the model
            suffix (str): Optional suffix

        Returns:
            Path: Full path to model file
        """
        parts = [model_name]
        if suffix:
            parts.append(suffix)
        return self.models_dir / f"{'_'.join(parts)}.pkl"


MODEL_COLORS = {
    'ElasticNet': 'lightblue',
    'SVR': 'lightgreen',
    'BayesianRidge': 'lightyellow'
}


def compare_model_metrics(*evaluators, figsize=(18, 6)):
    """
    Display side-by-side boxplots comparing models' metrics.

    Args:
        *evaluators: ModelEvaluator instances to compare
        figsize: Tuple specifying figure dimensions
    """
    plt.figure(figsize=figsize)

    # Prepare data and labels
    metrics = {
        'RMSE': [],
        'MAE': [],
        'R2': []
    }
    labels = [e.model_name for e in evaluators]

    for evaluator in evaluators:
        metrics['RMSE'].append(evaluator.metrics['RMSE'])
        metrics['MAE'].append(evaluator.metrics['MAE'])
        metrics['R2'].append(evaluator.metrics['R2'])

    # Create subplots
    for i, (metric_name, metric_values) in enumerate(metrics.items(), 1):
        plt.subplot(1, 3, i)
        boxes = plt.boxplot(
            metric_values,
            labels=labels,
            patch_artist=True,
            widths=0.6
        )

        # Apply colors
        for box, label in zip(boxes['boxes'], labels):
            # Default to gray if color not defined
            box.set_facecolor(MODEL_COLORS.get(label, 'gray'))

        plt.title(f'{metric_name} Comparison')
        plt.ylabel('Score' if metric_name == 'R2' else 'Error Value')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
