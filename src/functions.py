from pathlib import Path
import joblib
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet, BayesianRidge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.utils import resample


class BaselineBMIPredictor:
    """
    A class to implement baseline BMI prediction models using metagenomic data.
    Only handles training with default parameters, assumes data is preprocessed.
    """

    def __init__(self):
        """
        Initialize the BMI predictor.

        Args:
            models_dir (str): Directory to save trained models
        """

        self.models = {
            'ElasticNet': ElasticNet(),
            'SVR': SVR(),
            'BayesianRidge': BayesianRidge()
        }

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
        for model_name, model in self.models.items():
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
            'MAE': []
        }
        self.stats = None
        self.ci = 95

    def evaluate(self, X, y, n_iterations=200, random_state=42):
        """
        Run bootstrapped evaluation (stores results internally).

        Args:
            X: Evaluation features
            y: True labels
            n_iterations: Bootstrap samples (200-1000 recommended)
            ci: Confidence interval width (default 95%)
            random_state: Reproducibility seed
        """
        np.random.seed(random_state)
        for _ in range(n_iterations):
            X_bs, y_bs = resample(X, y)
            y_pred = self.model.predict(X_bs)

            self.metrics['RMSE'].append(
                np.sqrt(mean_squared_error(y_bs, y_pred)))
            self.metrics['MAE'].append(mean_absolute_error(y_bs, y_pred))

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
        if not self.stats:
            raise ValueError("Run evaluate() first")

        report_df = pd.DataFrame.from_dict(self.stats, orient='index')

        report_df[['CI_low', 'CI_high']] = pd.DataFrame(
            report_df[f'CI_{95}'].tolist(),
            index=report_df.index
        )

        report_df = report_df.round(3)

        return report_df[['mean', 'std', 'median', 'CI_low', 'CI_high']]


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
