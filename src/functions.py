from pathlib import Path
from sklearn.linear_model import ElasticNet, BayesianRidge
from sklearn.svm import SVR
import joblib
import os


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
