# src/explanations.py

import shap
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import RandomForestClassifier

def global_shap_explanation(model: RandomForestClassifier, X: pd.DataFrame) -> shap.Explanation:
    """
    Generate SHAP values for global model interpretation.
    :param model: Trained model.
    :param X: Feature matrix.
    :return: SHAP explanation.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return shap_values

def local_lime_explanation(model: RandomForestClassifier, X_train: pd.DataFrame, instance: np.ndarray) -> None:
    """
    Generate LIME explanation for a single prediction.
    :param model: Trained model.
    :param X_train: Training data.
    :param instance: Instance for which explanation is needed.
    """
    explainer = LimeTabularExplainer(X_train.values, mode='classification')
    explanation = explainer.explain_instance(instance, model.predict_proba)
    explanation.show_in_notebook()
