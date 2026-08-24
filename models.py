"""
models.py
Implements the three cost-sensitive strategies.
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
import numpy as np


def get_base_model(model_type: str):
    """Return the chosen model class and its default parameters."""
    if model_type == "Decision Tree":
        return DecisionTreeClassifier, {"max_depth": 5, "random_state": 42}
    elif model_type == "Logistic Regression":
        return LogisticRegression, {"max_iter": 1000, "random_state": 42}
    elif model_type == "Random Forest":
        return RandomForestClassifier, {"n_estimators": 100, "max_depth": 5, "random_state": 42}
    elif model_type == "SVM":
        return SVC, {"probability": True, "random_state": 42}
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_model_by_strategy(
    X_train, y_train,
    strategy: str,
    model_type: str = "Logistic Regression",
    fn_cost: float = 10.0,
    fp_cost: float = 1.0
):
    """
    Train a model according to the selected cost-sensitive strategy.

    Returns:
        model: the fitted model
        threshold: the decision threshold to use at prediction time

    Strategies:
        Threshold Moving  — train normally, adjust threshold using Elkan's formula
        Class Weighting   — pass cost ratio as class_weight during training
        Resampling (SMOTE) — oversample minority class before training
    """
    BaseModel, params = get_base_model(model_type)

    # Strategy 1 — Threshold Moving (post-training)
    if strategy == "Threshold Moving":
        model = BaseModel(**params)
        model.fit(X_train, y_train)
        # Elkan (2001): optimal threshold = FP_cost / (FP_cost + FN_cost)
        optimal_threshold = fp_cost / (fp_cost + fn_cost)
        return model, optimal_threshold

    # Strategy 2 — Class Weighting (during training)
    elif strategy == "Class Weighting":
        class_weight = {0: fp_cost, 1: fn_cost}
        model = BaseModel(**{**params, "class_weight": class_weight})
        model.fit(X_train, y_train)
        return model, 0.5

    # Strategy 3 — Resampling with SMOTE (pre-training)
    elif strategy == "Resampling (SMOTE)":
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        model = BaseModel(**params)
        model.fit(X_resampled, y_resampled)
        return model, 0.5

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def predict_with_threshold(model, X, threshold: float = 0.5):
    """
    Apply a custom probability threshold to make predictions.

    If the model supports predict_proba, use probabilities with the threshold.
    Otherwise fall back to standard predict.
    """
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[:, 1]
        return (probabilities >= threshold).astype(int)
    else:
        return model.predict(X)
