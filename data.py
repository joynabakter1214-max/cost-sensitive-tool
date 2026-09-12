"""
data.py
Handles synthetic data, Credit Card Fraud dataset, and custom CSV upload.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import os


def generate_synthetic_data(
    n_samples: int = 1000,
    imbalance_ratio: float = 0.95,
    class_sep: float = 0.8,
    random_state: int = 42
):
    """Generate controllable synthetic data and split into train/test."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=1,
        weights=[imbalance_ratio, 1 - imbalance_ratio],
        class_sep=class_sep,
        random_state=random_state
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def load_creditcard_data(sample_size: int = 30000, random_state: int = 42):
    """
    Load Credit Card Fraud dataset (sampled for speed).
    Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
    Place creditcard.csv in the same folder as app.py before using this option.
    """
    file_path = "creditcard.csv"

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            "creditcard.csv not found.\n"
            "Download it from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
            "and place it in the same folder as app.py"
        )

    df = pd.read_csv(file_path)

    if "Class" not in df.columns:
        raise ValueError("Expected column 'Class' not found in creditcard.csv")

    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=random_state)

    y = df["Class"].values
    X = df.drop(columns=["Class"]).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def load_custom_data(uploaded_file, test_size: float = 0.30, random_state: int = 42):
    """
    Load user-uploaded CSV.
    The last column is treated as the target (must contain only 0 and 1).
    Validates common mistakes with specific, actionable error messages,
    rather than letting a confusing internal error surface to the user.
    """
    df = pd.read_csv(uploaded_file)

    if df.shape[1] < 2:
        raise ValueError(
            f"Your file only has {df.shape[1]} column(s). You need at least one feature "
            "column plus a target column (2 columns minimum)."
        )

    target_name = df.columns[-1]
    feature_names = df.columns[:-1].tolist()

    X_df = df[feature_names]
    y_df = df[target_name]

    if X_df.isnull().values.any() or y_df.isnull().values.any():
        raise ValueError(
            "Your file contains missing (empty) cells. Please fill them in or remove those "
            "rows before uploading."
        )

    non_numeric = [c for c in feature_names if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(
            f"These feature column(s) contain non-numeric data: {', '.join(non_numeric)}. "
            "All feature columns must contain only numbers (convert text categories to "
            "numbers first, or remove those columns)."
        )

    unique = sorted(pd.unique(y_df))
    if not set(unique).issubset({0, 1}):
        raise ValueError(
            f"Your target column ('{target_name}') contains these values: {unique}. "
            "It must contain only 0 and 1."
        )

    X = X_df.values
    y = y_df.values

    smallest_class_count = int(min((y == 0).sum(), (y == 1).sum()))
    if smallest_class_count < 2:
        raise ValueError(
            f"Your rarer class only has {smallest_class_count} example(s) in the target "
            f"column ('{target_name}'). You need at least 2 examples of each class (0 and 1) "
            "to split the data into training and test sets."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test, feature_names, target_name
