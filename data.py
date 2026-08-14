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
    """
    df = pd.read_csv(uploaded_file)
    target_name = df.columns[-1]
    feature_names = df.columns[:-1].tolist()

    X = df[feature_names].values
    y = df[target_name].values

    unique = np.unique(y)
    if not set(unique).issubset({0, 1}):
        raise ValueError("Target column must contain only 0 and 1.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test, feature_names, target_name
