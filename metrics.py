"""
metrics.py
Calculates all evaluation metrics including Total Misclassification Cost.
"""

from sklearn.metrics import (
    balanced_accuracy_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
import numpy as np


def calculate_metrics(y_true, y_pred, fn_cost: float = 10.0, fp_cost: float = 1.0):
    """
    Calculate metrics and Total Misclassification Cost.

    Total Cost = (FN x fn_cost) + (FP x fp_cost)

    FN = False Negatives (missed targets — model said No, answer was Yes)
    FP = False Positives (false alarms — model said Yes, answer was No)
    """
    cm = confusion_matrix(y_true, y_pred)

    if cm.size == 1:
        if y_true[0] == 0:
            tn, fp, fn, tp = int(cm[0, 0]), 0, 0, 0
        else:
            tn, fp, fn, tp = 0, 0, 0, int(cm[0, 0])
    else:
        tn, fp, fn, tp = cm.ravel()

    total_cost = (fn * fn_cost) + (fp * fp_cost)

    return {
        "Confusion Matrix": cm,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "Total Cost": total_cost,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
    }
