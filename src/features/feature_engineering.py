"""
Feature engineering pipeline for churn prediction.
Best practice: keep all feature logic here, not in notebooks.
"""
import pandas as pd
import numpy as np
from typing import Tuple


def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load and do initial cleaning of the Telco churn dataset."""
    df = pd.read_csv(filepath)
    # Fix TotalCharges (blank strings for new customers)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(0, inplace=True)
    # Binary encode target
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create business-meaningful features.
    Each feature should have a clear business rationale.
    """
    df = df.copy()

    # --- Convert TotalCharges to numeric (it arrives as string in raw data) ---
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    # --- Tenure buckets (customer lifecycle stage) ---
    df["tenure_bucket"] = pd.cut(
        df["tenure"],
        bins=[0, 6, 12, 24, 48, 72],
        labels=["0-6m", "6-12m", "1-2y", "2-4y", "4y+"]
    )

    # --- Monthly spend ratio (high bill vs. what they use) ---
    df["spend_per_month_ratio"] = df["MonthlyCharges"] / (df["TotalCharges"] + 1)

    # --- Service count (more services = higher switching cost) ---
    service_cols = [
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    binary_services = df[service_cols].apply(
        lambda col: (col != "No").astype(int)
    )
    df["service_count"] = binary_services.sum(axis=1)

    # --- Contract risk flag (month-to-month = highest churn risk) ---
    df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)

    # --- Paperless billing + electronic payment combo ---
    df["digital_only"] = (
        (df["PaperlessBilling"] == "Yes") &
        (df["PaymentMethod"].str.contains("Electronic", na=False))
    ).astype(int)

    return df


def get_feature_columns() -> Tuple[list, list]:
    """Return (numeric_cols, categorical_cols) for the model pipeline."""
    numeric = [
        "tenure", "MonthlyCharges", "TotalCharges",
        "service_count", "spend_per_month_ratio",
        "is_month_to_month", "digital_only"
    ]
    categorical = [
        "gender", "SeniorCitizen", "Partner", "Dependents",
        "InternetService", "Contract", "PaymentMethod", "tenure_bucket"
    ]
    return numeric, categorical
