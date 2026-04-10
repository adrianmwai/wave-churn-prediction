import pytest
import pandas as pd
from src.features.feature_engineering import engineer_features, get_feature_columns

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "customerID": ["123-ABC"],
        "gender": ["Male"],
        "SeniorCitizen": [0],
        "Partner": ["Yes"],
        "Dependents": ["No"],
        "tenure": [12],
        "PhoneService": ["Yes"],
        "MultipleLines": ["No"],
        "InternetService": ["Fiber optic"],
        "OnlineSecurity": ["No"],
        "OnlineBackup": ["Yes"],
        "DeviceProtection": ["No"],
        "TechSupport": ["No"],
        "StreamingTV": ["No"],
        "StreamingMovies": ["Yes"],
        "Contract": ["Month-to-month"],
        "PaperlessBilling": ["Yes"],
        "PaymentMethod": ["Electronic check"],
        "MonthlyCharges": [79.85],
        "TotalCharges": ["958.2"],
        "Churn": ["Yes"]
    })

def test_feature_columns_created(sample_df):
    result = engineer_features(sample_df)
    for col in ["service_count", "is_month_to_month", "spend_per_month_ratio", "tenure_bucket"]:
        assert col in result.columns, f"Missing column: {col}"

def test_is_month_to_month_flag(sample_df):
    result = engineer_features(sample_df)
    assert result["is_month_to_month"].iloc[0] == 1

def test_digital_only_flag(sample_df):
    result = engineer_features(sample_df)
    assert result["digital_only"].iloc[0] == 1

def test_service_count_positive(sample_df):
    result = engineer_features(sample_df)
    assert result["service_count"].iloc[0] > 0

def test_get_feature_columns():
    numeric, categorical = get_feature_columns()
    assert isinstance(numeric, list) and len(numeric) > 0
    assert isinstance(categorical, list) and len(categorical) > 0
