from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import pickle, logging
from pathlib import Path
import pandas as pd

app = FastAPI(
    title='Wave Churn Prediction API',
    description='Predict customer churn probability for mobile money users',
    version='1.0.0'
)
log = logging.getLogger(__name__)

# Load model at startup
_model = None
def get_model():
    global _model
    if _model is None:
        model_files = list(Path('models').glob('churn_model_*.pkl'))
        if not model_files:
            raise FileNotFoundError('No trained model found. Run src/models/train.py first.')
        with open(model_files[0], 'rb') as f:
            _model = pickle.load(f)
        log.info(f'Model loaded: {model_files[0]}')
    return _model

class CustomerFeatures(BaseModel):
    tenure: int = Field(..., ge=0, le=72, description='Months as customer')
    MonthlyCharges: float = Field(..., gt=0)
    TotalCharges: float = Field(..., ge=0)
    Contract: str = Field(..., description='Month-to-month, One year, Two year')
    InternetService: str = Field(..., description='DSL, Fiber optic, No')
    PaymentMethod: str
    gender: str = 'Male'
    SeniorCitizen: int = 0
    Partner: str = 'No'
    Dependents: str = 'No'
    PaperlessBilling: str = 'No'
    PhoneService: str = 'Yes'
    MultipleLines: str = 'No'
    OnlineSecurity: str = 'No'
    OnlineBackup: str = 'No'
    DeviceProtection: str = 'No'
    TechSupport: str = 'No'
    StreamingTV: str = 'No'
    StreamingMovies: str = 'No'

class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: bool
    risk_tier: str

@app.get('/health')
def health():
    return {'status': 'ok', 'model_loaded': _model is not None}

@app.post('/predict', response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    model = get_model()
    from src.features.feature_engineering import engineer_features
    df = pd.DataFrame([customer.model_dump()])
    df = engineer_features(df)
    from src.features.feature_engineering import get_feature_columns
    numeric, categorical = get_feature_columns()
    features = numeric + categorical
    available = [c for c in features if c in df.columns]
    prob = float(model.predict_proba(df[available])[0, 1])
    tier = 'HIGH' if prob >= 0.7 else 'MEDIUM' if prob >= 0.4 else 'LOW'
    return PredictionResponse(
        churn_probability=round(prob, 4),
        churn_prediction=prob >= 0.5,
        risk_tier=tier
    )

@app.get('/')
def root():
    return {'message': 'Wave Churn Prediction API', 'docs': '/docs'}