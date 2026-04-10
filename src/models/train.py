import logging
import pickle
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from src.features.feature_engineering import load_raw_data, engineer_features, get_feature_columns

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('training.log')])
log = logging.getLogger(__name__)
MODELS_DIR = Path('models')
MODELS_DIR.mkdir(exist_ok=True)

def build_preprocessor(numeric_cols, categorical_cols):
    return ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
    ])

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob)
    log.info(f'Model: {model_name} | ROC-AUC: {roc_auc:.4f}')
    log.info(classification_report(y_test, y_pred, target_names=['Stay','Churn']))
    return {'model_name': model_name, 'roc_auc': roc_auc}

def train(data_path='data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv'):
    log.info('Loading data...')
    df = load_raw_data(data_path)
    df = engineer_features(df)
    numeric_cols, categorical_cols = get_feature_columns()
    feature_cols = numeric_cols + categorical_cols
    df = df.dropna(subset=feature_cols)
    X, y = df[feature_cols], df['Churn']
    log.info(f'Dataset: {len(X)} rows | Churn rate: {y.mean():.1%}')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    xgb_pipe = Pipeline([('preprocessor', preprocessor),
        ('classifier', XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
            scale_pos_weight=(y_train==0).sum()/(y_train==1).sum(), random_state=42, verbosity=0))])
    xgb_pipe.fit(X_train, y_train)
    xgb_m = evaluate_model(xgb_pipe, X_test, y_test, 'XGBoost')
    lgbm_pipe = Pipeline([('preprocessor', build_preprocessor(numeric_cols, categorical_cols)),
        ('classifier', LGBMClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
            class_weight='balanced', random_state=42, verbose=-1))])
    lgbm_pipe.fit(X_train, y_train)
    lgbm_m = evaluate_model(lgbm_pipe, X_test, y_test, 'LightGBM')
    best_pipe = xgb_pipe if xgb_m['roc_auc'] >= lgbm_m['roc_auc'] else lgbm_pipe
    best_name = 'xgboost' if xgb_m['roc_auc'] >= lgbm_m['roc_auc'] else 'lightgbm'
    model_path = MODELS_DIR / f'churn_model_{best_name}.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(best_pipe, f)
    log.info(f'Best model saved: {model_path}')
    return best_pipe, X_test, y_test, feature_cols

if __name__ == '__main__':
    train()