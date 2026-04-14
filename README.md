# Mobile Money Churn Prediction & Incentive Optimizer

> **Wave DS Portfolio · Project 1 of 5**
> Target Role: Data Scientist, Wave Mobile Money (Kenya)

## Problem Statement
Mobile money providers lose revenue when customers stop transacting. This project builds an end-to-end ML system that:
1. **Predicts** which customers are at risk of churning (XGBoost + LightGBM)
2. **Explains** why using SHAP values (model transparency)
3. **Optimises** a fixed retention budget — which at-risk customers to target with incentives

## Project Structure
```
wave-churn-prediction/
├── src/
│   ├── features/       # Feature engineering pipeline
│   ├── models/         # Training, evaluation, SHAP
│   ├── api/            # FastAPI prediction endpoint
│   └── utils/          # Shared helpers (logging, config)
├── notebooks/          # EDA and experimentation
├── tests/              # pytest unit tests
├── data/
│   ├── raw/            # Raw data (gitignored)
│   └── processed/      # Feature-engineered data (gitignored)
├── configs/            # Model hyperparameters (YAML)
├── requirements.txt
└── README.md
```

## Dataset
- **Source:** [Kaggle Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- ~7,000 customers, 20 features
- Binary target:  (Yes/No)

## Quickstart
```bash
# 1. Clone and set up environment
git clone https://github.com/YOUR_USERNAME/wave-churn-prediction
cd wave-churn-prediction
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scriptsctivate
pip install -r requirements.txt

# 2. Download dataset (requires Kaggle API key)
kaggle datasets download -d blastchar/telco-customer-churn -p data/raw --unzip

# 3. Run EDA notebook
jupyter notebook notebooks/01_eda.ipynb

# 4. Train model
python -m src.models.train

# 5. Start API
uvicorn src.api.main:app --reload
# Visit http://localhost:8000/docs
```

## Results
| Model | ROC-AUC | F1 (Churn) | Precision | Recall |
|-------|---------|------------|-----------|--------|
| XGBoost | TBD | TBD | TBD | TBD |
| LightGBM | TBD | TBD | TBD | TBD |

## Key Findings
*To be completed after model training*

## Incentive Optimizer
Given a fixed budget B and N at-risk customers, we solve:

**Maximize:** expected retention revenue  
**Subject to:** total incentive cost ≤ B

Using  with churn probability as the objective weight.

## API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
|  | POST | Single customer churn score |
|  | POST | Batch CSV prediction |
|  | GET | Service health check |

## Author
Andrian Mwai.
