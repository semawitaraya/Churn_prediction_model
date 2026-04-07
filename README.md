# Churn Prediction API

This project predicts customer churn using a machine learning model and serves it via FastAPI.

## Features
- Logistic Regression model
- API using FastAPI
- CI pipeline with GitHub Actions

## Run locally

```bash
python -m uvicorn app:app --reload



```text
CHURN_PREDICTION_MODEL/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── artifacts/
│   ├── churn_model.pkl
│   ├── training_columns.pkl
│   └── model_features.pkl
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── app.py
├── train.py
├── predict.py
├── requirements.txt
├── README.md
├── .gitignore