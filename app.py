from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Churn Prediction API")


MODEL_PATH = "artifacts/churn_model.pkl"
COLUMNS_PATH = "artifacts/training_columns.pkl"


class CustomerData(BaseModel):
    tenure: int
    contract: str
    monthlycharges: float
    internetservice: str
    onlinesecurity: str
    techsupport: str
    onlinebackup: str
    paymentmethod: str
    deviceprotection: str
    paperlessbilling: str
    totalcharges: float
    seniorcitizen: int


def prepare_input(input_data: dict, training_columns: list):
    df = pd.DataFrame([input_data])

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].str.lower().str.replace(" ", "_")

    df = pd.get_dummies(df, drop_first=True)
    df = df.reindex(columns=training_columns, fill_value=0)

    return df


@app.get("/")
def home():
    return {"message": "Churn Prediction API is running"}


@app.post("/predict")
def predict(customer: CustomerData):
    input_data = customer.dict()
    X = prepare_input(input_data, training_columns)

    churn_probability = model.predict_proba(X)[0, 1]
    churn_prediction = int(churn_probability >= 0.5)

    return {
        "churn_probability": float(churn_probability),
        "churn_prediction": churn_prediction
    }