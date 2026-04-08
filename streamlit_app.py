
import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("artifacts/churn_model.pkl")
training_columns = joblib.load("artifacts/training_columns.pkl")


def prepare_input(input_data):
    df = pd.DataFrame([input_data])

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].str.lower().str.replace(" ", "_")

    df = pd.get_dummies(df, drop_first=True)
    df = df.reindex(columns=training_columns, fill_value=0)

    return df


st.title("📊 Churn Prediction App")

st.write("Enter customer details:")

# Inputs
tenure = st.slider("Tenure", 0, 72, 12)
contract = st.selectbox("Contract", ["month-to-month", "one_year", "two_year"])
monthlycharges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
internetservice = st.selectbox("Internet Service", ["dsl", "fiber_optic", "no"])
onlinesecurity = st.selectbox("Online Security", ["yes", "no"])
techsupport = st.selectbox("Tech Support", ["yes", "no"])
onlinebackup = st.selectbox("Online Backup", ["yes", "no"])
paymentmethod = st.selectbox("Payment Method", ["electronic_check", "mailed_check", "bank_transfer", "credit_card"])
deviceprotection = st.selectbox("Device Protection", ["yes", "no"])
paperlessbilling = st.selectbox("Paperless Billing", ["yes", "no"])
totalcharges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)
seniorcitizen = st.selectbox("Senior Citizen", [0, 1])

if st.button("Predict"):
    input_data = {
        "tenure": tenure,
        "contract": contract,
        "monthlycharges": monthlycharges,
        "internetservice": internetservice,
        "onlinesecurity": onlinesecurity,
        "techsupport": techsupport,
        "onlinebackup": onlinebackup,
        "paymentmethod": paymentmethod,
        "deviceprotection": deviceprotection,
        "paperlessbilling": paperlessbilling,
        "totalcharges": totalcharges,
        "seniorcitizen": seniorcitizen
    }

    X = prepare_input(input_data)

    prob = model.predict_proba(X)[0, 1]
    pred = int(prob >= 0.5)

    st.subheader("Result:")
    st.write(f"Churn Probability: {prob:.2f}")
    st.write(f"Prediction: {'Churn' if pred == 1 else 'No Churn'}")