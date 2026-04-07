import joblib
import pandas as pd

MODEL_PATH = "artifacts/churn_model.pkl"
COLUMNS_PATH = "artifacts/training_columns.pkl"




def prepare_input(input_data,training_columns:list):
    #convert input dictionry yo daataframe
    df=pd.DataFrame([input_data])
    
    #standardize string value (same as training)
    for col in df.columns:
        if df[col].dtype == "object":
            df[col]=df[col].str.lower().str.replace(" ","_")
    # one -hot encoding
    df=pd.get_dummies(df,drop_first=True)
    
    # Align Columns with training 
    df=df.reindex(columns=training_columns,fill_value=0)
    
    return df

def predict_churn(input_data:dict):
    #load model+ columns
    model=joblib.load(MODEL_PATH)
    training_columns=joblib.load(COLUMNS_PATH)
    
    # Prepare input
    X=prepare_input(input_data,training_columns)
    
    #Predict
    churn_probability=model.predict_proba(X)[0,1]
    churn_prediction=int(churn_probability>=0.5)
    
    return {
        "churn_probability": float(churn_probability),
        "churn_prediction":churn_prediction
    }
    
if __name__ == "__main__":
        
    #examle Customtr (test input)
      sample_customer = {
        "tenure": 5,
        "contract": "month-to-month",
        "monthlycharges": 85.5,
        "internetservice": "fiber_optic",
        "onlinesecurity": "no",
        "techsupport": "no",
        "onlinebackup": "yes",
        "paymentmethod": "electronic_check",
        "deviceprotection": "no",
        "paperlessbilling": "yes",
        "totalcharges": 450.25,
        "seniorcitizen": 0
        }
result=predict_churn(sample_customer)
print(result)