import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

FEATURES = [
    "tenure",
    "contract",
    "monthlycharges",
    "internetservice",
    "onlinesecurity",
    "techsupport",
    "onlinebackup",
    "paymentmethod",
    "deviceprotection",
    "paperlessbilling",
    "totalcharges",
    "seniorcitizen"
]


def load_and_prepare_data(csv_path: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)

    # standardize column names
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    # standardize string values
    string_columns = list(df.dtypes[df.dtypes == "object"].index)
    for col in string_columns:
        df[col] = df[col].astype(str).str.strip().str.lower().str.replace(" ", "_")

    # debug: inspect churn before encoding
    print("Raw churn values after cleaning:")
    print(df["churn"].value_counts(dropna=False))

    # convert totalcharges
    df["totalcharges"] = pd.to_numeric(df["totalcharges"], errors="coerce")

    # encode target safely
    df["churn"] = (df["churn"] == "yes").astype(int)

    print("\nEncoded churn distribution:")
    print(df["churn"].value_counts(dropna=False))

    # drop identifier
    if "customerid" in df.columns:
        df = df.drop(columns=["customerid"])

    # remove rows with missing totalcharges
    df = df.dropna(subset=["totalcharges"])

    X = df[FEATURES].copy()
    y = df["churn"].copy()

    # one-hot encoding
    X = pd.get_dummies(X, drop_first=True)

    return X, y


def train_model(X: pd.DataFrame, y: pd.Series):
    print("\nFull target distribution:")
    print(y.value_counts(dropna=False))

    print("\nFeature matrix shape:")
    print(X.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\ny_train distribution:")
    print(y_train.value_counts(dropna=False))

    print("\ny_test distribution:")
    print(y_test.value_counts(dropna=False))

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    training_columns = X_train.columns.tolist()

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }

    return model, training_columns, metrics


if __name__ == "__main__":
    #csv_path = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    csv_path = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"


    print("Current working directory:", os.getcwd())
    print("CSV exists:", os.path.exists(csv_path))

    X, y = load_and_prepare_data(csv_path)
    model, training_columns, metrics = train_model(X, y)

    # save artifacts
    joblib.dump(model, "artifacts/churn_model.pkl")
    joblib.dump(training_columns, "artifacts/training_columns.pkl")
    joblib.dump(FEATURES, "artifacts/model_features.pkl")

    print("\nModel saved as churn_model.pkl")
    print("Training columns saved as training_columns.pkl")
    print("Raw feature list saved as model_features.pkl")

    print("\nEvaluation metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")