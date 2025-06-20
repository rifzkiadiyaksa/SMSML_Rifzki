import dagshub
import mlflow
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score

# Set up DagsHub for MLflow tracking
dagshub.init(
    repo_owner="rifzki", repo_name="SMSML_Rifzki", mlflow=True
)
mlflow.set_experiment("Cars_Seles")

# Enable MLflow autologging for LightGBM
mlflow.lightgbm.autolog()

# Load data
train_df = pd.read_csv("diabetes_health_indicators_preprocessing/train_processed.csv")
test_df = pd.read_csv("diabetes_health_indicators_preprocessing/test_processed.csv")

X_train = train_df.drop(columns="Diabetes_012")
y_train = train_df["Diabetes_012"]
X_test = test_df.drop(columns="Diabetes_012")
y_test = test_df["Diabetes_012"]

# Train a basic LightGBM model
with mlflow.start_run(run_name="LightGBM_Baseline"):
    model = LGBMClassifier(random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Log test accuracy
    preds = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, preds)
    mlflow.log_metric("test_accuracy", test_accuracy)