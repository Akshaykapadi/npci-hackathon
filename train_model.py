import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib  # For saving the model

# Load dataset
df = pd.read_csv("bangalore_toll_data.csv")

# Convert timestamp to hour of day
df["hour"] = pd.to_datetime(df["initiated_time"]).dt.hour

# Define congestion levels based on response time
def classify_congestion(time_sec):
    if time_sec < 30:
        return 0  # Low
    elif 30 <= time_sec <= 150:
        return 1  # Medium
    else:
        return 2  # High

df["congestion_level"] = df["inn_rr_time_sec"].apply(classify_congestion)

# Select relevant features
features = ["merchant_name", "direction", "lane", "vehicle_class_code", "hour"]
target = "congestion_level"

# Convert categorical data into numeric (Label Encoding)
label_encoders = {}
for col in ["merchant_name", "direction", "lane", "vehicle_class_code"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Save encoders for later use

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# Train XGBoost Model
model = xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save model and encoders
joblib.dump(model, "traffic_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
print("Model saved as traffic_model.pkl")