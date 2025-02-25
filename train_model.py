from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Load trained model and encoders
model = joblib.load("traffic_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict_congestion():
    try:
        # Get JSON request
        data = request.json

        # Extract features
        lane = data["lane"]
        vehicle_class = data["vehicle_class_code"]
        hour = pd.to_datetime(data["initiated_time"]).hour

        # Encode categorical variables
        lane_encoded = label_encoders["lane"].transform([lane])[0]
        vehicle_class_encoded = label_encoders["vehicle_class_code"].transform([vehicle_class])[0]

        # Prepare input for model
        input_data = pd.DataFrame([[lane_encoded, vehicle_class_encoded, hour]], columns=["lane", "vehicle_class_code", "hour"])

        # Predict congestion level
        prediction = model.predict(input_data)[0]
        print(model.predict(input_data))

        congestion_labels = {0: "Low", 1: "Medium", 2: "High"}
        result = congestion_labels[prediction]

        return jsonify({"congestion_level": result})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
