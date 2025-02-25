from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS

# Load trained model and encoders
model = joblib.load("traffic_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Load dataset to fetch required details dynamically
df = pd.read_csv("bangalore_toll_data.csv")

app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["POST"])
def predict_congestion():
    try:
        print("coming")
        # Get JSON request
        data = request.json
        merchant_name = data["merchant_name"]
        hour = pd.to_datetime(data["initiated_time"]).hour
        lane_filter = data.get("lane")

        # Fetch unique lanes, direction, and vehicle_class_code for the given merchant
        filtered_data = df[
            (df["merchant_name"] == merchant_name) & (
                pd.to_datetime(df["initiated_time"]).dt.hour == hour
            )
            ][["direction", "lane", "vehicle_class_code"]].drop_duplicates()

        if lane_filter:
            filtered_data = filtered_data[filtered_data["lane"] == lane_filter]

        if filtered_data.empty:
            return jsonify({"error": "No data found for the given merchant name"}), 404

         # Find the most frequent vehicle_class_code per lane
        top_vehicle_per_lane = (
            filtered_data.groupby(["lane", "vehicle_class_code"]) # Step 1: Group by lane and vehicle class
            .size()  # Step 2: Count occurrences in each group
            .reset_index(name="count") # Step 3: Convert to DataFrame and name the count column
            .sort_values(["lane", "count"], ascending=[True, False]) # Step 4: Sort by lane and then by count (descending)
            .drop_duplicates(subset=["lane"], keep="first")  # Step 5: Keep only the most frequent vehicle class per lane
        )

        predictions = []

        for _, row in top_vehicle_per_lane.iterrows():
            lane = row["lane"]
            vehicle_class = row["vehicle_class_code"]

            # Get corresponding direction
            direction = filtered_data[filtered_data["lane"] == lane]["direction"].iloc[0]

            # Encode categorical variables
            merchant_encoded = label_encoders["merchant_name"].transform([merchant_name])[0]
            direction_encoded = label_encoders["direction"].transform([direction])[0]
            lane_encoded = label_encoders["lane"].transform([lane])[0]
            vehicle_class_encoded = label_encoders["vehicle_class_code"].transform([vehicle_class])[0]

            # Prepare input for model
            input_data = pd.DataFrame(
                [[merchant_encoded, direction_encoded, lane_encoded, vehicle_class_encoded, hour]],
                columns=["merchant_name", "direction", "lane", "vehicle_class_code", "hour"]
            )

            # Predict congestion level
            prediction = model.predict(input_data)[0]
            congestion_labels = {0: "Low", 1: "Medium", 2: "High"}
            result = congestion_labels[prediction]

            predictions.append({
                "direction": direction,
                "lane": lane,
                "vehicle_class_code": vehicle_class,
                "congestion_level": result
            })

        return jsonify({"merchant_name": merchant_name, "predictions": predictions})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
