# Traffic Congestion Prediction API

This project provides a machine learning model to predict congestion levels at toll booths based on merchant name, time of day, lane.

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/Akshaykapadi/npci-hackathon.git
cd npci-hackathon
```

### 2. Create a Virtual Environment
```bash
python -m venv toll_env
```
Activate the virtual environment:
- **Windows:**
  ```bash
  toll_env\Scripts\activate
  ```
- **Mac/Linux:**
  ```bash
  source toll_env/bin/activate
  ```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
OR 
```
pip install pandas numpy xgboost scikit-learn flask
```

### 4. Train the Model
Ensure that you have the dataset (`bangalore_toll_data.csv`) in the project directory.
```bash
python train_model.py
```
This will train the model and save it as `traffic_model.pkl` along with the label encoders.

### 5. Start the API Server
```bash
python app.py
```

### 6. Execute API Requests
#### Predict Congestion for a Merchant & Time
```bash
curl -X POST "http://127.0.0.1:5000/predict" \
     -H "Content-Type: application/json" \
     -d '{
         "initiated_time": "2025-02-25 17:30:00",
         "merchant_name": "ELECTRONIC CITY Phase 1"
     }'
```
Response Example:
```json
{
  "merchant_name": "ELECTRONIC  CITY Phase 1",
  "predictions": [
    {
      "congestion_level": "Low",
      "direction": "E",
      "lane": "00",
      "vehicle_class_code": "VC11"
    },
    {
      "congestion_level": "High",
      "direction": "N",
      "lane": "LANE10",
      "vehicle_class_code": "VC10"
    },
    {
      "congestion_level": "High",
      "direction": "N",
      "lane": "LANE11",
      "vehicle_class_code": "VC10"
    },
    {
      "congestion_level": "High",
      "direction": "S",
      "lane": "LANE15",
      "vehicle_class_code": "VC10"
    },
    {
      "congestion_level": "High",
      "direction": "S",
      "lane": "LANE16",
      "vehicle_class_code": "VC10"
    },
    {
      "congestion_level": "High",
      "direction": "S",
      "lane": "LANE2",
      "vehicle_class_code": "VC10"
    },
    {
      "congestion_level": "High",
      "direction": "N",
      "lane": "LANE21",
      "vehicle_class_code": "VC10"
    },
    {
      "congestion_level": "High",
      "direction": "N",
      "lane": "LANE22",
      "vehicle_class_code": "VC10"
    },
    {
      "congestion_level": "High",
      "direction": "N",
      "lane": "LANE23",
      "vehicle_class_code": "VC10"
    },
    {
      "congestion_level": "High",
      "direction": "N",
      "lane": "LANE24",
      "vehicle_class_code": "VC10"
    },
    {
      "congestion_level": "High",
      "direction": "N",
      "lane": "LANE25",
      "vehicle_class_code": "VC10"
    },
    {
      "congestion_level": "High",
      "direction": "S",
      "lane": "LANE29",
      "vehicle_class_code": "VC10"
    },
    {
      "congestion_level": "High",
      "direction": "S",
      "lane": "LANE3",
      "vehicle_class_code": "VC10"
    },
    {
      "congestion_level": "High",
      "direction": "S",
      "lane": "LANE30",
      "vehicle_class_code": "VC10"
    },
    {
      "congestion_level": "High",
      "direction": "S",
      "lane": "LANE31",
      "vehicle_class_code": "VC10"
    },
    {
      "congestion_level": "High",
      "direction": "S",
      "lane": "LANE32",
      "vehicle_class_code": "VC10"
    },
    {
      "congestion_level": "High",
      "direction": "S",
      "lane": "LANE4",
      "vehicle_class_code": "VC10"
    },
    {
      "congestion_level": "High",
      "direction": "N",
      "lane": "LANE9",
      "vehicle_class_code": "VC10"
    }
  ]
}
```

### 7. Deactivate Virtual Environment (Optional)
```bash
deactivate
```

## Notes
- Ensure you have Python installed (`>=3.8`).

---
Happy coding! 🚀

