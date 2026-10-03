import joblib
import pandas as pd

# Load trained model
model = joblib.load("results/traffic_congestion_model.joblib")

print("======================================")
print("   TRAFFIC CONGESTION PREDICTION")
print("======================================")

road = input("Enter road name: ")
day = int(input("Enter day of week (0=Monday, 6=Sunday): "))
hour = int(input("Enter hour (0-23): "))

input_data = pd.DataFrame({
    "way": [road],
    "day_of_week": [day],
    "hour": [hour]
})

prediction = model.predict(input_data)[0]

print("\nPredicted Congestion Level:", prediction)

levels = {
    0: "Very Low",
    1: "Low",
    2: "Moderate",
    3: "High",
    4: "Very High"
}

print("Traffic Condition:", levels[prediction])