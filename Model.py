# Plant Watering ML Model.
# Predicts plant health based on temperature, humidity, and soil moisture
# Algorithm: Desicion Tree Classifier | Current Accuracy: 85 %
import pandas as pd

df = pd.read_csv("plant_moniter_health_data.csv")
print(df.head())
print(df.shape)
print(df["Health_Status"].value_counts())

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

X = df[["Temperature_C", "Humidity_%", "Soil_Moisture_%", "Soil_pH"]]
Y = df["Health_Status"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(random_state=42, max_depth=1)
model.fit(X_train, Y_train)
prediction = model.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y_test, prediction)
print(accuracy)

# Next Task: Build Predict function for sensor input
# Day: 14: Architecture Decided: ESP 32/Arduino -> Cloud Database -> ML Model -> Pump 
# Day: 15: Studied and analysed some other models using the similar mechanism

def predict_watering (temperature, humidity, soil_moisture, soil_pH):
    new_plant = pd.DataFrame([[temperature, humidity, soil_moisture, soil_pH]], columns = ["Temperature_C", "Humidity_%", "Soil_Moisture_%", "Soil_pH"])
    result = model.predict(new_plant)
    if result[0] == 1:
        return "Plant is healthy. No water needed."
    else:
        return "Plant needs attention. Consider watering"
print(predict_watering(30, 25, 15, 5.5))
# Example usage:
# print(predict_watering(42, 35, 28, 6.5))  # Output: Plant is healthy. No water needed.
# print(predict_watering(30, 25, 15, 5.5))  # Output: Plant needs attention. Consider watering.
# Day 19: predict_watering() function complete, Firebase integration next
