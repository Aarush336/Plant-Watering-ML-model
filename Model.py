# Plant Watering ML Model
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