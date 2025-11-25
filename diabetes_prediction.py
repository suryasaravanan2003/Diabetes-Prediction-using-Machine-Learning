# Day 5 - Diabetes Prediction ML Project

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load Dataset
data = pd.read_csv("diabetes.csv")
print("Dataset Loaded Successfully!")
print(data.head())

# Check missing values
print("\nMissing Values:\n", data.isnull().sum())

# Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Features & Target
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Standardize Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model Training
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# Classification Report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# User Input Prediction
print("\n=== Diabetes Prediction ===")
vals = []

features = ["Pregnancies", "Glucose", "Blood Pressure", "Skin Thickness", 
            "Insulin", "BMI", "Diabetes Pedigree Function", "Age"]

for f in features:
    vals.append(float(input(f"Enter {f}: ")))

user_data = scaler.transform([vals])
prediction = model.predict(user_data)[0]

if prediction == 1:
    print("\nResult: HIGH RISK of Diabetes ❗")
else:
    print("\nResult: LOW RISK of Diabetes ✅")
