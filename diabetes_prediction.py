import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load Dataset
df = pd.read_csv("diabetes.csv")

print("Dataset Loaded Successfully!")
print(df.head())

# Data Cleaning
cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

for col in cols:
    df[col] = df[col].replace(0, df[col].median())

# Features and Target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(random_state=42)
}

best_model = None
best_accuracy = 0

print("\nModel Comparison:")

for name, model in models.items():

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, pred)

    print(f"{name}: {accuracy:.4f}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

print("\nBest Model:", type(best_model).__name__)
print("Accuracy:", round(best_accuracy * 100, 2), "%")

# User Input
print("\n========== Diabetes Prediction ==========")

pregnancies = float(input("Pregnancies: "))
glucose = float(input("Glucose: "))
bp = float(input("Blood Pressure: "))
skin = float(input("Skin Thickness: "))
insulin = float(input("Insulin: "))
bmi = float(input("BMI: "))
dpf = float(input("Diabetes Pedigree Function: "))
age = float(input("Age: "))

user_data = np.array([[
    pregnancies,
    glucose,
    bp,
    skin,
    insulin,
    bmi,
    dpf,
    age
]])

user_data = scaler.transform(user_data)

prediction = best_model.predict(user_data)
probability = best_model.predict_proba(user_data)

risk_percent = probability[0][1] * 100

print("\n========== Result ==========")

print("Diabetes Probability:", round(risk_percent, 2), "%")

if risk_percent < 30:
    risk = "LOW"
elif risk_percent < 70:
    risk = "MODERATE"
else:
    risk = "HIGH"

print("Risk Category:", risk)

if prediction[0] == 1:
    print("Prediction: Diabetes Detected")
else:
    print("Prediction: No Diabetes")

print("\n========== Recommendations ==========")

if risk == "LOW":
    print("- Maintain healthy diet")
    print("- Exercise regularly")
    print("- Drink enough water")

elif risk == "MODERATE":
    print("- Reduce sugar intake")
    print("- Walk 30 minutes daily")
    print("- Monitor glucose regularly")

else:
    print("- Consult a doctor")
    print("- Follow diabetic diet")
    print("- Monitor blood sugar frequently")
    print("- Exercise daily")

print("\nThank You!")