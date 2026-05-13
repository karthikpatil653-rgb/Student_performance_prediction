import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load Dataset
df = pd.read_excel(
    r'c:\Users\karth\OneDrive\Desktop\Student_data.xlsx'
)

# Show Dataset
print("===== DATASET PREVIEW =====")
print(df.head())

# Input Features
X = df[['student _hours', 'Attendedance', 'PreviousMarks']]

# Output Label
y = df['Result']

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Logistic Regression Model
model = LogisticRegression()

# Train Model
model.fit(X_train, y_train)

# Predict Test Data
predictions = model.predict(X_test)

print("\n===== PREDICTIONS =====")
print(predictions)

# Model Accuracy
accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", accuracy)

# Single Student Prediction
new_student = pd.DataFrame({
    'student _hours': [5],
    'Attendedance': [80],
    'PreviousMarks': [65]
})

# Predict Result
result = model.predict(new_student)

print("\n===== SINGLE STUDENT PREDICTION =====")

if result[0] == 1:
    print("Student Will PASS")
else:
    print("Student Will FAIL")