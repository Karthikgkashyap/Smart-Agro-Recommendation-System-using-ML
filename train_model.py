"""
Advanced Crop Recommendation System - Model Training
This script trains a Machine Learning model (Random Forest) for crop recommendation.
It includes data preprocessing, scaling, model training, evaluation, and saving the artifacts.

For your viva, remember:
1. Data: We use N, P, K (soil nutrients), temperature, humidity, pH, and rainfall as features.
2. Preprocessing: We use StandardScaler to normalize the data. Normalization ensures that features 
   with larger ranges (like rainfall) do not dominate those with smaller ranges (like pH).
3. Model: Random Forest is an ensemble learning method. It builds multiple decision trees 
   and merges them together to get a more accurate and stable prediction.
4. Metric: We use accuracy score and classification report to evaluate how well it predicts the crop.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# 1. Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "Crop_recommendation.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

def train_and_save_model():
    print("Starting Advanced Crop Recommendation Model Training...")

    # 2. Load the Dataset
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {DATA_PATH}. Please check the data folder.")
        # Try local path if running from root
        DATA_PATH_ALT = os.path.join("data", "Crop_recommendation.csv")
        try:
            df = pd.read_csv(DATA_PATH_ALT)
            print("Loaded dataset from alternate path.")
        except FileNotFoundError:
            return

    print(f"Dataset Loaded Successfully. Shape: {df.shape}")
    
    # 3. Separate Features (X) and Target Label (y)
    # Features: N, P, K, temperature, humidity, ph, rainfall
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    # Target: label (the crop name)
    y = df['label']

    # 4. Split data into Training and Testing sets
    # 80% data for training the model, 20% for testing its performance
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data Split: {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples.")

    # 5. Feature Scaling
    # We scale the features so they have a mean of 0 and variance of 1.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Feature Scaling Complete (StandardScaler).")

    # 6. Initialize and Train the Model
    # Random Forest with 100 trees
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    print("Training Random Forest Classifier...")
    model.fit(X_train_scaled, y_train)

    # 7. Evaluate the Model
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Training Complete. Accuracy on Test Set: {acc * 100:.2f}%")
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))

    # 8. Save the Model and Scaler
    # We save these objects to disk so the Flask backend can load them for predictions without retraining.
    joblib.dump(model, os.path.join(MODEL_DIR, "rf_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    print(f"Model saved to {MODEL_DIR}/rf_model.pkl")
    print(f"Scaler saved to {MODEL_DIR}/scaler.pkl")
    print("All done!")

if __name__ == "__main__":
    train_and_save_model()
