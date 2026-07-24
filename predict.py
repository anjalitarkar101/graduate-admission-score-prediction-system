# ==========================================================
# predict.py - Prediction functions for Admission Model
# ==========================================================

import numpy as np
import pandas as pd
from tensorflow import keras
import pickle
import os


def load_model_and_artifacts(model_path='models/admission_model.h5'):
    """
    Load the trained model, scaler, and feature names.

    Args:
        model_path: Path to the saved model file

    Returns:
        model: Trained neural network
        scaler: Fitted MinMaxScaler
        feature_names: List of feature names in correct order
    """
    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    # Load the neural network model
    model = keras.models.load_model(model_path)

    # Load the MinMaxScaler used during training
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    # Load feature names in the correct order
    with open('models/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)

    return model, scaler, feature_names


def preprocess_user_input(data_dict, scaler, feature_names):
    """
    Preprocess user input data for prediction.

    Args:
        data_dict: Dictionary with user input values
        scaler: Fitted MinMaxScaler
        feature_names: List of feature names in correct order

    Returns:
        scaled_input: Preprocessed numpy array ready for prediction
    """
    # Create DataFrame from user input (1 row)
    input_df = pd.DataFrame([data_dict])

    # Ensure all required columns exist (add missing ones with 0)
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns to match training data
    input_df = input_df[feature_names]

    # Scale features using the same scaler from training
    scaled_input = scaler.transform(input_df)

    return scaled_input


def predict_admission_score(scaled_input, model):
    """
    Predict admission chance score for a single student.

    Args:
        scaled_input: Preprocessed numpy array
        model: Trained neural network

    Returns:
        admission_chance_score: Float between 0 and 1
    """
    # Make prediction using the neural network
    prediction = model.predict(scaled_input, verbose=0)

    # Extract the prediction value from the array
    admission_chance_score = float(prediction[0][0])

    # Clamp to [0, 1] range (ensure valid probability)
    admission_chance_score = max(0, min(1, admission_chance_score))

    return admission_chance_score