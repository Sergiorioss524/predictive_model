import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# ----------------------
# DATA GENERATION
# ----------------------
def generate_persona_data(n=1000, random_state=42):
    """
    Generate dummy data for persona assignment.
    Each row simulates a user with various behavioral scores and assigns a persona based on simple rules.
    Args:
        n (int): Number of samples to generate.
        random_state (int): Seed for reproducibility.
    Returns:
        pd.DataFrame: Generated persona data.
    """
    np.random.seed(random_state)
    data = []
    for _ in range(n):
        # Simulate user survey scores
        time_pressure = np.clip(np.random.normal(0.5, 0.3), 0, 1)
        health_index = np.random.choice([0, 1, 2, 3], p=[0.2, 0.3, 0.3, 0.2])
        waste_score = np.clip(np.random.normal(1.5, 1.0), 0, 3)
        ai_recept = np.clip(np.random.beta(2, 2), 0, 1)
        # Assign persona based on rules
        if time_pressure > 0.7:
            persona = 'Time-Strapped'
        elif health_index > 1 and ai_recept > 0.6:
            persona = 'Time-Strapped'
        elif waste_score > 2:
            persona = 'Budgeter'
        elif health_index == 0:
            persona = 'Traditional'
        else:
            persona = np.random.choice(['Budgeter', 'Traditional'])
        data.append([time_pressure, health_index, waste_score, ai_recept, persona])
    return pd.DataFrame(data, columns=['time_pressure_score', 'health_consciousness_index', 'waste_propensity_score', 'ai_receptiveness_score', 'persona'])


def generate_buying_data(persona_data, n=2000, random_state=42):
    """
    Generate dummy data for buying habits prediction.
    Each row simulates a user with a persona and AI receptiveness, and assigns an impulse buying frequency.
    Args:
        persona_data (pd.DataFrame): DataFrame with persona categories (not directly used, but for future extensibility).
        n (int): Number of samples to generate.
        random_state (int): Seed for reproducibility.
    Returns:
        pd.DataFrame: Generated buying habits data.
    """
    np.random.seed(random_state)
    persona_encode = {'Time-Strapped': 2, 'Budgeter': 1, 'Traditional': 0}
    impulse_choices = ['Rarely', 'Sometimes', 'Often']
    data = []
    for _ in range(n):
        # Randomly assign persona and AI receptiveness
        persona = np.random.choice(list(persona_encode.keys()))
        ai_recept = np.clip(np.random.beta(2, 2), 0, 1)
        # Assign impulse buying frequency based on rules
        if persona == 'Time-Strapped' and ai_recept > 0.7:
            impulse = 'Often'
        elif persona == 'Budgeter' and ai_recept < 0.4:
            impulse = 'Rarely'
        elif persona == 'Traditional':
            impulse = np.random.choice(['Rarely', 'Sometimes'], p=[0.7, 0.3])
        else:
            impulse = np.random.choice(impulse_choices)
        data.append([persona, ai_recept, impulse])
    return pd.DataFrame(data, columns=['persona', 'ai_receptiveness_score', 'impulse_buy_frequency'])

# ----------------------
# MODEL TRAINING & EVALUATION
# ----------------------
def train_persona_classifier(df):
    """
    Train and tune a RandomForestClassifier for persona assignment.
    Uses feature scaling and grid search cross-validation for hyperparameter tuning.
    Args:
        df (pd.DataFrame): Persona data.
    Returns:
        best_model: Trained RandomForestClassifier.
        scaler: Fitted StandardScaler.
        report (str): Classification report on training data.
        best_params (dict): Best hyperparameters found.
    """
    # Select features and target
    X = df[['time_pressure_score', 'health_consciousness_index', 'waste_propensity_score', 'ai_receptiveness_score']]
    y = df['persona']
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Define model and hyperparameter grid
    rf = RandomForestClassifier(random_state=42)
    param_grid = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Grid search with cross-validation
    grid = GridSearchCV(rf, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    grid.fit(X_scaled, y)
    best_model = grid.best_estimator_
    # Evaluate on training data (for demonstration)
    y_pred = best_model.predict(X_scaled)
    report = classification_report(y, y_pred)
    return best_model, scaler, report, grid.best_params_


def train_buying_model(df):
    """
    Train and tune a LogisticRegression model for impulse buying prediction.
    Uses feature scaling and grid search cross-validation for hyperparameter tuning.
    Args:
        df (pd.DataFrame): Buying habits data.
    Returns:
        best_model: Trained LogisticRegression model.
        scaler: Fitted StandardScaler.
        report (str): Classification report on training data.
        best_params (dict): Best hyperparameters found.
    """
    # Encode categorical variables
    persona_encode = {'Time-Strapped': 2, 'Budgeter': 1, 'Traditional': 0}
    impulse_encode = {'Rarely': 0, 'Sometimes': 1, 'Often': 2}
    df['persona_encoded'] = df['persona'].map(persona_encode)
    df['impulse_encoded'] = df['impulse_buy_frequency'].map(impulse_encode)
    # Select features and target
    X = df[['persona_encoded', 'ai_receptiveness_score']]
    y = df['impulse_encoded']
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Define model and hyperparameter grid
    lr = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=42, max_iter=500)
    param_grid = {'C': [0.1, 1, 10]}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Grid search with cross-validation
    grid = GridSearchCV(lr, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    grid.fit(X_scaled, y)
    best_model = grid.best_estimator_
    # Evaluate on training data (for demonstration)
    y_pred = best_model.predict(X_scaled)
    report = classification_report(y, y_pred)
    return best_model, scaler, report, grid.best_params_

# ----------------------
# INTEGRATED PREDICTION PIPELINE
# ----------------------
def predict_user_behavior(survey_response, persona_model, persona_scaler, buying_model, buying_scaler):
    """
    Predict persona and impulse buying for a new user.
    Args:
        survey_response (dict): User's survey scores.
        persona_model: Trained persona classifier.
        persona_scaler: Fitted scaler for persona features.
        buying_model: Trained buying habits model.
        buying_scaler: Fitted scaler for buying features.
    Returns:
        tuple: (predicted_persona, predicted_impulse_buying)
    Raises:
        ValueError: If required survey fields are missing.
    """
    persona_encode = {'Time-Strapped': 2, 'Budgeter': 1, 'Traditional': 0}
    impulse_decode = {0: 'Rarely', 1: 'Sometimes', 2: 'Often'}
    required_keys = ['time_pressure_score', 'health_consciousness_index', 'waste_propensity_score', 'ai_receptiveness_score']
    # Check for missing fields
    for key in required_keys:
        if key not in survey_response:
            raise ValueError(f"Missing required survey field: {key}")
    # Prepare input for persona prediction
    survey_df = pd.DataFrame([survey_response])
    X_persona = persona_scaler.transform(survey_df)
    persona_pred = persona_model.predict(X_persona)[0]
    persona_encoded = persona_encode[persona_pred]
    # Prepare input for buying prediction
    buy_input = np.array([[persona_encoded, survey_response['ai_receptiveness_score']]])
    X_buy = buying_scaler.transform(buy_input)
    buying_pred_encoded = buying_model.predict(X_buy)[0]
    buying_pred = impulse_decode[buying_pred_encoded]
    return persona_pred, buying_pred

# ----------------------
# MAIN EXECUTION
# ----------------------
def main():
    """
    Main function to generate data, train models, save models, make a sample prediction, and write results to a file.
    """
    # Step 1: Generate large dummy datasets
    persona_data = generate_persona_data(n=1000)
    buying_data = generate_buying_data(persona_data, n=2000)

    # Step 2: Train and tune models
    persona_model, persona_scaler, persona_report, persona_params = train_persona_classifier(persona_data)
    buying_model, buying_scaler, buying_report, buying_params = train_buying_model(buying_data)

    # Step 3: Save trained models and scalers to disk for future use
    joblib.dump(persona_model, 'persona_classifier.joblib')
    joblib.dump(persona_scaler, 'persona_scaler.joblib')
    joblib.dump(buying_model, 'buying_model.joblib')
    joblib.dump(buying_scaler, 'buying_scaler.joblib')

    # Step 4: Example prediction for a new user
    new_user = {
        'time_pressure_score': 0.75,  # High time pressure
        'health_consciousness_index': 2,  # Moderately health conscious
        'waste_propensity_score': 0.9,  # Low waste propensity
        'ai_receptiveness_score': 0.8   # High AI receptiveness
    }
    predicted_persona, predicted_impulse_buying = predict_user_behavior(
        new_user, persona_model, persona_scaler, buying_model, buying_scaler)

    # Step 5: Output results to a timestamped file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'model_results_{timestamp}.txt'
    with open(output_file, 'w') as file:
        file.write("AI Kitchen Wellness Assistant Prediction Results\n")
        file.write("="*50 + "\n\n")
        file.write("Persona Assignment Classification Report:\n")
        file.write(persona_report + "\n\n")
        file.write(f"Best Persona Model Params: {persona_params}\n\n")
        file.write("Buying Habits Classification Report:\n")
        file.write(buying_report + "\n\n")
        file.write(f"Best Buying Model Params: {buying_params}\n\n")
        file.write("New User Prediction:\n")
        file.write("-" * 25 + "\n")
        file.write(f"Survey Scores:\n")
        for key, value in new_user.items():
            file.write(f"  {key}: {value}\n")
        file.write(f"\nPredicted Persona: {predicted_persona}\n")
        file.write(f"Predicted Impulse Buying Frequency: {predicted_impulse_buying}\n")
    print(f"Predictions successfully written to '{output_file}'.")

# Entry point for script execution
if __name__ == "__main__":
    main()
