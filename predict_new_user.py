import joblib
import numpy as np
import pandas as pd

# Load the saved models and scalers
persona_model = joblib.load('persona_classifier.joblib')
persona_scaler = joblib.load('persona_scaler.joblib')
buying_model = joblib.load('buying_model.joblib')
buying_scaler = joblib.load('buying_scaler.joblib')

# Define the encoding/decoding as in your main script
persona_encode = {'Time-Strapped': 2, 'Budgeter': 1, 'Traditional': 0}
impulse_decode = {0: 'Rarely', 1: 'Sometimes', 2: 'Often'}

# Mock data for a new user (change these values to test different scenarios)
new_user = {
    'time_pressure_score': 0.65,  # Example: moderate time pressure
    'health_consciousness_index': 3,  # Example: very health conscious
    'waste_propensity_score': 0.5,  # Example: low waste propensity
    'ai_receptiveness_score': 0.9   # Example: very receptive to AI
}

# Prepare data for persona prediction
survey_df = pd.DataFrame([new_user])
X_persona = persona_scaler.transform(survey_df)
predicted_persona = persona_model.predict(X_persona)[0]

# Prepare data for buying prediction
persona_encoded = persona_encode[predicted_persona]
buy_input = np.array([[persona_encoded, new_user['ai_receptiveness_score']]])
X_buy = buying_scaler.transform(buy_input)
buying_pred_encoded = buying_model.predict(X_buy)[0]
predicted_impulse = impulse_decode[buying_pred_encoded]

print("Mock User Data:")
for k, v in new_user.items():
    print(f"  {k}: {v}")
print(f"\nPredicted Persona: {predicted_persona}")
print(f"Predicted Impulse Buying Frequency: {predicted_impulse}") 