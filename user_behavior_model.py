import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ----------------------
# STEP 1: Persona Assignment
# ----------------------

persona_data = pd.DataFrame({
    'time_pressure_score': [0.9, 0.2, 0.4, 0.8, 0.1, 0.3],
    'health_consciousness_index': [3, 1, 0, 2, 1, 0],
    'waste_propensity_score': [0.5, 2.5, 1.0, 0.6, 2.8, 1.2],
    'ai_receptiveness_score': [0.9, 0.4, 0.2, 0.85, 0.3, 0.25],
    'persona': ['Time-Strapped', 'Budgeter', 'Traditional', 'Time-Strapped', 'Budgeter', 'Traditional']
})

X_persona = persona_data[['time_pressure_score', 'health_consciousness_index',
                          'waste_propensity_score', 'ai_receptiveness_score']]
y_persona = persona_data['persona']

X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_persona, y_persona, test_size=0.3, random_state=42)

persona_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
persona_classifier.fit(X_train_p, y_train_p)

y_pred_p = persona_classifier.predict(X_test_p)
persona_report = classification_report(y_test_p, y_pred_p)

# ----------------------
# STEP 2: Buying Habits Prediction
# ----------------------

buying_data = pd.DataFrame({
    'persona': ['Time-Strapped', 'Budgeter', 'Traditional', 'Time-Strapped', 'Budgeter', 'Traditional', 'Time-Strapped', 'Budgeter', 'Traditional'],
    'ai_receptiveness_score': [0.9, 0.3, 0.6, 0.85, 0.4, 0.55, 0.8, 0.35, 0.45],
    'impulse_buy_frequency': ['Often', 'Rarely', 'Sometimes', 'Often', 'Rarely', 'Sometimes', 'Often', 'Rarely', 'Sometimes']
})

persona_encode = {'Time-Strapped': 2, 'Budgeter': 1, 'Traditional': 0}
impulse_encode = {'Rarely': 0, 'Sometimes': 1, 'Often': 2}

buying_data['persona_encoded'] = buying_data['persona'].map(persona_encode)
buying_data['impulse_encoded'] = buying_data['impulse_buy_frequency'].map(impulse_encode)

X_buy = buying_data[['persona_encoded', 'ai_receptiveness_score']]
y_buy = buying_data['impulse_encoded']

X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_buy, y_buy, test_size=0.3, random_state=42)

buying_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=42)
buying_model.fit(X_train_b, y_train_b)

y_pred_b = buying_model.predict(X_test_b)
buying_report = classification_report(y_test_b, y_pred_b)

# ----------------------
# STEP 3: Integrated Prediction Pipeline (New User Example)
# ----------------------

def predict_user_behavior(survey_response):
    survey_df = pd.DataFrame([survey_response])
    persona_pred = persona_classifier.predict(survey_df)[0]
    persona_encoded = persona_encode[persona_pred]
    buy_input = np.array([[persona_encoded, survey_response['ai_receptiveness_score']]])
    buying_pred_encoded = buying_model.predict(buy_input)[0]
    impulse_decode = {0: 'Rarely', 1: 'Sometimes', 2: 'Often'}
    buying_pred = impulse_decode[buying_pred_encoded]

    return persona_pred, buying_pred

# Example Prediction
new_user = {
    'time_pressure_score': 0.75,
    'health_consciousness_index': 2,
    'waste_propensity_score': 0.9,
    'ai_receptiveness_score': 0.8
}

predicted_persona, predicted_impulse_buying = predict_user_behavior(new_user)

# ----------------------
# STEP 4: Writing Results to a Text File
# ----------------------

with open('model_results.txt', 'w') as file:
    file.write("AI Kitchen Wellness Assistant Prediction Results\n")
    file.write("="*50 + "\n\n")

    file.write("Persona Assignment Classification Report:\n")
    file.write(persona_report + "\n\n")

    file.write("Buying Habits Classification Report:\n")
    file.write(buying_report + "\n\n")

    file.write("New User Prediction:\n")
    file.write("-" * 25 + "\n")
    file.write(f"Survey Scores:\n")
    for key, value in new_user.items():
        file.write(f"  {key}: {value}\n")
    file.write(f"\nPredicted Persona: {predicted_persona}\n")
    file.write(f"Predicted Impulse Buying Frequency: {predicted_impulse_buying}\n")

print("Predictions successfully written to 'model_results.txt'.")
