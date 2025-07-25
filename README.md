# Predictive User Behavior Modeling

This project demonstrates a simple, production-style pipeline for predicting user personas and buying habits using machine learning. All data is randomly generated to simulate real-world survey responses and behaviors.

## Project Overview

- **Goal:** Predict a user's persona (e.g., "Time-Strapped", "Budgeter", "Traditional") and their likelihood of impulse buying ("Rarely", "Sometimes", "Often") based on survey-like input data.
- **Approach:**
  - Generate large, realistic dummy datasets for both persona assignment and buying habits.
  - Train and tune machine learning models (Random Forest and Logistic Regression) on this data.
  - Save the trained models and preprocessing steps for future use.
  - Provide scripts to make predictions for new users.

## How the Data is Generated

- **Persona Data:**
  - Each user is assigned random scores for time pressure, health consciousness, waste propensity, and AI receptiveness.
  - Simple rules (with some randomness) assign a persona label based on these scores.
- **Buying Habits Data:**
  - Each entry randomly selects a persona and AI receptiveness score.
  - Rules and randomness determine the impulse buying frequency label.

## How the Models Work

- **Persona Model:**
  - Uses a Random Forest Classifier to predict the persona from the four survey scores.
  - The model is trained and tuned using cross-validation and hyperparameter search.
- **Buying Model:**
  - Uses a Logistic Regression model to predict impulse buying frequency from the persona and AI receptiveness.
  - Also trained and tuned with cross-validation.
- **Preprocessing:**
  - All features are scaled using `StandardScaler` to ensure consistent input for the models.
- **Model Saving:**
  - Trained models and scalers are saved as `.joblib` files for easy reuse.

## Files and Scripts

- `user_behavior_model.py`: Main script for generating data, training models, and saving results/models.
- `predict_new_user.py`: Example script to load the saved models and make predictions for a new (mock) user.
- `*.joblib`: Saved models and scalers.
- `model_results_*.txt`: Output files with training reports and example predictions.

## How to Run

1. **Train the Models and Generate Data:**
   ```
   python user_behavior_model.py
   ```
   - This will generate data, train the models, save them, and write a results file.

2. **Make a Prediction for a New User:**
   ```
   python predict_new_user.py
   ```
   - Edit the `new_user` dictionary in this script to test different scenarios.
   - The script will print the predicted persona and impulse buying frequency.

## Next Steps / Suggestions

- Replace the random data generation with real survey or behavioral data for more meaningful predictions.
- Integrate the models into a web or desktop application for interactive predictions.
- Experiment with more features, different algorithms, or advanced model tuning.
- Add automated tests or a REST API for production deployment.

## Contact

For questions or suggestions, please reach out to the project maintainer.
