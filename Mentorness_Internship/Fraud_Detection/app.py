from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np

# Load the trained model
model_filename = 'fastag_fraud_detection_model.pkl'
loaded_model = joblib.load(model_filename)

# Initialize Flask application
app = Flask(__name__)

# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from POST request
    new_data = request.form.to_dict()
    
    # Convert the received data into a DataFrame
    new_df = pd.DataFrame([new_data])
    
    # Preprocess the new data
    new_df['Day'] = new_df['Day'].astype(int)
    new_df['Month'] = new_df['Month'].astype(int)
    new_df['Hour'] = new_df['Hour'].astype(int)
    
    # Define categorical and numerical features
    categorical_features = ['Vehicle_Type', 'FastagID', 'TollBoothID', 'Lane_Type', 'Vehicle_Dimensions']
    numerical_features = ['Transaction_Amount', 'Amount_paid', 'Vehicle_Speed', 'Latitude', 'Longitude', 'Day', 'Month', 'Hour']
    
    # Ensure the new data columns match the expected columns used during training
    new_df = new_df[numerical_features + categorical_features]
    
    # Handle missing values for categorical features
    new_df['FastagID'].fillna('Unknown', inplace=True)
    
    # Encode categorical features
    new_df_categorical = pd.get_dummies(new_df[categorical_features], drop_first=True)
    
    # Combine numerical and encoded categorical data
    new_df_processed = pd.concat([new_df[numerical_features], new_df_categorical], axis=1)
    
    # Ensure the columns are in the same order as expected during training
    expected_columns = ['Transaction_Amount', 'Amount_paid', 'Vehicle_Speed', 'Latitude', 'Longitude', 'Day', 'Month', 'Hour',
                        'Vehicle_Type', 'FastagID', 'TollBoothID', 'Lane_Type', 'Vehicle_Dimensions']
    
    # Reorder columns in case they are shuffled during preprocessing
    new_df_processed = new_df_processed.reindex(columns=expected_columns, fill_value=0)
    
    # Make predictions using the loaded model
    predictions = loaded_model.predict(new_df_processed)
    
    # Return the prediction as response
    if predictions[0] == 1:
        result = 'Fraudulent Transaction'
    else:
        result = 'Non-Fraudulent Transaction'
    
    return render_template('index.html', prediction_text=result)

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
