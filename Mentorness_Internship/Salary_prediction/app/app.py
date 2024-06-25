from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load('salary_prediction_model.joblib')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    AGE = int(request.form['AGE'])
    PAST_EXP = int(request.form['PAST_EXP'])
    TENURE = int(request.form['TENURE'])
    TOTAL_LEAVES = int(request.form['TOTAL_LEAVES'])
    PERFORMANCE_PER_YEAR = int(request.form['PERFORMANCE_PER_YEAR'])
    UNIT = request.form['UNIT']
    DESIGNATION = request.form['DESIGNATION']
    SENIORITY = request.form['SENIORITY']
    
    # Create a DataFrame with the new data
    new_data = pd.DataFrame({
        'AGE': [AGE],
        'PAST EXP': [PAST_EXP],
        'TENURE': [TENURE],
        'TOTAL LEAVES': [TOTAL_LEAVES],
        'PERFORMANCE_PER_YEAR': [PERFORMANCE_PER_YEAR],
        'UNIT': [UNIT],
        'DESIGNATION': [DESIGNATION],
        'SENIORITY': [SENIORITY]
    })
    
    # Use the model to make prediction
    prediction = model.predict(new_data)
    
    # Render result.html with the predicted salary
    return render_template('result.html', predicted_salary=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)

