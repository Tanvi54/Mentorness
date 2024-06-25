from flask import Flask

app = Flask(__name__)

# Load the model
from joblib import load
model = load('salary_prediction_model.joblib')

from . import routes
