from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "house_price_model.pkl")
model = joblib.load(MODEL_PATH)

@app.route("/", methods=["GET", "HEAD"])
def index():
    return render_template("index.html", prediction=None)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [
            float(request.form["OverallQual"]),
            float(request.form["GrLivArea"]),
            float(request.form["TotalBsmtSF"]),
            float(request.form["GarageCars"]),
            float(request.form["FullBath"]),
            float(request.form["YearBuilt"]),
        ]

        final_input = np.array([features])
        prediction = model.predict(final_input)[0]

        return render_template("index.html", prediction=prediction)

    except Exception as e:
        return render_template("index.html", prediction=f"Error: {e}")
