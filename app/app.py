from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# load model
with open("../model/dropout_model.pkl", 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    df = pd.read_csv(file)

    expected_columns = ['attendance_rate', 'test_score', 'engagement_score']
    missing = [col for col in expected_columns if col not in df.columns]

    if missing:
        return f"Missing columns in CSV: {', '.join(missing)}", 400

    # Make prediction
    X = df[expected_columns]
    preds = model.predict(X)
    df['dropout_prediction'] = preds

    # Save results if needed
    df.to_csv('results.csv', index=False)

    return render_template('results.html', tables=[df.to_html(classes='table table-bordered', index=False)], title='Prediction Results')


    
@app.route("/results")
def results():
    return render_template("results.html")
    
if __name__ == '__main__':
    app.run(debug=True)