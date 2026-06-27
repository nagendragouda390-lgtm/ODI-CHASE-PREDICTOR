from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load("model_rfc.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    target = float(request.form["target"])
    total_balls = float(request.form["total_balls"])
    curr_run = float(request.form["curr_run"])
    curr_wicket = float(request.form["curr_wicket"])
    balls_left = float(request.form["balls_left"])
    req_runs = float(request.form["req_runs"])
    wick_left = float(request.form["wick_left"])
    curr_rr = float(request.form["curr_rr"])
    req_rr = float(request.form["req_rr"])


    features = np.array([[
        target,
        total_balls,
        curr_run,
        curr_wicket,
        balls_left,
        req_runs,
        wick_left,
        curr_rr,
        req_rr
    ]])


    prediction = model.predict(features)[0]

    probability = model.predict_proba(features)[0][1] * 100


    if prediction == 1:
        result = "🏆 Chasing Team Will Win"
    else:
        result = "❌ Chasing Team Will Lose"


    return render_template(
        "result.html",
        result=result,
        probability=round(probability,2)
    )


# Render uses this
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
