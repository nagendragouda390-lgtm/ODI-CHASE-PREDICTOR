from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load("model_rfc.pkl")

TOTAL_SEGMENTS = 24  # number of bars in the scoreboard-style probability gauge


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    target = float(request.form["target"])
    curr_run = float(request.form["curr_run"])
    total_balls = float(request.form["total_balls"])
    curr_wicket = float(request.form["curr_wicket"])

    # Calculations
    balls_left = 300 - total_balls

    req_runs = target - curr_run

    wick_left = 10 - curr_wicket

    if total_balls > 0:
        curr_rr = curr_run / (total_balls / 6)
    else:
        curr_rr = 0

    if balls_left > 0:
        req_rr = req_runs / (balls_left / 6)
    else:
        req_rr = 0

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
        result = "🏆 The Chase Gets There"
        outcome = "win"
    else:
        result = "❌ The Chase Falls Short"
        outcome = "lose"

    probability = round(probability, 2)
    filled_segments = int(round(probability / 100 * TOTAL_SEGMENTS))

    return render_template(
        "result.html",
        result=result,
        outcome=outcome,
        probability=probability,
        target=int(target),
        curr_run=int(curr_run),
        curr_wicket=int(curr_wicket),
        total_balls=int(total_balls),
        curr_rr=round(curr_rr, 2),
        req_rr=round(req_rr, 2),
        balls_left=int(balls_left),
        wick_left=int(wick_left),
        total_segments=TOTAL_SEGMENTS,
        filled_segments=filled_segments,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
