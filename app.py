"""Flask web app that serves the academic advisor model."""
from flask import Flask, jsonify, render_template, request

from advisor_engine import ValidationError, calculate_gpa, load_bundle, predict

app = Flask(__name__)


@app.get("/")
def index():
    return render_template("index.html", accuracy=round(load_bundle().get("accuracy", 0) * 100))


@app.get("/health")
def health():
    return jsonify(status="ok")


@app.post("/api/predict")
def api_predict():
    try:
        return jsonify(predict(request.get_json(force=True, silent=True) or {}))
    except ValidationError as e:
        return jsonify(error=str(e)), 400


@app.post("/api/gpa")
def api_gpa():
    body = request.get_json(force=True, silent=True) or {}
    try:
        return jsonify(calculate_gpa(body.get("courses", [])))
    except ValidationError as e:
        return jsonify(error=str(e)), 400


if __name__ == "__main__":
    app.run(debug=True)
