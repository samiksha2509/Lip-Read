from __future__ import annotations

import base64
import json
import logging
import os
import tempfile
import threading
import time
import urllib.request

import cv2
from flask import Flask, jsonify, render_template, request, send_from_directory

from lipnet.decode import ctc_greedy_decode, grid_correct
from lipnet.preprocess import crops_to_strip, standardise, video_to_mouth_crops

BASE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.environ.get("WEIGHTS_PATH", os.path.join(BASE, "models", "lipnet.weights.h5"))
WEIGHTS_URL = os.environ.get("WEIGHTS_URL", "")
SAMPLES_DIR = os.path.join(BASE, "static", "samples")
ALLOWED = {".mp4", ".webm", ".mov", ".avi", ".mpg", ".mpeg", ".mkv", ".m4v", ".3gp"}

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("lipreading")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MAX_UPLOAD_MB", "25")) * 1024 * 1024

_model = None
_model_error = None
_lock = threading.Lock()


def get_model():
    """Load the Keras model once (lazily, thread-safe). Returns None if unavailable."""
    global _model, _model_error
    if _model is not None or _model_error is not None:
        return _model
    with _lock:
        if _model is not None or _model_error is not None:
            return _model
        try:
            if not os.path.exists(WEIGHTS_PATH) and WEIGHTS_URL:
                os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
                log.info("downloading weights from %s", WEIGHTS_URL)
                urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS_PATH)
            if not os.path.exists(WEIGHTS_PATH):
                raise FileNotFoundError(
                    "Trained weights not found. Train the model with the Colab notebook "
                    "and put lipnet.weights.h5 in the models/ folder.")
            from lipnet.model import load_trained_model   # imports TensorFlow
            _model = load_trained_model(WEIGHTS_PATH)
            log.info("model loaded from %s", WEIGHTS_PATH)
        except Exception as e:  # keep the site up and explain what is missing
            _model_error = str(e)
            log.error("model unavailable: %s", e)
    return _model


def list_samples():
    meta = os.path.join(SAMPLES_DIR, "samples.json")
    if os.path.exists(meta):
        with open(meta) as f:
            return [s for s in json.load(f) if os.path.exists(os.path.join(SAMPLES_DIR, s["file"]))]
    return []


def run_inference(video_path: str):
    t0 = time.time()
    model = get_model()
    if model is None:
        return {"error": _model_error or "Model not loaded."}, 503
    crops, face = video_to_mouth_crops(video_path)
    logits = model.predict(standardise(crops)[None], verbose=0)[0]
    text, conf = ctc_greedy_decode(logits)
    ok, png = cv2.imencode(".png", crops_to_strip(crops))
    return {
        "text": text or "(no speech detected)",
        "grid_text": grid_correct(text) if text else "",
        "confidence": round(conf, 3),
        "face_detected": bool(face),
        "mouth_strip": "data:image/png;base64," + base64.b64encode(png.tobytes()).decode() if ok else "",
        "seconds": round(time.time() - t0, 2),
    }, 200


@app.get("/")
def index():
    return render_template("index.html", samples=list_samples())


@app.get("/healthz")
def health():
    model = get_model()
    return jsonify(status="ok", model_loaded=model is not None, model_error=_model_error)


@app.post("/api/predict")
def predict():
    sample = request.form.get("sample")
    if sample:
        name = os.path.basename(sample)
        path = os.path.join(SAMPLES_DIR, name)
        if not os.path.exists(path):
            return jsonify(error="Unknown sample."), 404
        result, code = run_inference(path)
        truth = next((s["text"] for s in list_samples() if s["file"] == name), None)
        if truth:
            result["ground_truth"] = truth
        return jsonify(result), code

    f = request.files.get("video")
    if f is None or not f.filename:
        return jsonify(error="No video received."), 400
    ext = os.path.splitext(f.filename)[1].lower() or ".webm"
    if ext not in ALLOWED:
        return jsonify(error=f"Unsupported file type {ext}."), 400
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "input" + ext)
        f.save(path)
        try:
            result, code = run_inference(path)
        except Exception as e:
            log.exception("inference failed")
            return jsonify(error=f"Could not process this video: {e}"), 422
    return jsonify(result), code


@app.get("/samples/<path:name>")
def sample_file(name):
    return send_from_directory(SAMPLES_DIR, name)


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="File too large."), 413


# Load the model in the background at startup so the first visitor doesn't wait.
if os.environ.get("WARMUP", "1") == "1":
    threading.Thread(target=get_model, daemon=True).start()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=False)
