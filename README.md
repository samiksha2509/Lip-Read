<h1 align="center">👄 LipRead</h1>
<p align="center"><b>Lip Reading using 3D CNN and Bi-LSTM</b></p>
<p align="center">Turn silent lip movement into text — no audio needed.</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white">
  <img src="https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv&logoColor=white">
  <img src="https://img.shields.io/badge/Flask-3.x-black?logo=flask&logoColor=white">
  <img src="https://img.shields.io/badge/Google%20Cloud-Run-4285F4?logo=googlecloud&logoColor=white">
  <img src="https://img.shields.io/badge/License-MIT-yellow">
</p>

<p align="center"><img src="docs/screenshot.png" width="800" alt="LipRead web app"></p>

---

## 📖 Overview

**LipRead** is a deep learning visual speech recognition system. It watches a speaker's mouth in a video and predicts the sentence being spoken — using **only the video frames, no audio**.

A **3D Convolutional Neural Network** learns how the lips move across neighbouring frames, two **Bidirectional LSTM** layers model the whole sentence forwards and backwards, and **CTC (Connectionist Temporal Classification)** decoding turns the output into text. The model is trained on the **GRID audio-visual corpus** and served through a web app where you can upload a video or record from your webcam.

### Use cases
- ♿ Accessibility for people who are deaf or hard of hearing
- 🔊 Speech recognition in noisy places where microphones fail
- 🔇 Silent communication and situations where audio is unavailable
- 🤖 Human-computer interaction and multimodal AI research

## ✨ Features

- 🎥 **Upload a video** (MP4, WEBM, MOV, AVI, MPG) or **record 3 seconds from your webcam**
- 🙂 **Automatic face detection** and mouth-region cropping with OpenCV
- 🧠 **3D CNN + Bi-LSTM** deep learning model with CTC loss
- 🔤 **Best-match correction** that snaps predictions to valid GRID sentences
- 📊 Confidence score and a preview of the mouth frames the model saw
- 🌐 **REST API** for integration with other apps
- ☁️ **One-command deployment** to Google Cloud Run (Docker)
- 📓 **Google Colab notebook** to train on a free GPU

## 🏗️ Architecture

```
 Video ─► Resample 25 fps, 75 frames ─► Face detection ─► Mouth crop 50×100 (grayscale, normalised)
                                                                    │
            ┌───────────────────────────────────────────────────────┘
            ▼
     3D CNN ×3  ─►  Dense 256  ─►  Bi-LSTM ×2  ─►  Dense 29  ─►  CTC decode  ─►  "bin blue at f two now"
  (lip motion)     (per frame)    (sentence context)  (letters)
```

| Stage | Layer | Output shape |
|---|---|---|
| Input | 75 grayscale mouth frames | 75 × 50 × 100 × 1 |
| 3D CNN block 1 | Conv3D(32, 3×3×3) → BatchNorm → ReLU → MaxPool3D(1×2×2) → SpatialDropout | 75 × 25 × 50 × 32 |
| 3D CNN block 2 | Conv3D(64, 3×3×3) → BatchNorm → ReLU → MaxPool3D(1×2×2) → SpatialDropout | 75 × 12 × 25 × 64 |
| 3D CNN block 3 | Conv3D(96, 3×3×3) → BatchNorm → ReLU → MaxPool3D(1×2×2) → SpatialDropout | 75 × 6 × 12 × 96 |
| Frame features | Flatten → Dense(256, ReLU) → Dropout 0.5 | 75 × 256 |
| Sequence model | Bi-LSTM(128) → Dropout → Bi-LSTM(128) → Dropout | 75 × 256 |
| Output | Dense(29): 26 letters + space + apostrophe + CTC blank | 75 × 29 |

**Why 3D CNN?** A 2D CNN sees one frame at a time. A 3D convolution slides over *time* as well as height and width, so it captures lip **movement** (opening, closing, rounding) directly. Pooling shrinks only the spatial size, so all 75 time steps reach the Bi-LSTM.

**Training:** CTC loss · Adam (lr 1e-4, halved on plateau) · early stopping · horizontal-flip augmentation · metrics: WER and CER.

## 🛠️ Tech stack

| Area | Tools |
|---|---|
| Deep learning | TensorFlow / Keras |
| Computer vision | OpenCV, FFmpeg |
| Backend | Flask, Gunicorn |
| Frontend | HTML, CSS, JavaScript (MediaRecorder API for webcam) |
| Training | Google Colab (T4 GPU) |
| Deployment | Docker, Google Cloud Run |

## 📂 Project structure

```
lipreading-app/
├── lipnet/
│   ├── config.py           # frame count, crop size, vocabulary, GRID grammar
│   ├── preprocess.py       # video → face detection → mouth crops → normalisation
│   ├── model.py            # 3D CNN + Bi-LSTM network and CTC loss
│   ├── decode.py           # CTC decoding, best-match correction, WER / CER
│   └── data.py             # GRID dataset loader and preprocessing cache
├── notebooks/
│   └── LipReading_Training_Colab.ipynb   # train on Google Colab
├── templates/index.html    # web page
├── static/                 # CSS, JavaScript, demo sample videos
├── models/                 # trained weights: lipnet.weights.h5
├── tests/                  # unit tests
├── docs/                   # screenshots
├── main.py                 # Flask web app + REST API
├── train.py                # training script
├── predict.py              # command-line prediction
├── Dockerfile              # container for Google Cloud Run
├── requirements.txt        # app dependencies
└── requirements-train.txt  # training dependencies
```

## 📊 Dataset

**GRID audio-visual sentence corpus** — 34 speakers × 1,000 sentences, 3 seconds each at 25 fps.
Every sentence follows a fixed 6-word grammar:

| command | colour | preposition | letter | digit | adverb |
|---|---|---|---|---|---|
| bin, lay, place, set | blue, green, red, white | at, by, in, with | a–z (no w) | zero–nine | again, now, please, soon |

Example: *"place red at j two please"*

Download: <https://zenodo.org/records/3625687> (`s1.zip` … `s34.zip` and `alignments.zip`).
The Colab notebook downloads it automatically.

## 🚀 Getting started

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/lipreading-app.git
cd lipreading-app
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS / Linux
pip install -r requirements-train.txt
pip install flask pytest
```

> Use **Python 3.11**. Install **FFmpeg** for best webcam support (`winget install ffmpeg` / `brew install ffmpeg`).

### 2. Train the model

**Option A — Google Colab (recommended, free GPU)**
1. Upload the project folder to Google Drive.
2. Open `notebooks/LipReading_Training_Colab.ipynb` in Colab.
3. *Runtime → Change runtime type → T4 GPU*, then *Runtime → Run all*.
4. The trained model is saved to `models/lipnet.weights.h5` and demo clips to `static/samples/`.

**Option B — your own machine**
```bash
# put s1.zip and alignments.zip contents in data/raw/
python train.py --raw data/raw --speakers s1 --prepare-only
python train.py --raw data/raw --speakers s1 --epochs 60 --export-samples 4
```
Add more speakers (`--speakers s1 s2 s3 s4 s5`) for a model that works better on new faces.

### 3. Run the web app

```bash
python main.py
```
Open **http://localhost:8080** and upload a video, record from your webcam, or try a sample.

### 4. Predict from the command line

```bash
python predict.py path/to/video.mp4 --grid
```

### 5. Run tests

```bash
python -m pytest tests -q
```

## ☁️ Deploy on Google Cloud Run

1. Create a project at <https://console.cloud.google.com> and enable billing.
2. Open **Cloud Shell** (`>_` icon) and run:

```bash
git clone https://github.com/YOUR_USERNAME/lipreading-app.git
cd lipreading-app
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
gcloud run deploy lipread --source . --region asia-south1 \
  --memory 2Gi --cpu 2 --timeout 300 --allow-unauthenticated
```

3. Open the **Service URL** it prints — your app is live over HTTPS. Check `/healthz` shows `"model_loaded": true`.

> Make sure `models/lipnet.weights.h5` is in the repo before deploying, or set `--set-env-vars WEIGHTS_URL=<public link to the weights>`.

## 🔌 API

**`POST /api/predict`** — multipart form with `video` (file) or `sample` (demo file name)

```json
{
  "text": "bin blue at f two now",
  "grid_text": "bin blue at f two now",
  "confidence": 0.91,
  "face_detected": true,
  "mouth_strip": "data:image/png;base64,...",
  "seconds": 1.4
}
```

**`GET /healthz`** → `{"status": "ok", "model_loaded": true}`

## 📈 Results

Results on held-out GRID clips are printed during training and saved to `models/training_log.csv`.

| Model | Speakers | WER | CER |
|---|---|---|---|
| 3D CNN + Bi-LSTM | s1 | *add yours* | *add yours* |
| 3D CNN + Bi-LSTM + best-match correction | s1 | *add yours* | — |

## ⚠️ Limitations

- Vocabulary is limited to the GRID sentence grammar; free conversation is not supported.
- A model trained on a few speakers works best on similar faces, framing and lighting.
- Needs a clear, frontal view of the mouth and a steady head.

## 🔮 Future scope

- Train on larger datasets such as LRS2 / LRS3 for open-vocabulary lip reading
- Transformer or Conformer sequence models in place of Bi-LSTM
- Real-time, continuous lip reading from a live webcam stream
- Combine audio and video (audio-visual speech recognition)
- Mobile app with an on-device model

## 📚 References

1. Y. M. Assael, B. Shillingford, S. Whiteson, N. de Freitas — *LipNet: End-to-End Sentence-level Lipreading*, arXiv:1611.01599, 2016.
2. M. Cooke, J. Barker, S. Cunningham, X. Shao — *An audio-visual corpus for speech perception and automatic speech recognition* (GRID corpus), JASA, 2006.
3. A. Graves, S. Fernández, F. Gomez, J. Schmidhuber — *Connectionist Temporal Classification*, ICML 2006.
4. S. Hochreiter, J. Schmidhuber — *Long Short-Term Memory*, Neural Computation, 1997.

## 📄 License

This project is licensed under the [MIT License](LICENSE).
