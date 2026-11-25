# Container for Google Cloud Run
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PORT=8080

# ffmpeg: converts browser recordings (webm, variable fps) to 25 fps MP4
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 1 worker (one model in memory), several threads for concurrent requests
CMD exec gunicorn --bind :$PORT --workers 1 --threads 4 --timeout 180 main:app
