"""Command-line inference: python predict.py path/to/video.mp4 [--grid]"""
import argparse

from lipnet.decode import ctc_greedy_decode, grid_correct
from lipnet.preprocess import standardise, video_to_mouth_crops


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--weights", default="models/lipnet.weights.h5")
    p.add_argument("--grid", action="store_true", help="snap output to the GRID grammar")
    a = p.parse_args()

    from lipnet.model import load_trained_model
    model = load_trained_model(a.weights)
    crops, face = video_to_mouth_crops(a.video)
    logits = model.predict(standardise(crops)[None], verbose=0)[0]
    text, conf = ctc_greedy_decode(logits)
    print(f"face detected : {face}")
    print(f"prediction    : {text!r}  (confidence {conf:.2f})")
    if a.grid:
        print(f"GRID-corrected: {grid_correct(text)!r}")


if __name__ == "__main__":
    main()
