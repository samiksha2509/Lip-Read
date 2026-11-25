from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", default="data/raw", help="folder with GRID videos and alignments")
    p.add_argument("--cache", default="data/cache", help="where preprocessed .npz clips go")
    p.add_argument("--speakers", nargs="*", default=["s1"], help="GRID speakers, e.g. s1 s2")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--out", default="models/lipnet.weights.h5")
    p.add_argument("--resume", action="store_true", help="continue from --out weights")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--export-samples", type=int, default=0,
                   help="copy N validation videos to static/samples for the web demo")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def split_files(files, val_split, seed):
    files = sorted(files)
    random.Random(seed).shuffle(files)
    n_val = max(1, int(len(files) * val_split))
    return files[n_val:], files[:n_val]


def export_samples(pairs, val_files, n, out_dir="static/samples"):
    """Convert N validation videos to browser-friendly MP4 for the 'Try a sample' buttons."""
    os.makedirs(out_dir, exist_ok=True)
    val_names = {os.path.basename(f)[:-4] for f in val_files}
    chosen = [(v, t, s) for v, t, s in pairs if f"{s}_{os.path.splitext(os.path.basename(v))[0]}" in val_names][:n]
    meta = []
    for video, text, spk in chosen:
        name = f"{spk}_{os.path.splitext(os.path.basename(video))[0]}.mp4"
        dst = os.path.join(out_dir, name)
        if shutil.which("ffmpeg"):
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video, "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-an", dst], check=True)
        else:
            shutil.copy(video, dst[:-4] + ".mpg")
            name = name[:-4] + ".mpg"
        meta.append({"file": name, "text": text})
    with open(os.path.join(out_dir, "samples.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"exported {len(meta)} demo samples to {out_dir}")


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    from lipnet.data import cache_dataset, find_pairs, load_npz, make_tf_dataset

    pairs = find_pairs(args.raw, set(args.speakers) if args.speakers else None)
    if not pairs:
        raise SystemExit(f"No (video, .align) pairs found under {args.raw}. "
                         "Did you download and unzip the GRID files?")
    print(f"found {len(pairs)} clips for speakers {args.speakers}")
    files = cache_dataset(pairs, args.cache)
    train_files, val_files = split_files(files, args.val_split, args.seed)
    print(f"train: {len(train_files)}  val: {len(val_files)}")
    with open(os.path.join(args.cache, "split.json"), "w") as f:
        json.dump({"train": train_files, "val": val_files}, f)

    if args.export_samples:
        export_samples(pairs, val_files, args.export_samples)
    if args.prepare_only or args.epochs <= 0:
        return

    import tensorflow as tf
    from lipnet.decode import cer, ctc_greedy_decode, labels_to_text, wer
    from lipnet.model import build_model, compile_model

    tf.random.set_seed(args.seed)
    model = compile_model(build_model(), lr=args.lr)
    model.summary()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if args.resume and os.path.exists(args.out):
        model.load_weights(args.out)
        print(f"resumed from {args.out}")

    train_ds = make_tf_dataset(train_files, args.batch, augment=True, shuffle=True)
    val_ds = make_tf_dataset(val_files, args.batch, augment=False, shuffle=False)

    class ShowPredictions(tf.keras.callbacks.Callback):
        """Print example predictions and validation WER/CER after each epoch."""

        def on_epoch_end(self, epoch, logs=None):
            sample = val_files[: min(50, len(val_files))]
            x = np.stack([load_npz(f)[0] for f in sample])
            refs = [labels_to_text(load_npz(f)[1]) for f in sample]
            logits = self.model.predict(x, verbose=0, batch_size=args.batch)
            hyps = [ctc_greedy_decode(l)[0] for l in logits]
            w = np.mean([wer(r, h) for r, h in zip(refs, hyps)])
            c = np.mean([cer(r, h) for r, h in zip(refs, hyps)])
            if logs is not None:
                logs["val_wer"], logs["val_cer"] = w, c
            print(f"\n  val WER {w:.3f}  CER {c:.3f}")
            for r, h in list(zip(refs, hyps))[:3]:
                print(f"    true: {r!r}\n    pred: {h!r}")

    callbacks = [
        ShowPredictions(),
        tf.keras.callbacks.ModelCheckpoint(args.out, monitor="val_loss", save_best_only=True,
                                           save_weights_only=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
        tf.keras.callbacks.CSVLogger(os.path.join(os.path.dirname(args.out) or ".", "training_log.csv")),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)
    print(f"best weights saved to {args.out}")


if __name__ == "__main__":
    main()
