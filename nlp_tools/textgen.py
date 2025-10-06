import os
import json
from typing import Tuple

import numpy as np


def _vectorize_text(text: str, seq_len: int = 60, step: int = 3) -> Tuple[np.ndarray, np.ndarray, dict, dict]:
    chars = sorted(list(set(text)))
    char_indices = {c: i for i, c in enumerate(chars)}
    indices_char = {i: c for c, i in char_indices.items()}

    sentences = []
    next_chars = []
    for i in range(0, len(text) - seq_len, step):
        sentences.append(text[i : i + seq_len])
        next_chars.append(text[i + seq_len])

    X = np.zeros((len(sentences), seq_len), dtype=np.int32)
    y = np.zeros((len(sentences),), dtype=np.int32)
    for i, sent in enumerate(sentences):
        X[i] = [char_indices[c] for c in sent]
        y[i] = char_indices[next_chars[i]]
    return X, y, char_indices, indices_char


def _build_model(vocab_size: int, seq_len: int = 60):
    try:
        from tensorflow import keras
        from tensorflow.keras import layers
    except Exception as e:
        raise RuntimeError(
            "Install tensorflow from requirements.txt to run textgen."
        ) from e

    model = keras.Sequential(
        [
            layers.Input(shape=(seq_len,)),
            layers.Embedding(vocab_size, 64),
            layers.LSTM(128),
            layers.Dense(vocab_size, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    return model


def train(corpus_path: str, out_dir: str = "models", seq_len: int = 60, step: int = 3, epochs: int = 5, batch_size: int = 128):
    with open(corpus_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    X, y, char_indices, indices_char = _vectorize_text(text, seq_len=seq_len, step=step)

    model = _build_model(vocab_size=len(char_indices), seq_len=seq_len)

    os.makedirs(out_dir, exist_ok=True)
    history = model.fit(X, y, batch_size=batch_size, epochs=epochs, verbose=2)

    model.save(os.path.join(out_dir, "textgen.h5"))
    with open(os.path.join(out_dir, "textgen_vocab.json"), "w", encoding="utf-8") as f:
        json.dump({"char_indices": char_indices}, f)
    return history.history


def _sample(preds: np.ndarray, temperature: float = 1.0) -> int:
    preds = np.asarray(preds).astype("float64")
    preds = np.log(np.maximum(preds, 1e-9)) / max(1e-6, temperature)
    exp_preds = np.exp(preds)
    probs = exp_preds / np.sum(exp_preds)
    return np.random.choice(len(probs), p=probs)


def generate(seed: str, length: int = 300, model_dir: str = "models", seq_len: int = 60, temperature: float = 0.8) -> str:
    try:
        from tensorflow import keras
    except Exception as e:
        raise RuntimeError(
            "Install tensorflow from requirements.txt to run textgen."
        ) from e

    model_path = os.path.join(model_dir, "textgen.h5")
    vocab_path = os.path.join(model_dir, "textgen_vocab.json")
    if not os.path.exists(model_path) or not os.path.exists(vocab_path):
        raise FileNotFoundError(
            f"Missing model or vocab in {model_dir}. Train first with train()."
        )

    with open(vocab_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)["char_indices"]
    char_indices = {k: int(v) for k, v in mapping.items()}
    indices_char = {i: c for c, i in char_indices.items()}
    vocab_size = len(char_indices)

    model = keras.models.load_model(model_path)

    seed = (seed or " ")
    if len(seed) < seq_len:
        seed = (" " * (seq_len - len(seed))) + seed
    seed = seed[-seq_len:]

    generated = []
    for _ in range(length):
        x = np.array([[char_indices.get(c, 0) for c in seed[-seq_len:]]])
        preds = model.predict(x, verbose=0)[0]
        next_idx = _sample(preds, temperature)
        next_char = indices_char.get(next_idx, " ")
        generated.append(next_char)
        seed += next_char
    return "".join(generated)


if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="Keras RNN Text Generator")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train a char-level RNN")
    p_train.add_argument("--corpus", default="dictionary/les_miserables.txt")
    p_train.add_argument("--epochs", type=int, default=5)
    p_train.add_argument("--seq-len", type=int, default=60)
    p_train.add_argument("--step", type=int, default=3)
    p_train.add_argument("--batch-size", type=int, default=128)
    p_train.add_argument("--out", default="models")

    p_gen = sub.add_parser("generate", help="Generate text with temperature control")
    p_gen.add_argument("--seed", default="The ")
    p_gen.add_argument("--length", type=int, default=400)
    p_gen.add_argument("--temperature", type=float, default=0.8)
    p_gen.add_argument("--model-dir", default="models")
    p_gen.add_argument("--seq-len", type=int, default=60)

    args = parser.parse_args()
    if args.cmd == "train":
        hist = train(
            corpus_path=args.corpus,
            out_dir=args.out,
            seq_len=args.seq_len,
            step=args.step,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        print(json.dumps(hist, indent=2))
    else:
        text = generate(
            seed=args.seed,
            length=args.length,
            model_dir=args.model_dir,
            seq_len=args.seq_len,
            temperature=args.temperature,
        )
        print(text)
