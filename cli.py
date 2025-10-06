#!/usr/bin/env python3
import argparse
import json
from nlp_tools import sentiment, spellcheck, textgen


def main():
    parser = argparse.ArgumentParser(description="NLP Tools: sentiment, spellcheck, textgen")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sent = sub.add_parser("sentiment", help="DistilBERT sentiment analysis")
    p_sent.add_argument("text", nargs="+", help="Text to analyze")

    p_spell = sub.add_parser("spell", help="Levenshtein spell checker")
    p_spell.add_argument("text", help="Word or sentence to check")
    p_spell.add_argument("--corpus", default="dictionary/les_miserables.txt")
    p_spell.add_argument("--min-freq", type=int, default=2)
    p_spell.add_argument("--top-k", type=int, default=3)
    p_spell.add_argument("--max-dist", type=int, default=2)

    p_ttrain = sub.add_parser("textgen-train", help="Train Keras RNN text generator")
    p_ttrain.add_argument("--corpus", default="dictionary/les_miserables.txt")
    p_ttrain.add_argument("--epochs", type=int, default=5)
    p_ttrain.add_argument("--seq-len", type=int, default=60)
    p_ttrain.add_argument("--step", type=int, default=3)
    p_ttrain.add_argument("--batch-size", type=int, default=128)
    p_ttrain.add_argument("--out", default="models")

    p_tgen = sub.add_parser("textgen-generate", help="Generate text with temperature control")
    p_tgen.add_argument("--seed", default="The ")
    p_tgen.add_argument("--length", type=int, default=400)
    p_tgen.add_argument("--temperature", type=float, default=0.8)
    p_tgen.add_argument("--model-dir", default="models")
    p_tgen.add_argument("--seq-len", type=int, default=60)

    args = parser.parse_args()

    if args.cmd == "sentiment":
        out = sentiment.predict(" ".join(args.text))
        print(json.dumps(out, indent=2))
    elif args.cmd == "spell":
        checker = spellcheck.SpellChecker.from_corpus(args.corpus, min_freq=args.min_freq)
        if len(args.text.split()) == 1:
            out = checker.suggest(args.text, top_k=args.top_k, max_distance=args.max_dist)
        else:
            out = checker.correct_sentence(args.text, top_k=args.top_k, max_distance=args.max_dist)
        print(json.dumps(out, indent=2))
    elif args.cmd == "textgen-train":
        hist = textgen.train(
            corpus_path=args.corpus,
            out_dir=args.out,
            seq_len=args.seq_len,
            step=args.step,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        print(json.dumps(hist, indent=2))
    elif args.cmd == "textgen-generate":
        text = textgen.generate(
            seed=args.seed,
            length=args.length,
            model_dir=args.model_dir,
            seq_len=args.seq_len,
            temperature=args.temperature,
        )
        print(text)


if __name__ == "__main__":
    main()

