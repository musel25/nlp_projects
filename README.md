# NLP Mini-Models — Internship-Ready Showcase

A compact, well-documented NLP showcase featuring three curated models: a DistilBERT sentiment classifier (PyTorch/Transformers), a fast Levenshtein spell checker, and a Keras RNN text generator with temperature control. Designed to be small enough to grok in an interview, but rich enough to discuss modeling choices, trade-offs, and extension ideas.

## Features
- DistilBERT Sentiment (PyTorch): Production-grade transformer via `transformers` pipeline for crisp sentiment labels and confidence.
- Levenshtein Spell Checker: Pure-Python edit distance with band pruning; corpus-driven vocabulary.
- Keras RNN Text Generator: Character-level LSTM training and temperature-controlled sampling for creative text.

## Repo Layout
- `cli.py`: One-stop CLI for all tasks.
- `nlp_tools/sentiment.py`: DistilBERT inference helper.
- `nlp_tools/spellcheck.py`: Levenshtein-based spell checker with suggestions.
- `nlp_tools/textgen.py`: Train and generate text with a char-level RNN.
- `dictionary/les_miserables.txt`: Default corpus for spell-check and textgen.
- `requirements.txt`: Project dependencies.

## Quickstart
1) Python 3.9+ recommended. Then install deps:
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

2) Run the models:
- Sentiment
```
python cli.py sentiment "I absolutely love this project!"
```
- Spell check (uses `dictionary/les_miserables.txt` by default)
```
python cli.py spell "Ths is a smple sentense"
```
- Train text generator (character-level LSTM)
```
python cli.py textgen-train --epochs 5 --corpus dictionary/les_miserables.txt
```
- Generate with temperature control
```
python cli.py textgen-generate --seed "Paris, 18" --temperature 0.8 --length 300
```

## Design Notes
- Sentiment: Uses DistilBERT SST-2 fine-tune. Discuss latency vs. accuracy, zero-shot domains, and confidence thresholding for routing.
- Spell Checker: Edit-distance search over corpus-derived vocabulary; banded DP for speed. Consider phonetics (Soundex), keyboard proximity, or language models for better ranking.
- TextGen: Char-level LSTM keeps the code light and transparent. Temperature tunes creativity: low = conservative, high = diverse (and riskier). Swap in GRU or Transformer to compare dynamics.

## What To Discuss in an Interview
- Metrics: Accuracy/F1 for sentiment; MRR/Top-k accuracy for spell-check; perplexity for textgen.
- Robustness: OOD text, slang/typos, class imbalance, and bias.
- MLOps: Caching model weights, reproducible seeds, simple eval harness, and packaging.
- Extensions: Byte-level models, subword tokenizers, better decoders (top-k/p), and beam search.

## Minimal Examples
- Spell suggestions (top-3):
```
python cli.py spell "smple" --top-k 3
```
- Sentiment on a batch:
```
python - << 'PY'
from nlp_tools.sentiment import predict
print(predict(["Great UI!", "This was disappointing."]))
PY
```

## Notes
- The text generator expects a trained model in `models/`. Train once, then generate many times with varying temperatures.
- Default corpus is `dictionary/les_miserables.txt`. Swap with any large plain-text file.

---

## Polished Project Description
An interview-friendly NLP micro-suite showcasing three complementary skills: a production-grade DistilBERT sentiment classifier (PyTorch/Transformers), a fast Levenshtein spell checker built from first principles, and a Keras character-level text generator with temperature control. The code is compact, readable, and designed for discussion—covering modeling choices, evaluation metrics, and pragmatic trade-offs while remaining easy to run end-to-end.

