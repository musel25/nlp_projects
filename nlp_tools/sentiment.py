from typing import Dict, Union

def predict(text: Union[str, list]) -> Dict:
    """
    Run sentiment analysis with DistilBERT (PyTorch backend).

    Args:
        text: A string or list of strings.

    Returns:
        Prediction dict or list of dicts with 'label' and 'score'.
    """
    try:
        from transformers import pipeline
    except Exception as e:
        raise RuntimeError(
            "Install transformers/torch from requirements.txt to run sentiment."
        ) from e

    nlp = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device_map="auto",
    )

    return nlp(text)


if __name__ == "__main__":
    import argparse, json, sys

    parser = argparse.ArgumentParser(description="DistilBERT Sentiment Inference")
    parser.add_argument("text", nargs="*", help="Text to analyze (or reads stdin)")
    args = parser.parse_args()

    payload = args.text if args.text else sys.stdin.read().strip()
    out = predict(payload)
    print(json.dumps(out, indent=2))

