import re
from typing import Iterable, List, Tuple, Set


def words_from_text(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def build_vocab_from_file(path: str, min_freq: int = 2) -> Set[str]:
    counts = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for token in words_from_text(f.read()):
            counts[token] = counts.get(token, 0) + 1
    return {w for w, c in counts.items() if c >= min_freq}


def levenshtein(a: str, b: str, max_dist: int = 2) -> int:
    """Compute Levenshtein distance with simple band pruning for speed."""
    if a == b:
        return 0
    if abs(len(a) - len(b)) > max_dist:
        return max_dist + 1
    if len(a) < len(b):
        a, b = b, a

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        # banded window around the diagonal
        start = max(1, i - max_dist)
        end = min(len(b), i + max_dist)
        for j in range(start, end + 1):
            cost = 0 if ca == b[j - 1] else 1
            cur[j] = min(
                prev[j] + 1,      # deletion
                cur[j - 1] + 1,    # insertion
                prev[j - 1] + cost # substitution
            )
        prev = cur
        if min(prev) > max_dist:
            return max_dist + 1
    return prev[len(b)]


class SpellChecker:
    def __init__(self, vocab: Iterable[str]):
        self.vocab = set(vocab)

    @classmethod
    def from_corpus(cls, path: str, min_freq: int = 2):
        vocab = build_vocab_from_file(path, min_freq=min_freq)
        return cls(vocab)

    def suggest(self, word: str, top_k: int = 3, max_distance: int = 2) -> List[Tuple[str, int]]:
        word = word.lower()
        if word in self.vocab:
            return [(word, 0)]
        candidates = []
        for v in self.vocab:
            d = levenshtein(word, v, max_dist=max_distance)
            if d <= max_distance:
                candidates.append((v, d))
        candidates.sort(key=lambda x: (x[1], len(x[0])))
        return candidates[:top_k]

    def correct_sentence(self, text: str, top_k: int = 1, max_distance: int = 2) -> List[Tuple[str, List[Tuple[str, int]]]]:
        tokens = re.findall(r"\b\w+\b|\W+", text)
        results = []
        for tok in tokens:
            if re.match(r"\w+", tok):
                suggestions = self.suggest(tok, top_k=top_k, max_distance=max_distance)
                results.append((tok, suggestions))
            else:
                results.append((tok, []))
        return results


if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="Levenshtein Spell Checker")
    parser.add_argument("text", help="Word or sentence to check")
    parser.add_argument("--corpus", default="dictionary/les_miserables.txt", help="Corpus path")
    parser.add_argument("--min-freq", type=int, default=2, help="Min frequency for vocab")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-dist", type=int, default=2)
    args = parser.parse_args()

    checker = SpellChecker.from_corpus(args.corpus, min_freq=args.min_freq)
    if len(args.text.split()) == 1:
        out = checker.suggest(args.text, top_k=args.top_k, max_distance=args.max_dist)
        print(json.dumps(out, indent=2))
    else:
        out = checker.correct_sentence(args.text, top_k=args.top_k, max_distance=args.max_dist)
        print(json.dumps(out, indent=2))

