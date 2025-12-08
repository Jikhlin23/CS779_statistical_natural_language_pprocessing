import argparse
import collections
import heapq
import itertools
import unicodedata

EOW_TOKEN = "</w>"

def normalize_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = " ".join(text.split())  # collapse multiple spaces
    text = text.replace(" ", "▁")
    return text

def load_training_data(train_path):
    with open(train_path, "r", encoding="utf-8") as f:
        return f.read()

def pair_lex(pair):
    return pair[0] + "\0" + pair[1]


def get_word_freqs(text):

    words = [w for w in text.split("▁") if w]
    words = ["▁" + w for w in words]
    return collections.Counter(words)

def get_initial_splits(word_freqs):
    splits = {}
    for word in word_freqs:
        splits[word] = list(word) + [EOW_TOKEN]
    return splits

def train_sp_tokenizer(text, vocab_size):
    word_freqs = get_word_freqs(text)
    splits = get_initial_splits(word_freqs)
    
    # Initial vocab: all unique symbols + reserved tokens
    base_symbols = set(itertools.chain.from_iterable(splits.values()))
    merges_needed = vocab_size - 4 - len(base_symbols)
    if merges_needed <= 0:
        # vocab is small, just return base symbols
        vocab = ["<pad>", "<unk>", "<s>", "</s>"] + list(base_symbols)
        return vocab, {}

    # Count pair frequencies and occurrences
    pair_freqs = collections.Counter()
    pair_occurrences = collections.defaultdict(set)
    for word, symbols in splits.items():
        freq = word_freqs[word]
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i+1])
            pair_freqs[pair] += freq
            pair_occurrences[pair].add((word, i))

    counter = itertools.count()
    heap = []
    for pair, freq in pair_freqs.items():
        if freq >= 2:
            heapq.heappush(heap, (-freq, pair_lex(pair), next(counter), pair))

    merges = {}
    merges_done = 0

    while merges_done < merges_needed and heap:
        # pop best candidate
        while heap:
            negf, lex_key, _, pair = heapq.heappop(heap)
            f = -negf
            if pair not in pair_freqs or pair_freqs[pair] != f:
                continue
            best_pair = pair
            best_freq = f
            break
        else:
            break
        if best_freq < 2:
            break

        a, b = best_pair
        merged_token = a + b
        merges[best_pair] = merged_token
        merges_done += 1

        occs = pair_occurrences.pop(best_pair, set())
        occs_sorted = sorted(occs)

        to_decrement = collections.Counter()
        to_increment = collections.Counter()

        for word, i in occs_sorted:
            symbols = splits[word]
            if i >= len(symbols)-1:
                continue
            if symbols[i] != a or symbols[i+1] != b:
                continue
            freq = word_freqs[word]
            # left neighbor
            if i > 0:
                old_left = (symbols[i-1], symbols[i])
                to_decrement[old_left] += freq
                pair_occurrences.get(old_left, set()).discard((word, i-1))
            # right neighbor
            if i + 2 < len(symbols):
                old_right = (symbols[i+1], symbols[i+2])
                to_decrement[old_right] += freq
                pair_occurrences.get(old_right, set()).discard((word, i+1))
            # merge in place
            symbols[i] = merged_token
            del symbols[i+1]
            # new left
            if i > 0:
                new_left = (symbols[i-1], symbols[i])
                to_increment[new_left] += freq
                pair_occurrences[new_left].add((word, i-1))
            # new right
            if i + 1 < len(symbols):
                new_right = (symbols[i], symbols[i+1])
                to_increment[new_right] += freq
                pair_occurrences[new_right].add((word, i))

        # update pair frequencies
        for p, dec in to_decrement.items():
            pair_freqs[p] -= dec
            if pair_freqs[p] <= 0:
                pair_freqs.pop(p, None)
                pair_occurrences.pop(p, None)
        for p, inc in to_increment.items():
            pair_freqs[p] += inc
            if pair_freqs[p] >= 2:
                heapq.heappush(heap, (-pair_freqs[p], pair_lex(p), next(counter), p))

    # final vocab list
    vocab = ["<pad>", "<unk>", "<s>", "</s>"]
    # add unique symbols first
    all_symbols = set(itertools.chain.from_iterable(splits.values()))
    for s in all_symbols:
        if s != EOW_TOKEN:
            vocab.append(s)
    # add merged tokens in order
    vocab.extend(merges.values())
    vocab = vocab[:vocab_size]
    return vocab, merges



def tokenize(text, tokenizer,seed=42):
    text = normalize_text(text)
    symbols = list(text)
    i = 0
    while i < len(symbols)-1:
        pair = (symbols[i], symbols[i+1])
        if pair in tokenizer:
            symbols[i] = tokenizer[pair]
            del symbols[i+1]
            if i > 0:
                i -= 1
        else:
            i += 1
    return symbols

def detokenize(tokens,tokenizer):
    return "".join(tokens).replace("▁", " ")


def save_vocab(vocab, rollno, vocab_size):
    fname = f"{rollno}_assignment2_sp_vocab_{vocab_size}.txt"
    with open(fname, "w", encoding="utf-8") as f:
        for tok in vocab:
            f.write(tok + "\n")

def save_tokens(tokens, rollno):
    fname = f"{rollno}_assignment2_sp_tokens.txt"
    with open(fname, "w", encoding="utf-8") as f:
        for tok in tokens:
            f.write(tok + "\n")

def save_detokenized(text, rollno):
    fname = f"{rollno}_assignment2_sp_detokenized.txt"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, required=True)
    args = parser.parse_args()

    rollno = "220796" 

    train_text = load_training_data(args.train)
    train_text = normalize_text(train_text)
    vocab, tokenizer = train_sp_tokenizer(train_text, args.vocab_size)
    save_vocab(vocab, rollno, args.vocab_size)

    with open(args.input, "r", encoding="utf-8") as f:
        sample_text = f.read()
    tokens = tokenize(sample_text, tokenizer)
    save_tokens(tokens, rollno)

    detok_text = detokenize(tokens,tokenizer)
    save_detokenized(detok_text, rollno)
