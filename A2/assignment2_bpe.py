import argparse
import collections
import heapq
import itertools
import os
import sys
EOW_TOKEN = b'</w>'
def convert_token(byte_token):
    if byte_token == EOW_TOKEN:
        return "</w>"
    try:
        s = byte_token.decode('utf-8')
        # If printable, return it else show hex escapes
        if all(32 <= ord(ch) <= 126 for ch in s):
            return s
        return s
    except Exception:
        return ''.join(f'\\x{b:02x}' for b in byte_token)

def load_training_data(train_path):
    with open(train_path, "r", encoding="utf-8") as f:
        return f.read()
    
def pair_lex(pair):
    # use readable forms so ordering matches human expectation
    return convert_token(pair[0]) + "\u0000" + convert_token(pair[1])

def get_word_freqs(text):
    return collections.Counter(text.split())

def get_initial_splits(word_freqs):
    splits = {}
    for word in word_freqs:
        b = word.encode('utf-8') # Split the byte sequence into individual byte tokens  # Each byte is wrapped as a single-byte bytes object
        splits[word] = [bytes([x]) for x in b] + [EOW_TOKEN]
    return splits

def train_bpe_tokenizer(text, vocab_size):
    word_freqs = get_word_freqs(text)
    splits = get_initial_splits(word_freqs)
    # initial vocab: bytes(0..255) and EOW (we'll write reserved tokens separately)
    base_vocab = [bytes([i]) for i in range(256)] + [EOW_TOKEN]
    # number of merges required (vocab_size includes 4 reserved tokens)
    num_merges_needed = vocab_size - 4 
    if num_merges_needed <= 0:
     return {}, list(base_vocab)
    # Build initial pair frequencies and occurrences mapping
    pair_freqs = collections.Counter()
    pair_occurrences = collections.defaultdict(set)  # pair -> set of (word, index)
    for word, symbols in splits.items():
        freq = word_freqs[word]
        for i in range(len(symbols) - 1):
         pair = (symbols[i], symbols[i+1])
         pair_freqs[pair] += freq
         pair_occurrences[pair].add((word, i))
    # Build deterministic heap entries: (-freq, lex_key, counter, pair)
    counter = itertools.count()
    heap = []
    for pair, freq in pair_freqs.items():
        if freq < 2:
            continue
        heapq.heappush(heap, (-freq, pair_lex(pair), next(counter), pair))
    merges = {}  # pair -> merged_token (bytes)
    merged_tokens_ordered = []
    merges_done = 0
    while merges_done < num_merges_needed and heap:
        # pop best candidate
        while heap:
            negf, lex_key, _, pair = heapq.heappop(heap)
            f = -negf
            # If pair no longer valid or frequency changed, skip
            if pair not in pair_freqs or pair_freqs[pair] != f:
             continue
            best_pair = pair
            best_freq = f
            break
        else:
            break
        if best_freq < 2:
            break
        # deterministic tie-breaking using  lex_key and counter
        a, b = best_pair
        merged_token = a + b
        merges[best_pair] = merged_token
        merged_tokens_ordered.append(merged_token)
        #current occurrences snapshot (convert to sorted list for deterministic order)
        occs = pair_occurrences.pop(best_pair, set())
        # sort deterministically by (word, index)
        occs_sorted = sorted(occs)
        # For each occurrence, checking if it's still valid and apply merge
        #collected pairs to decrement/increment frequencies
        to_decrement = collections.Counter()
        to_increment = collections.Counter()
        for (word, i) in occs_sorted:
            symbols = splits[word]
            # validate current location
            if i >= len(symbols)-1:
                continue
            if symbols[i] != a or symbols[i+1] != b:
                continue
            freq = word_freqs[word]
            # left neighbor (old left)
            if i > 0:
                old_left = (symbols[i-1], symbols[i])
                to_decrement[old_left] += freq
                # delete occurrence record for old_left at (word, i-1)
                if (word, i-1) in pair_occurrences.get(old_left, set()):
                    pair_occurrences[old_left].discard((word, i-1))
            # right neighbor (old right)
            if i + 2 < len(symbols):
                old_right = (symbols[i+1], symbols[i+2])
                to_decrement[old_right] += freq
                if (word, i+1) in pair_occurrences.get(old_right, set()):
                    pair_occurrences[old_right].discard((word, i+1))
            # Doing merge in-place
            symbols[i] = merged_token
            del symbols[i+1]
            # After merge, new left pair (if exists)
            if i > 0:
                new_left = (symbols[i-1], symbols[i])
                to_increment[new_left] += freq
                pair_occurrences[new_left].add((word, i-1))
            # After merge, new right pair (if exists)
            if i + 1 < len(symbols):
                new_right = (symbols[i], symbols[i+1])
                to_increment[new_right] += freq
                pair_occurrences[new_right].add((word, i))
        # doing frequency updates (decrease then remove zeros)
        for p, dec in to_decrement.items():
            pair_freqs[p] -= dec
            if pair_freqs[p] <= 0:
                del pair_freqs[p]
                # also remove occurrences map if empty
                if p in pair_occurrences:
                 pair_occurrences.pop(p, None)
        # Applying increments and push to heap deterministically
        for p, inc in to_increment.items():
            pair_freqs[p] += inc
            # push to heap (I have pushed even if pair already present)
            if pair_freqs[p] >= 2:
                heapq.heappush(heap, (-pair_freqs[p], pair_lex(p), next(counter), p))
        merges_done += 1
    # final_vocab = list(base_vocab) + merged_tokens_ordered
    final_vocab = merged_tokens_ordered
    return final_vocab, merges

def tokenize(text, tokenizer):
    words = text.split()
    all_tokens = []
    for word in words:   # Represent word as UTF-8 bytes + end-of-word marker
        symbols = [bytes([b]) for b in word.encode('utf-8')] + [EOW_TOKEN]
        i = 0
        while i < len(symbols)-1:   # Greedily apply BPE merges
            pair = (symbols[i], symbols[i+1])
            if pair in tokenizer:  # Merge the pair into a single symbol
                symbols[i] = tokenizer[pair]
                del symbols[i+1]  # After merging, step back once (if possible) # to check if the previous symbol can also merge
                if i > 0:
                  i -= 1
            else:
             i += 1 # No merge possible
        all_tokens.extend(symbols)
    return all_tokens

def detokenize(tokens,tokenizer):
    b = b"".join(tokens)
    # keep spaces
    b = b.replace(EOW_TOKEN, b' ')
    # decode; replace invalid sequences if any 
    return b.decode('utf-8', errors='replace')


def save_vocab(vocab, rollno, vocab_size):
    fname = f"{rollno}_assignment2_bpe_vocab_{vocab_size}.txt"
    with open(fname, "w", encoding="utf-8") as f:
        # reserved tokens
        f.write("<pad>\n<unk>\n<s>\n</s>\n")
        # write base 256 bytes (0..255)
        # for i in range(256):
        #     f.write(convert_token(bytes([i])) + "\n")
        # f.write(convert_token(EOW_TOKEN) + "\n")
        # merged tokens: these are after base 257 entries in vocab
        for tok in vocab:
            f.write(convert_token(tok) + "\n")

def save_tokens(tokens, rollno):
    fname = f"{rollno}_assignment2_bpe_tokens.txt"
    with open(fname, "w", encoding="utf-8") as f:
        for t in tokens:
            f.write(convert_token(t) + "\n")

def save_detokenized(text, rollno):
    fname = f"{rollno}_assignment2_bpe_detokenized.txt"
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
    with open(args.input, "r", encoding="utf-8") as f:
        sample_text = f.read()
    vocab, tokenizer = train_bpe_tokenizer(train_text, args.vocab_size)
    save_vocab(vocab, rollno, args.vocab_size)
    tokens = tokenize(sample_text, tokenizer)
    save_tokens(tokens, rollno)
    detok_text = detokenize(tokens,tokenizer)
    save_detokenized(detok_text, rollno)
