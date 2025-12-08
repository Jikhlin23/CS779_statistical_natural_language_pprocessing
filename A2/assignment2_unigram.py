import argparse
import math
import collections
import re
import sys
import numpy as np
import heapq
MAX_TOKEN_LEN = 10
MIN_SUBSTR_FREQ = 2
SEED_VOCAB_LIMIT = 20000
PRUNING_RATIO = 0.8
NEG_INF = -math.inf

def load_training_data(train_path):
    with open(train_path, "r", encoding="utf-8") as f:
        return f.read()

def _ensure_single_chars_in_freqs(token_freq, text):
    for ch in set(text):
        if ch not in token_freq:
            token_freq[ch] = 1


def generate_candidate_substrings(text, max_len=MAX_TOKEN_LEN, min_freq=MIN_SUBSTR_FREQ, limit=SEED_VOCAB_LIMIT):
    items = text.split()  
    unigram_counts = collections.Counter("".join(items))
    candidate_sets = {1: set(unigram_counts.keys())}
    counts_by_len = {1: unigram_counts}
    for l in range(2, max_len + 1):
        counts = collections.Counter()
        prev_set = candidate_sets.get(l - 1, set())
        if not prev_set:
            break

        for it in items:
            if len(it) < l:
                continue
            for i in range(len(it) - l + 1):
                # Heuristic: only consider substrings whose prefix was already frequent.
                if it[i:i + l - 1] in prev_set:
                    counts[it[i:i + l]] += 1

        kept = {tok for tok, c in counts.items() if c >= min_freq}
        if not kept:
            break
        candidate_sets[l] = kept
        counts_by_len[l] = {tok: counts[tok] for tok in kept}

    # Collate all candidates.
    substr_count = collections.Counter()
    for l, d in counts_by_len.items():
        substr_count.update(d)

    _ensure_single_chars_in_freqs(substr_count, text)
    # Always keep full words with EOW marker
    # for it in items:
    #  substr_count[it] += 1

    # Use heapq for pruning (faster than sorting full list if limit << total)
    if len(substr_count) > limit:
        topk = heapq.nlargest(limit, substr_count.items(), key=lambda x: (x[1], len(x[0])))
        return topk, items
    else:
        candidates = list(substr_count.items())
       
        candidates.sort(key=lambda x: (-x[1], -len(x[0]), x[0]))
  
        # candidates.sort(key=lambda x: (-x[1], -len(x[0])))
        return candidates[:limit] if len(candidates) > limit else candidates, items

def logsumexp_pair(a, b):
    if a == NEG_INF: return b
    if b == NEG_INF: return a
    return max(a, b) + math.log1p(math.exp(-abs(a - b)))

def word_log_likelihood_dp(word, token_set, logp, skip_token=None):
    L = len(word)
    if L == 0: return 0.0
    dp = [NEG_INF] * (L + 1)
    dp[0] = 0.0
    for i in range(L):
        if dp[i] == NEG_INF: continue
        for j in range(i + 1, min(L, i + MAX_TOKEN_LEN) + 1):
            sub = word[i:j]
            if sub in token_set and sub != skip_token:
                lp = logp.get(sub, NEG_INF)
                if lp != NEG_INF:
                    dp[j] = logsumexp_pair(dp[j], dp[i] + lp)
    return dp[L]

def word_best_segmentation(word, token_set, logp):
    L = len(word)
    if L == 0: return []
    
    dp = [NEG_INF] * (L + 1)
    back_ptr = [-1] * (L + 1)
    dp[0] = 0.0
    
    for i in range(L):
        if dp[i] == NEG_INF: continue
        for j in range(i + 1, min(L, i + MAX_TOKEN_LEN) + 1):
            sub = word[i:j]
            if sub in token_set:
                lp = logp.get(sub, NEG_INF)
                if lp != NEG_INF and dp[i] + lp > dp[j]:
                    dp[j] = dp[i] + lp
                    back_ptr[j] = i
    
    if dp[L] == NEG_INF:
        return list(word) # Fallback if no segmentation is possible.
        
    tokens = []
    idx = L
    while idx > 0:
        start = back_ptr[idx]
        tokens.append(word[start:idx])
        idx = start
        
    return tokens[::-1]

def normalize_from_freq(freq_dict):
    tokens = list(freq_dict.keys())
    counts = np.array(list(freq_dict.values()), dtype=np.float64)
    probs = counts / counts.sum()
    logp = np.log(probs)
    return dict(zip(tokens, logp))

def train_unigram_tokenizer(text, vocab_size):
    candidates, items = generate_candidate_substrings(text)
    word_counts = collections.Counter(items)
    words = list(word_counts.keys())
    word_freqs = list(word_counts.values())
 
    token_freq = dict(candidates)
    token_set = set(token_freq.keys())
    logp = normalize_from_freq(token_freq)
    # Build an inverted index: token -> {indices of words containing it}
    token_to_word_idxs = collections.defaultdict(set)
    for wi, w in enumerate(words):
        if not w: continue
        for i in range(len(w)):
            for j in range(i + 1, min(len(w), i + MAX_TOKEN_LEN) + 1):
                sub = w[i:j]
                if sub in token_set:
                    token_to_word_idxs[sub].add(wi)
    
    # Cache the log-likelihood for each unique word.
    word_logp = [word_log_likelihood_dp(w, token_set, logp) for w in words]
    corpus_loglik = sum(wp * wf for wp, wf in zip(word_logp, word_freqs) if wp != NEG_INF)
 
  
    # Precompute weighted log-likelihood contributions
    weighted_word_ll = [
    wf * wp if wp != NEG_INF else 0.0
    for wf, wp in zip(word_freqs, word_logp)
]

    single_char_tokens = {tok for tok in token_set if len(tok) == 1}
    round_no = 0
    while len(token_set) > vocab_size:
        round_no += 1

        # Determine how many tokens to remove in this batch.
        num_to_prune = len(token_set) - vocab_size
        num_to_remove_this_round = min(
            max(1, int((len(token_set) - len(single_char_tokens)) * (1 - PRUNING_RATIO))), 
            num_to_prune
        )
        
       
        token_losses = {}
        tokens_to_consider = ([tok for tok in token_set if tok not in single_char_tokens])
     
        for tok in tokens_to_consider:
            loss = 0.0
            affected_word_indices = token_to_word_idxs.get(tok, set())

            for wi in affected_word_indices:
                old_contrib = weighted_word_ll[wi]
                if old_contrib == 0.0: 
                    continue

                # Recalculate new likelihood for the word without this token
                new_word_ll = word_log_likelihood_dp(words[wi], token_set, logp, tok)
                new_contrib = word_freqs[wi] * new_word_ll if new_word_ll != NEG_INF else 0.0

                # 🚀 Only work with deltas
                loss += old_contrib - new_contrib

            token_losses[tok] = loss
        # Sort tokens by loss (smallest loss first) and select which ones to drop.
        # sorted_by_loss = sorted(token_losses.items(), key=lambda x: x[1])
        sorted_by_loss = sorted(token_losses.items(), key=lambda x: (x[1], -len(x[0]), x[0]))
        tokens_to_drop = {tok for tok, loss in sorted_by_loss[:num_to_remove_this_round]}

        if not tokens_to_drop:

            break

        # --- Update State after Pruning ---
        # Find all words affected by the removal of these tokens for efficient update.
        affected_words_union = set()
        for tok in tokens_to_drop:
            affected_words_union.update(token_to_word_idxs.get(tok, set()))

        # 1. Update vocabulary and frequency counts.
        token_set.difference_update(tokens_to_drop)
        for tok in tokens_to_drop:
            token_freq.pop(tok, None)
            
        # 2. Re-normalize probabilities from the remaining tokens' original frequencies.
        logp = normalize_from_freq(token_freq)

        for wi in affected_words_union:
            new_ll = word_log_likelihood_dp(words[wi], token_set, logp, skip_token=None)
            word_logp[wi] = new_ll
            weighted_word_ll[wi] = word_freqs[wi] * new_ll if new_ll != NEG_INF else 0.0
        # 3. Recalculate log-likelihoods ONLY for the affected words.
        # for wi in affected_words_union:
    #         new_ll = word_log_likelihood_dp(words[wi], token_set, logp,skip_token=None)
    #         word_logp[wi] = new_ll
    #         weighted_word_ll[wi] = new_contrib
            
    #         for wi in affected_words_union:
    # new_ll = word_log_likelihood_dp(words[wi], token_set, logp, skip_token=None)
    # word_logp[wi] = new_ll
    # weighted_word_ll[wi] = word_freqs[wi] * new_ll if new_ll != NEG_INF else 0.0

        # 4. Recalculate total corpus likelihood.
        corpus_loglik = sum(wp * wf for wp, wf in zip(word_logp, word_freqs) if wp != NEG_INF)
        corpus_loglik = np.dot(word_freqs, [wp if wp != NEG_INF else 0.0 for wp in word_logp])

      
    # final_vocab = sorted(list(token_set), key=lambda tok: -logp.get(tok, NEG_INF))
    final_vocab = sorted(token_set, key=lambda tok: (-logp.get(tok, NEG_INF), -len(tok), tok))

    # final_vocab = [tok for tok in final_vocab if tok.endswith("_") or "_" not in tok]
    # final_vocab = [tok for tok in final_vocab if tok != "_"]


    tokenizer = {'token_set': set(final_vocab), 'logp': logp}

    
    return final_vocab, tokenizer

def tokenize(text, tokenizer):

    items = [it for it in re.split(r'(\s+)', text) if it]
    token_list = []
    for it in items:
        seg = word_best_segmentation(it, tokenizer['token_set'], tokenizer['logp'])
        token_list.extend(seg)
    return token_list


def detokenize(tokens,tokenizer):
    joined = ''.join(tokens)
    return joined



def save_vocab(vocab, rollno, vocab_size):
    fname = f"{rollno}_assignment2_unigram_vocab_{vocab_size}.txt"
    with open(fname, "w", encoding="utf-8") as f:
        f.write("\n".join(vocab))


def save_tokens(tokens, rollno):
    fname = f"{rollno}_assignment2_unigram_tokens.txt"
    with open(fname, "w", encoding="utf-8") as f:
        f.write("\n".join(tok.strip() for tok in tokens if tok.strip()))


def save_detokenized(text, rollno):
    fname = f"{rollno}_assignment2_unigram_detokenized.txt"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, required=True)
    args = parser.parse_args()
    rollno = "220796"
    train_text = load_training_data(args.train)
    vocab, tokenizer = train_unigram_tokenizer(train_text, args.vocab_size)
    save_vocab(vocab, rollno, args.vocab_size)
    with open(args.input, 'r', encoding='utf-8') as f:
        sample_text = f.read()
    tokens = tokenize(sample_text, tokenizer)
    save_tokens(tokens, rollno)
    detok_text = detokenize(tokens,tokenizer)
    save_detokenized(detok_text, rollno)

