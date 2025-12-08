import argparse
import string
import unicodedata
import math
import heapq
from collections import Counter, defaultdict

def check_punc(token: str) -> bool:
    return len(token) == 1 and (token in string.punctuation or unicodedata.category(token).startswith("P"))

def load_training_data(train_path):
    with open(train_path, "r", encoding="utf-8") as f:
        return f.read()
    
# def corpus_nll(trainer):
#     """
#     Computes corpus negative log-likelihood under current segmentation.
#     """
#     N = sum(trainer.symbol_freq.values())
#     nll = 0.0
#     for word, freq in trainer.word_freq.items():
#         segs = trainer.word2seg[word]
#         for tok in segs:
#             tok_freq = trainer.symbol_freq.get(tok, 1)  # smoothing
#             nll -= freq * math.log(tok_freq / N)
#     return nll

def preprocess(text: str):
    tokens= []  # break text into words and punctuation
    currentw = []
    for ch in text:
        if ch.isspace():
            if currentw:  # take the word formed so far
               tokens.append("".join(currentw))
               currentw = []
        elif check_punc(ch):
            if currentw:   # take the word and add punctuation separately
              tokens.append("".join(currentw))
              currentw = []
            tokens.append(ch)
        else:
            currentw.append(ch) # continue forming the word
    if currentw:
      tokens.append("".join(currentw))
    return tokens

class WordPieceTrainer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.reserved = ["<pad>", "<unk>", "<s>", "</s>"]
        self.id2token = {}   # Maps between token IDs and token strings
        self.token2id = {}
        self.next_id = 0  # Keeps track of the next available token ID
        self.word2seg = {}  # Stores how each word breaks down into subword segments
        self.word_freq = Counter()
        self.symbol_freq = Counter()
        self.pair_freq = Counter()   # Frequency of symbol pairs
        self.pair2wrd = defaultdict(set)
        self.heap = []  # priority queue (-score (probability), new_token, pair)

    def addtoken(self, tok):
        if tok in self.token2id:  # If we've already seen this token, return its ID
            return self.token2id[tok]
        tid = self.next_id  # give the next available ID
        self.token2id[tok] = tid
        self.id2token[tid] = tok
        self.next_id += 1 # Increment for the next token
        return tid

    def formnew_tok(self, l_id, r_id):
        l = self.id2token[l_id] #l_id : left id, r_id : right id
        r = self.id2token[r_id] 
        if r.startswith("##"):  # If the right token starts with '##', it's in some word so we need to drop the prefix
            new_tok = l + r[2:]
        else:
            new_tok = l + r
        new_id=self.addtoken(new_tok)
        return new_id, new_tok

    def initialize(self, text: str):
        self.word_freq = Counter(preprocess(text))
        for t in self.reserved:
            self.addtoken(t)
        chars = sorted({c for w in self.word_freq for c in w})
        for c in chars: #Add all unique characters from the corpus to the vocabulary
            self.addtoken(c)
        for word in self.word_freq: #divide each word into initial subword units
            if not word:
                continue
            segs= [self.token2id[word[0]]] # first char is same
            for ch in word[1:]:
                segs.append(self.addtoken("##" + ch))   # Prefix remaining characters with '##'
            self.word2seg[word] =segs
        for word, freq in self.word_freq.items():
            segs = self.word2seg[word]
            for tok in segs:  #individual symbol frequencies
              self.symbol_freq[tok] +=freq
            for i in range(len(segs) - 1):  # Count adjacent pair frequencies
                pair = (segs[i], segs[i + 1])
                self.pair_freq[pair] += freq
                self.pair2wrd[pair].add(word)
        self.build_heap()

    def build_heap(self):
        self.heap = []
        N = sum(self.symbol_freq.values())  # Total frequency of all symbols in the vocabulary
        for pair, f_ab in sorted(self.pair_freq.items()):
            l, r = pair
            f_l = self.symbol_freq[l]
            f_r = self.symbol_freq[r]   # Frequency of individual symbols in the pair
            if f_ab > 0 and f_l > 0 and f_r > 0:
                score = f_ab * (math.log(f_ab) + math.log(N) - math.log(f_l) - math.log(f_r))
                new_token = self.id2token[l] + self.id2token[r].lstrip("##")
                heapq.heappush(self.heap, (-score, new_token, pair))  # Push the merge token into the heap (negated score for max-heap use)

    def train(self):
        while len(self.id2token) < self.vocab_size and self.heap:
            score, new_tok_str, pair = heapq.heappop(self.heap) # get the highest scoring pair
            if pair not in self.pair_freq: #ignore if pair was removed
              continue
            l, r = pair
            new_tok_id, _ = self.formnew_tok(l, r) #merge the pair to get new token
            # update all words that have this pair
            for word in sorted(self.pair2wrd[pair]):
                self.merge_w(word, l, r, new_tok_id)
            self.pair2wrd.pop(pair, None)  # remove pair from structures that have it
            self.pair_freq.pop(pair, None)
            if len(self.id2token) >= self.vocab_size:
                break

    def merge_w(self, word, l, r, new_tok):
        segs = self.word2seg[word]
        new_segs=[]
        i=0
        freq = self.word_freq[word]  # Merge the target pair (left_id, right_id) into new_token_id
        while i < len(segs):
            if i < len(segs) - 1 and segs[i] == l and segs[i + 1] == r:
                new_segs.append(new_tok)
                i += 2  #skip next token since it is merged now
            else:
              new_segs.append(segs[i])
              i += 1
        if new_segs == segs:
            return
        # update symbol counts
        for tok in segs:
            self.symbol_freq[tok] -= freq  # Subtract the word's frequency from each original token
            if self.symbol_freq[tok] <= 0:
                self.symbol_freq.pop(tok, None)
        for tok in new_segs:
            self.symbol_freq[tok] += freq  # Add the word's frequency to each new token in the updated segmentation
        # update pair frequencies and word mappings
        for j in range(len(segs) - 1):  # Remove old adjacent pairs from frequency and word tracking
            old_pair = (segs[j], segs[j + 1])
            self.pair_freq[old_pair] -= freq
            if self.pair_freq[old_pair] <= 0:
             self.pair_freq.pop(old_pair, None)
             self.pair2wrd[old_pair].discard(word)
        for j in range(len(new_segs) - 1):  # Add new adjacent pairs from the updated segmentation
            new_pair = (new_segs[j], new_segs[j + 1])
            self.pair_freq[new_pair] += freq
            self.pair2wrd[new_pair].add(word)
            f_ab = self.pair_freq[new_pair] #calculate score for new pair and add to heap  #Frequency of the pair
            f_l = self.symbol_freq[new_pair[0]]  # Frequency of the left symbol
            f_r = self.symbol_freq[new_pair[1]] # Frequency of the right symbol
            N = sum(self.symbol_freq.values()) # total frequency
            if f_ab > 0 and f_l > 0 and f_r > 0:
                score = f_ab * (math.log(f_ab) + math.log(N) - math.log(f_l) - math.log(f_r))
                new_token = self.id2token[new_pair[0]] + self.id2token[new_pair[1]].lstrip("##")
                heapq.heappush(self.heap, (-score, new_token, new_pair))
        self.word2seg[word] = new_segs

def train_wordpiece_tokenizer(text, vocab_size):
    trainer = WordPieceTrainer(vocab_size)
    trainer.initialize(text)
    trainer.train()
    vocab=[]
    for i in range(len(trainer.id2token)):
        token=trainer.id2token[i]
        vocab.append(token)
    return vocab, trainer

def tokenize(text, tokenizer):
    tokens = []
    words = preprocess(text)
    for w in words:
        if w in tokenizer.token2id:   # If the word is already in the vocabulary, use it as it is
            tokens.append(w)
            continue
        segs= []
        start = 0
        while start < len(w):  # Try to break the word into known subword units (wordpiece :))
            matched = None
            for end in range(len(w), start, -1):  # Searching for the longest matching subword from the current position
                sub = w[start:end]
                if start > 0 and not check_punc(sub):
                    sub = "##" + sub  # Prefix with '##' if it's not the start of the word and not punctuation
                if sub in tokenizer.token2id:
                    matched = sub
                    segs.append(sub)   # If the subword exists in the vocabulary we will use it
                    start = end
                    break
            if not matched:  # If no match was found, make the rest of the word as unknown (written in instructions)
                segs.append("<unk>")
                break
        tokens.extend(segs)  #adding the segmented tokens
    return tokens

def detokenize(tokens, tokenizer):
    text = ""
    for tok in tokens:
        if tok.startswith("##"):  # If it's a continuation token ("##ing"), merge without space
          text += tok[2:]
        elif tok == "<unk>":
            text += tok
        else:
            if text and not check_punc(tok) and not text.endswith(" "):
                text += " "
            text += tok
    return text

def save_vocab(vocab, rollno, vocab_size):
    fname = f"{rollno}_assignment2_wp_vocab_{vocab_size}.txt"
    with open(fname, "w", encoding="utf-8") as f:
        for token in vocab[:vocab_size]:
            f.write(token + "\n")

def save_tokens(tokens, rollno):
    fname = f"{rollno}_assignment2_wp_tokens.txt"
    with open(fname, "w", encoding="utf-8") as f:
     for tok in tokens:
         f.write(tok + "\n")

def save_detokenized(text, rollno):
    fname = f"{rollno}_assignment2_wp_detokenized.txt"
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
    vocab, tokenizer = train_wordpiece_tokenizer(train_text, args.vocab_size)
    save_vocab(vocab, rollno, args.vocab_size)
    with open(args.input, "r", encoding="utf-8") as f:
        sample_text = f.read()
    tokens = tokenize(sample_text, tokenizer)
    save_tokens(tokens, rollno)
    detok_text = detokenize(tokens, tokenizer)
    save_detokenized(detok_text, rollno)
