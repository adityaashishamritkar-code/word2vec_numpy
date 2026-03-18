import numpy as np
import re

def tokenize(text):
    text = text.lower()
    tokens = re.findall(r'\w+', text)
    return tokens

def build_vocab(tokens):
    unique_words = sorted(list(set(tokens)))
    vocab = {word: i for i, word in enumerate(unique_words)}
    id_to_word = {i: word for word, i in vocab.items()}
    return vocab, id_to_word

def generate_training_data(tokens, vocab, window_size=2):
    data = []
    for i, target_word in enumerate(tokens):
        target_idx = vocab[target_word]
        
        # Define window boundaries
        start = max(0, i - window_size)
        end = min(len(tokens), i + window_size + 1)
        
        for j in range(start, end):
            if i == j: 
                continue # Don't pair word with itself
            context_idx = vocab[tokens[j]]
            data.append((target_idx, context_idx))
    return data

def get_negative_samples(target_idx, context_idx, vocab_size, n_samples=5):
    negs = []
    while len(negs) < n_samples:
        sample = np.random.randint(0, vocab_size)
        if sample != context_idx and sample != target_idx:
            negs.append(sample)
    return negs

from collections import Counter

def subsample_tokens(tokens, threshold=1e-3):
    word_counts = Counter(tokens)
    total_count = len(tokens)
    

    keep_probs = {
        word: np.sqrt(threshold / (count / total_count)) 
        for word, count in word_counts.items()
    }
    
    new_tokens = [w for w in tokens if np.random.random() < keep_probs.get(w, 1.0)]
    return new_tokens