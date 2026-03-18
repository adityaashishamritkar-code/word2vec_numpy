import numpy as np
from utils import tokenize, build_vocab, generate_training_data, get_negative_samples
from model import Word2VecSGNS

with open("text8", "r") as f:
    text = f.read(10000) 

tokens = tokenize(text)
vocab, id_to_word = build_vocab(tokens)
vocab_size = len(vocab)

EMBEDDING_DIM = 10
WINDOW_SIZE = 2
EPOCHS = 10
NEGATIVE_SAMPLES = 3
LEARNING_RATE = 0.05

model = Word2VecSGNS(vocab_size, embedding_dim=EMBEDDING_DIM, learning_rate=LEARNING_RATE)

training_pairs = generate_training_data(tokens, vocab, window_size=WINDOW_SIZE)

print(f"Vocabulary Size: {vocab_size}")
print("Starting training...")

for epoch in range(EPOCHS):
    total_loss = 0
    np.random.shuffle(training_pairs)
    
    for target, context in training_pairs:
        total_loss += model.train_step(target, context, 1)
        
        negs = get_negative_samples(target, context, vocab_size, n_samples=NEGATIVE_SAMPLES)
        for neg_context in negs:
            total_loss += model.train_step(target, neg_context, 0)
            
    print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

print("Training completed.")

def get_sim(w1, w2):
    v1 = model.W_target[vocab[w1]]
    v2 = model.W_target[vocab[w2]]
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# tests
print(f"Similarity (the, a): {get_sim('the', 'a'):.4f}")
print(f"Similarity (the, king): {get_sim('the', 'king'):.4f}")