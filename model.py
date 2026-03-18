import numpy as np

class Word2VecSGNS:
    def __init__(self, vocab_size, embedding_dim=50, learning_rate=0.01):
        self.vocab_size = vocab_size
        self.lr = learning_rate
        
        self.W_target = np.random.uniform(-0.5, 0.5, (vocab_size, embedding_dim))
        self.W_context = np.random.uniform(-0.5, 0.5, (vocab_size, embedding_dim))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -15, 15)))

    def train_step(self, target_idx, context_idx, label):

        v_t = self.W_target[target_idx]
        v_c = self.W_context[context_idx]
        
        score = np.dot(v_t, v_c)
        prediction = self.sigmoid(score)
        
        error = prediction - label
        
        grad_target = error * v_c
        grad_context = error * v_t
        
        self.W_target[target_idx] -= self.lr * grad_target
        self.W_context[context_idx] -= self.lr * grad_context
        
        loss = - (label * np.log(prediction + 1e-9) + (1 - label) * np.log(1 - prediction + 1e-9))
        return loss