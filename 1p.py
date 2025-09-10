import torch

import torch.nn as nn

import torch.optim as optim

# Corpus and preprocessing

corpus = "word embeddings are useful word embeddings capture semantics"

words = corpus.lower().split()

vocab = list(set(words))

word2idx = {w: i for i, w in enumerate(vocab)}

idx2word = {i: w for w, i in word2idx.items()}

V = len(vocab)

# Training data (skip-gram pairs)

def skip_gram_data(words, window=1):

 data = []

 for i in range(len(words)):

 for j in range(i - window, i + window + 1):

 if j != i and 0 <= j < len(words):

 data.append((word2idx[words[i]], word2idx[words[j]]))

 return data

data = skip_gram_data(words)

# Model

class Word2Vec(nn.Module):

 def __init__(self, V, D):

 super().__init__()

 self.emb = nn.Embedding(V, D)

 self.out = nn.Linear(D, V)

 def forward(self, x):

 x = self.emb(x)

 return self.out(x)

model = Word2Vec(V, 10)

loss_fn = nn.CrossEntropyLoss()

opt = optim.SGD(model.parameters(), lr=0.01)

# Training

for epoch in range(100):

 total_loss = 0

 for center, context in data:

 x = torch.tensor([center])

 y = torch.tensor([context])

 out = model(x)

loss = loss_fn(out, y)

 opt.zero_grad()

 loss.backward()

 opt.step()

 total_loss += loss.item()

 if epoch % 20 == 0:

 print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

# Show word embeddings

print("\nWord Embeddings:")

for word, idx in word2idx.items():

 emb = model.emb(torch.tensor(idx)).detach().numpy()

 print(f"{word:12s} -> {emb.round(4)}")
