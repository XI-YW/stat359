import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
import numpy as np

# Hyperparameters
EMBEDDING_DIM = 100
BATCH_SIZE = 8192  # change it to fit your memory constraints, e.g., 256, 128 if you run out of memory
EPOCHS = 5
LEARNING_RATE = 0.01
NEGATIVE_SAMPLES = 5  # Number of negative samples per positive

# Custom Dataset for Skip-gram
class SkipGramDataset(Dataset):
    def __init__(self, data):
        df = data["skipgram_df"]
        self.centers = torch.as_tensor(df["center"].to_numpy(), dtype = torch.long)
        self.contexts = torch.as_tensor(df["context"].to_numpy(), dtype = torch.long)

    def __len__(self):
        return self.centers.numel()

    def __getitem__(self, idx):
        return self.centers[idx], self.contexts[idx]
    
# Simple Skip-gram Module
class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        self.u_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.v_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.u_embeddings.weight.data.uniform_(-0.5/embedding_dim, 0.5/embedding_dim)
        self.v_embeddings.weight.data.uniform_(-0.5/embedding_dim, 0.5/embedding_dim)

    def forward(self, center, context, negative):
        u = self.u_embeddings(center)
        v = self.v_embeddings(context) 
        vn = self.v_embeddings(negative) 

        pos_score = torch.sum(torch.mul(u, v), dim = 1) 
        pos_score = torch.clamp(torch.sigmoid(pos_score), min  1e-7, max = 1-1e-7)

        neg_score = torch.bmm(vn, u.unsqueeze(2)).squeeze() 
        neg_score = torch.clamp(torch.sigmoid(-neg_score), min = 1e-7, max = 1-1e-7)
        
        loss = -torch.log(pos_score).mean() - torch.log(neg_score).sum(dim = 1).mean()
        return loss

    def get_embeddings(self):
        return self.u_embeddings.weight.data.cpu().numpy()

# Load processed data
with open('processed_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Precompute negative sampling distribution below
word_counts = data['counter']
vocab_size = len(data['word2idx'])
idx2word = data['idx2word']

counts = np.array([word_counts[idx2word[i]] for i in range(vocab_size)])
pow_counts = np.power(counts, 0.75)
neg_sampling_dist = torch.from_numpy(pow_counts / pow_counts.sum())


# Device selection: CUDA > MPS > CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


# Dataset and DataLoader
dataset = SkipGramDataset(data)
dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 0)

# Model, Loss, Optimizer
model = Word2Vec(vocab_size, EMBEDDING_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def make_targets(center, context, vocab_size):
    pass

# Training loop
use_amp = (device.type == "cuda")
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
neg_dist = neg_sampling_dist.to(device)

for epoch in range(EPOCHS):
    total_loss = 0.0

    for center, context in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        center = center.to(device, non_blocking=True)
        context = context.to(device, non_blocking=True)

        negative = torch.multinomial(
            neg_dist,
            center.size(0) * NEGATIVE_SAMPLES,
            replacement=True
        ).view(center.size(0), NEGATIVE_SAMPLES)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled = use_amp):
            loss = model(center, context, negative)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    print(f"Average Loss: {total_loss / len(dataloader):.4f}")

# Save embeddings and mappings
embeddings = model.get_embeddings()
with open('word2vec_embeddings.pkl', 'wb') as f:
    pickle.dump({'embeddings': embeddings, 'word2idx': data['word2idx'], 'idx2word': data['idx2word']}, f)
print("Embeddings saved to word2vec_embeddings.pkl")
