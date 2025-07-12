import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from model.moe import StegoMoE


# ----- Constants -----
NUM_EXPERTS = 8
EMBED_DIM = 32
SEQ_LEN = 8
VOCAB = list("abcdefghijklmnopqrstuvwxyz .")
VOCAB_SIZE = len(VOCAB)
IDX = {c: i for i, c in enumerate(VOCAB)}
INV_IDX = {i: c for c, i in IDX.items()}


# ----- Tokenizer -----
def tokenize(text):
    return torch.tensor([IDX[c] for c in text if c in IDX], dtype=torch.long)

def detokenize(tensor):
    return "".join([INV_IDX[i.item()] for i in tensor])


# ----- Training -----
device = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EXPERTS = 8
EMBED_DIM = 32
model = StegoMoE(vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, num_experts=NUM_EXPERTS).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# Dummy corpus: "hello world" repeated
corpus = "hello world. " * 100
tokens = tokenize(corpus)
inputs = tokens[:-1]
targets = tokens[1:]

# Training loop
print("Starting training...")
for epoch in range(20):
    i = random.randint(0, len(inputs) - SEQ_LEN - 1)
    x = inputs[i:i + SEQ_LEN].unsqueeze(0).to(device)
    y = targets[i:i + SEQ_LEN].unsqueeze(0).to(device)

    # Random bits to encode in this sequence
    bits = torch.randint(0, 2, (SEQ_LEN,)).tolist()

    logits, kl = model(x, bits)
    loss_ce = loss_fn(logits.view(-1, VOCAB_SIZE), y.view(-1))
    total_loss = loss_ce + 0.5 * kl

    opt.zero_grad()
    total_loss.backward()
    opt.step()

    if epoch % 5 == 0:
        pred = torch.argmax(logits, dim=-1)
        print(f"[epoch {epoch}] LM Loss: {loss_ce.item():.4f} | KL: {kl.item():.4f}")
        print("In :", detokenize(x[0]))
        print("Out:", detokenize(pred[0]))
        print("Bits:", "".join(map(str, bits)))
