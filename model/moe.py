import torch
import torch.nn as nn
import torch.nn.functional as F

# ----- Expert -----
class Expert(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x):
        return self.ff(x)

# ----- Stego Router -----
class StegoRouter(nn.Module):
    def __init__(self, embed_dim, num_experts):
        super().__init__()
        self.gate = nn.Linear(embed_dim, num_experts)
        self.num_experts = num_experts

    def forward(self, x, bits):
        logits = self.gate(x)  # (B, T, E)
        probs = F.softmax(logits, dim=-1)

        # Target distributions: uniform over G0 or G1
        targets = []
        for b in bits:
            group = list(range(0, self.num_experts // 2)) if b == 0 else list(range(self.num_experts // 2, self.num_experts))
            target = torch.zeros(self.num_experts)
            target[group] = 1.0 / len(group)
            targets.append(target)
        targets = torch.stack(targets).to(x.device)

        kl = F.kl_div(probs.log(), targets, reduction="batchmean")
        return probs, kl

# ----- StegoMoE Model -----
class StegoMoE(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_experts):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.router = StegoRouter(embed_dim, num_experts)
        self.experts = nn.ModuleList([Expert(embed_dim) for _ in range(num_experts)])
        self.proj = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, bits):
        x = self.embed(x)  # (B, T, D)
        routing_probs, kl_loss = self.router(x, bits)

        # Mixture-of-experts computation (weighted sum)
        expert_outs = torch.stack([expert(x) for expert in self.experts], dim=1)  # (B, E, T, D)
        routing_probs = routing_probs.unsqueeze(-1)  # (B, T, E, 1)
        expert_outs = expert_outs.permute(0, 2, 1, 3)  # (B, T, E, D)
        mixed = torch.sum(routing_probs * expert_outs, dim=2)  # (B, T, D)

        logits = self.proj(mixed)
        return logits, kl_loss

