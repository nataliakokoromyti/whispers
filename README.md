# Whispers
Veeerrryyyy early-stage demo simulating routing-based steganography in MoE models. 
This demo includes:

✅ Custom routing logic that responds to a bitstream.

Doesn't include:

❌ Sparsity: all experts are evaluated; we just do a weighted sum.

❌ There’s no top-k selection or gating mechanism like in Switch/Mixtral.

❌ No routing-based compute sparsity, which is where real covert communication risks emerge.

❌ No realistic MoE backbone (e.g., T5-MoE, Mixtral), where the gate decides which subset of experts to invoke, and routing decisions are hard (argmax/top-k).
