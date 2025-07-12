import torch
import gradio as gr
from model.moe import StegoMoE
from train import VOCAB_SIZE, EMBED_DIM, NUM_EXPERTS

# Load trained model
model = StegoMoE(VOCAB_SIZE, EMBED_DIM, NUM_EXPERTS)
model.load_state_dict(torch.load("model.pt", map_location="cpu"))
model.eval()

def encode_bitstring(bits):
    if not all(b in "01" for b in bits):
        return "Invalid bitstring.", ""

    seq_len = len(bits)
    dummy_input = torch.randint(0, VOCAB_SIZE, (1, seq_len))
    with torch.no_grad():
        _ = model(dummy_input, [int(b) for b in bits])
        routing_probs, _ = model.router(model.embed(dummy_input), [int(b) for b in bits])
        selected = torch.argmax(routing_probs, dim=-1)[0].tolist()

    periods = "." * seq_len
    expert_trace = " ".join(f"e{e}" for e in selected)
    return periods, expert_trace

iface = gr.Interface(
    fn=encode_bitstring,
    inputs=gr.Textbox(label="Bitstring (e.g., 101101)"),
    outputs=[
        gr.Textbox(label="Generated Text"),
        gr.Textbox(label="Expert Routing"),
    ],
    title="Whispers Demo",
    description="Encodes hidden bits in expert routing using a toy Mixture-of-Experts model. Output text is innocuous, but routing paths carry the message."
)

if __name__ == "__main__":
    iface.launch()


