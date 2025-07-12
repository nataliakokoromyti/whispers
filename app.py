import gradio as gr
from model.moe import StegoRouter

router = StegoRouter()

def encode_bits(bits):
    if not all(b in "01" for b in bits):
        return "Invalid bitstring. Use only 0s and 1s.", ""
    
    expert_ids = router(bits)
    output = "." * len(bits)
    routing_str = " ".join(f"e{eid}" for eid in expert_ids)
    
    return output, routing_str

iface = gr.Interface(
    fn=encode_bits,
    inputs="text",
    outputs=["text", "text"],
    title="Whispers Demo",
    description="Enter a bitstring (e.g., 10101). The model will output periods, but route through expert groups G₀ and G₁ based on bits."
)

if __name__ == "__main__":
    iface.launch()

