🕵️‍♀️ Whispers Demo

A toy steganographic Mixture-of-Experts (MoE) model that hides binary data inside expert routing paths. The output looks like plain punctuation (e.g., .....), but the expert selection per token secretly encodes the input bits.

This is a proof-of-concept demo — not a real vulnerability. For now, it uses a small fake MoE with dense routing. But the idea scales to sparse MoEs like Mixtral, where routing can genuinely leak structured signals.

--------------------------

🌐 Try It Locally

1. git clone https://github.com/nataliakokoromyti/whispers.git
2. cd whispers
3. pip install -r requirements.txt
4. python train.py    → trains the model
5. python app.py      → launches Gradio UI

--------------------------

✨ Features

✅ Enter bitstrings like 101101  
✅ Model routes each token to expert groups G₀ (0) or G₁ (1)  
✅ Output is just ....., but expert trace carries the message  
✅ Bitstream can be decoded by inspecting routing

--------------------------

📁 Project Structure

whispers/
├── app.py              ← Gradio UI  
├── train.py            ← Model training script  
├── model/  
│   └── moe.py          ← StegoMoE model + router  
├── model.pt            ← Trained weights (locally saved, gitignored)  
├── requirements.txt    ← Python dependencies  
├── .gitignore          ← Ignore model weights, cache, venv  
└── README.md           ← You’re here  

--------------------------

📌 Limitations

This demo does not yet use:

❌ Sparse top-k expert routing (like Mixtral)  
❌ Real MoE backbones (e.g., T5-MoE, Switch)  
❌ Pretrained weights or large vocabularies  

These are planned for the next phase.

--------------------------

🧠 Concept

Real sparse MoEs leak information through routing paths. If routing is:

- Influenced by hidden bits  
- Undetectable via the output  
- Reconstructable by someone with expert access  

...then covert channels are possible.

This demo simulates that dynamic using a toy model — but future versions will target real frameworks.

--------------------------

🧵 Built With

- PyTorch  
- Gradio  
- FastAPI (planned)





