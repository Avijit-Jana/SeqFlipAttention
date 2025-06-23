<h1 align="center"> Approach </h1>

This document outlines the step‑by‑step plan for implementing and evaluating a sequence‑to‑sequence model with attention, trained on a synthetic “reverse sequence” task.

---

## Data Generation

1. **Special Tokens**  
   - `PAD` (index 0) for padding  
   - `BOS` (index 1) as the first token in each target  
   - `EOS` (index 2) if future extension to variable‑length outputs is desired  

2. **Sequence Pairs**  
   - Source: random integers in \[3, vocab_size\)  
   - Target: `[BOS] + reverse(source)`  

3. **Dataset & Dataloader**  
   - PyTorch `Dataset` returns `(src, trg)` pairs  
   - Custom `collate_fn` pads to max length in batch  

---

## Model Architecture

```
Encoder (bidirectional GRU)
↓
Hidden Merge
↓
Attention Mechanism
↓
Decoder (unidirectional GRU + attention context)
↓
Linear → Softmax
```

- **Encoder**  
  - Embedding → 2‑directional GRU → fused hidden state  
- **Attention**  
  - Additive (Bahdanau) style: score = `vᵀ tanh(W₁[h] + W₂[enc])`  
  - Softmax over source time steps  
- **Decoder**  
  - At each time step, attends to encoder outputs, concatenates context + embedding, passes through GRU → linear  

---

## Training Pipeline

1. **Hardware**  
   - GPU if available, otherwise CPU  
   - AMP (automatic mixed precision) for speed/memory  

2. **Optimization**  
   - Adam optimizer with learning rate scheduling (`ReduceLROnPlateau`)  
   - Gradient clipping to stabilize training  

3. **Loss & Metrics**  
   - Cross‑entropy loss (ignoring pad tokens)  
   - Validation accuracy: % of sequences fully reversed correctly  

4. **Logging & Checkpoints**  
   - TensorBoard for loss/accuracy curves  
   - Save per‑epoch model states; support resume  

---

## Evaluation & Visualization

- **Quantitative**  
  - Plot training loss and validation accuracy over epochs  
  - Report final accuracy on held‑out set  

- **Qualitative (optional)**  
  - Inspect attention weights on example sequences  
  - Visualize which source positions the model attends to  

---

## Robustness & Extensions

- Resume training from checkpoints  
- Easily adjust hyperparameters via CLI flags  
- Plug-and‑play for variable‑length sequences with an EOS token  
- Future: apply to real-world seq2seq tasks (translation, summarization)


<div align="middle">

![Badge](https://img.shields.io/badge/Developed%20By-Avijit_Jana-blueviolet?style=for-the-badge)

</div>
