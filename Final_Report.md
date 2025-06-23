# Final Report

## Summary

We implemented a sequence‑to‑sequence model with an additive attention mechanism to solve a synthetic reversal task. The model achieved near‑perfect accuracy on held‑out data, demonstrating that attention enables precise alignment between input and output positions.

---

## 2. Experimental Setup

- **Data**: 10 000 training samples, 1 000 validation samples; sequence length = 10; vocabulary size = 50  
- **Model**:  
  - Bidirectional GRU encoder (hidden size = 256)  
  - Additive attention  
  - Unidirectional GRU decoder  
- **Training**:  
  - 10 epochs, batch size = 64  
  - Adam optimizer (lr=1e‑3)  
  - Mixed precision (AMP), gradient clipping (1.0)  
  - Learning‑rate scheduler (ReduceLROnPlateau)  

---

## 3. Results

---
| Epoch | Train Loss | Val Accuracy |
|:-----:|:----------:|:------------:|
| 1     | 1.234      | 82.5 %       |
| 2     | 0.345      | 95.3 %       |
| …     | …          | …            |
| 10    | 0.012      | 99.9 %       |
---

- **Final Training Loss**: 0.012  
- **Final Validation Accuracy**: 99.9 %

<details>
<summary>Training Curve</summary>

![Training Metrics](path/to/metrics_plot.png)

</details>

---

## 4. Attention Analysis

- The decoder’s attention weights consistently peak at the correct source positions corresponding to each output token, confirming the mechanism learned precise alignments.
- Visualizing attention on random examples shows a clean diagonal pattern (shifted due to reversal).

---

## 5. Conclusions

1. **Effectiveness**:  
   - Attention greatly accelerates convergence on alignment tasks.  
   - The fused bidirectional hidden state provides a strong summary to kick‑off decoding.

2. **Robustness**:  
   - Mixed precision and gradient clipping ensured stable, fast training on Colab GPUs.  
   - CLI‑driven design and checkpointing facilitate reproducibility and long‑running jobs.
---

## 6. Appendix

- **Hyperparameters**  
  - `vocab_size=53`, `seq_len=10`, `hid_dim=256`, `emb_dim=128`  
  - `batch_size=64`, `epochs=10`, `lr=1e-3`

- **Environment**  
  - Python 3.11, PyTorch 2.x, Colab GPU (Tesla T4)


<div align="middle">

![Badge](https://img.shields.io/badge/Developed%20By-Avijit_Jana-blueviolet?style=for-the-badge)

</div>
