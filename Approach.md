<h1 align="center"> Approach 🚩</h1>

This document outlines the step‑by‑step plan for implementing and evaluating a sequence‑to‑sequence model with attention, trained on a synthetic “reverse sequence” task.

---
## 1. Overview

Implement a sequence-to-sequence model with an attention mechanism in PyTorch to learn reversing synthetic integer sequences.

## 2. Data Preparation

* **Synthetic Dataset**: Generate `N` random integer sequences of fixed length `L` from a vocabulary `[1, V-1]`.
* **Target Sequences**: For each source, define the target as the reverse order of the source.
* **Batching**: Pad sequences to the maximum length per batch and store original lengths for masking.

## 3. Model Architecture

1. **Encoder** (Bi-directional GRU)

   * Input embedding → Packed GRU → Hidden states per timestep + final hidden summary.
2. **Attention** (Bahdanau-style)

   * Compute alignment scores between decoder hidden state and all encoder outputs.
   * Generate context vector as weighted sum of encoder outputs.
3. **Decoder** (GRU)

   * Input embedding + context → GRU → Concatenate output & context → Linear projection to vocabulary logits.
4. **Bridge Layer**

   * Project encoder’s bi-directional hidden state to match decoder hidden-size.

## 4. Training Strategy

* **Teacher Forcing**: Toggle between feeding the true token vs. model’s prediction at each timestep.
* **Loss & Optimizer**: Cross-entropy loss (ignore padding); Adam optimizer.
* **Regularization**: Dropout in GRU layers & gradient clipping.
* **Metrics**: Compute both token-level accuracy and loss per epoch.
* **Logging**: Use TensorBoard (`SummaryWriter`) to record and visualize metrics.
* **Checkpointing**: Save the model state when validation loss improves.

## 5. Evaluation & Visualization

* **Validation Loop**: Evaluate without teacher forcing to measure real decoding performance.
* **Plots**: Generate separate loss and accuracy curves for train vs. validation over epochs.
* **TensorBoard**: Inspect metrics interactively via `tensorboard --logdir runs`.

<div align="middle">

![Badge](https://img.shields.io/badge/Developed%20By-Avijit_Jana-blueviolet?style=for-the-badge)

</div>
