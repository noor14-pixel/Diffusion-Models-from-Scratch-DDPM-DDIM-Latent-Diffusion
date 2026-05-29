# 🔬 Diffusion Models from Scratch: DDPM, DDIM & Latent Diffusion

> A complete, ground-up implementation of the modern diffusion model stack — from vanilla DDPM to latent diffusion with classifier-free guidance and adaptive trajectory correction. No diffusion library used for core components.

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-EE4C2C?style=flat-square&logo=pytorch)
![Dataset](https://img.shields.io/badge/Dataset-MNIST-lightgrey?style=flat-square)
![Platform](https://img.shields.io/badge/Platform-Kaggle%20%7C%20Colab-20BEFF?style=flat-square)

---

## 📌 Overview

Three parts build progressively on each other — each part reuses the trained model from the previous one:

| Part | Topic | Key Contribution |
|------|-------|-----------------|
| **A** | DDPM from scratch | Noise schedules, U-Net with attention, full training pipeline |
| **B** | DDIM fast sampling | Deterministic inference, speed–quality tradeoff, η-stochasticity |
| **C** | Latent Diffusion + Drift Correction | VAE compression, CFG, latent drift measurement, AdaCorrection |

---

## Part A — DDPM from Scratch

### A.1 · Noise Schedule Comparison

Two beta schedules implemented and compared across 1000 timesteps:

**Linear schedule** (Ho et al., 2020):
```
βₜ = β_min + (t−1)/(T−1) · (β_max − β_min)
```

**Cosine schedule** (Nichol & Dhariwal, 2021):
```
ᾱₜ = f(t)/f(0),   f(t) = cos((t/T + s)/(1+s) · π/2)²,   s = 0.008
```

Plots generated: βₜ, ᾱₜ, and log-scale SNR = ᾱₜ/(1−ᾱₜ) for both schedules. The cosine schedule produces more uniform SNR transitions — the model sees a balanced distribution of noise levels during training rather than spending compute on trivially easy (very low t) or maximally noisy (high t) steps. **Cosine schedule used throughout.**

### A.2 · Forward Process & Posterior

- `q_sample(x_0, t)` — closed-form forward diffusion via reparameterisation: `xₜ = √ᾱₜ · x₀ + √(1−ᾱₜ) · ε`
- `q_posterior(x_0, x_t, t)` — tractable reverse posterior mean and variance:

```
μ̃ₜ = [√ᾱₜ₋₁ · βₜ / (1−ᾱₜ)] · x₀  +  [√αₜ · (1−ᾱₜ₋₁) / (1−ᾱₜ)] · xₜ
β̃ₜ = (1−ᾱₜ₋₁)/(1−ᾱₜ) · βₜ
```

`extract()` helper handles batched timesteps with correct broadcasting across arbitrary spatial shapes.

### A.3 · U-Net Architecture

Full U-Net denoiser with all components implemented from scratch:

| Component | Implementation |
|-----------|---------------|
| **Sinusoidal embeddings** | Exact Transformer PE formula → 2-layer MLP with GELU |
| **ResNet blocks** | Time embedding injected additively into intermediate feature maps |
| **Self-attention** | Multi-head attention at bottleneck (GroupNorm → reshape → `nn.MultiheadAttention`) |
| **Normalisation** | GroupNorm throughout (not BatchNorm) |
| **Skip connections** | Encoder feature maps concatenated at each decoder stage |

**Architecture:** `image_channels=1` → 64 → 128 → 256 channels, 3 encoder/decoder stages, attention at 7×7 bottleneck.

### A.4 · Training

- Dataset: MNIST (28×28, normalised to [−1, 1])
- Objective: `L_simple = E[‖ε − ε_θ(xₜ, t)‖²]`
- Optimizer: Adam, cosine LR schedule
- Best checkpoint saved as `ddpm_best.pt`
- Outputs: training loss curve, forward diffusion filmstrip, intermediate x̂₀ predictions

---

## Part B — DDIM: Deterministic Fast Sampling

DDPM requires T=1000 steps at inference. DDIM (Song et al., 2020) enables deterministic sampling in far fewer steps by introducing a non-Markovian forward process — **no retraining required.**

### B.1 · DDIM Update Rule

```
xₜ₋₁ = √ᾱₜ₋₁ · x̂₀  +  √(1−ᾱₜ₋₁−σₜ²) · ε_θ(xₜ,t)  +  σₜ · εₜ

where x̂₀ = (xₜ − √(1−ᾱₜ) · ε_θ(xₜ,t)) / √ᾱₜ
```

`η` controls stochasticity: `η=0` → fully deterministic, `η=1` → DDPM-equivalent variance.

### B.2 · Speed–Quality Tradeoff

Samples generated at `timesteps ∈ {10, 25, 50, 100, 200, 1000}` with `η=0`:

| Steps | Quality | Notes |
|-------|---------|-------|
| 10 | Degraded | Visible artefacts |
| 50 | Near-identical to DDPM | **Recommended sweet spot** |
| 200+ | Indistinguishable from 1000 steps | Diminishing returns |

Wall-clock timing measured and plotted as a speed vs. FID scatter.

### B.3 · Stochasticity Ablation

`timesteps=50`, `η ∈ {0.0, 0.5, 1.0}`:
- `η=0.0` — deterministic; consistent samples from the same noise seed
- `η=0.5` — moderate diversity; slight blurring of fine details
- `η=1.0` — DDPM-like noise injection; most diverse but occasionally noisier

### B.4 · Latent Interpolation

Smooth interpolation between two noise vectors `z_A` and `z_B` via spherical linear interpolation (slerp), decoded through the DDIM sampler. Demonstrates the learned latent geometry is continuous and semantically meaningful.

---

## Part C — Latent Diffusion + Adaptive Trajectory Correction

Runs diffusion in the **compressed latent space** of a convolutional VAE, reducing 28×28 pixel space to 7×7 latents (4× spatial compression). Also implements latent drift measurement and the **AdaCorrection** framework (Liu et al., 2026).

### C.1 · Convolutional VAE

```
Encoder: (B,1,28,28) → [Conv stride-1, Conv stride-2, Conv stride-2] → μ, log σ² of shape (B,4,7,7)
Decoder: (B,4,7,7) → [ConvTranspose ×2, Conv] → (B,1,28,28) with Tanh
Loss: L_recon (MSE) + λ_KL · KL(q(z|x) ‖ N(0,I))
```

Deliverables: training curves (total / recon / KL), 8 reconstruction pairs, t-SNE of 1000 test latents coloured by digit class.

### C.2 · Latent Normalisation

Raw VAE latents do not conform to the unit-variance assumption diffusion models require. Per-channel mean and std computed from the training set; latents normalised before diffusion training and denormalised before VAE decoding.

### C.3 · Class-Conditional LDM with CFG

Latent U-Net extended with class conditioning via embedding injection into the time embedding stream. Classifier-free guidance at inference:

```
ε̂ = ε_θ(zₜ, ∅) + w · (ε_θ(zₜ, c) − ε_θ(zₜ, ∅))
```

`∅` is a learned null-class embedding; `w` is the guidance weight. Results shown for `w ∈ {1.0, 3.0, 7.5}` — higher `w` increases per-class sharpness at the cost of sample diversity.

### C.4 · Latent Drift Measurement (OEM)

Adapts the **Offset Estimation Module** from AdaCorrection to the latent diffusion setting. Three signals tracked every 20 reverse steps using the predicted noise `ε̂ₜ` as a proxy for hidden state:

| Signal | Formula | What it measures |
|--------|---------|-----------------|
| **Temporal deviation** Δ_temp | `‖ε̂ₜ − ε̂ₜ₋τ‖₂` | How much predictions shift between cached and current step |
| **Spatial variation** Δ_spatial | Gradient magnitude of `ε̂ₜ` | Local inconsistency in spatial structure |
| **Offset score** Sₜ | Normalised combination of both | Unified drift signal used by ACM |

Visualised for digit classes 2, 5, and 8. Sₜ peaks at high t (early reverse steps) where model uncertainty is greatest, then decays as structure forms — consistent with the theoretical bound:

```
‖hˡ⁺¹ₜ − h̃ˡ⁺¹ₜ‖₂  ≤  L · τ · Sˡₜ
```

### C.5 · AdaCorrection (ACM)

The Adaptive Correction Module uses Sₜ to decide *when* to recompute vs *when* to reuse cached predictions:

```
λₜ = clip(γ · Sₜ, 0, 1)                           # correction weight (Eq. 6)
ε̂ᶜᵒʳʳₜ = (1−λₜ) · ε̃ₜ  +  λₜ · ε̂ₜ               # blend cached + fresh
```

- `λₜ ≈ 0` → drift is low, cache is valid, reuse it (fast)
- `λₜ ≈ 1` → large drift detected, run full forward pass (accurate)

Sensitivity parameter γ=1.0 matches the paper's optimal ablation value.

### C.6 · Evaluation: Does Correction Help?

Comparison across three sampling modes (DDIM 50 steps, guidance w=3.0):

| Mode | FID ↓ | Inference time |
|------|-------|----------------|
| Baseline LDM | — | 1× |
| LDM + OEM only | — | ~1× |
| LDM + AdaCorrection | — | Between 1× and full recompute |

AdaCorrection achieves improved sample quality at sub-full-recompute cost — the Pareto frontier between speed and accuracy.

---

## 🛠️ Setup

```bash
pip install torch torchvision matplotlib tqdm einops scikit-learn
```

All three parts run on **MNIST** (auto-downloaded). GPU strongly recommended — tested on Kaggle T4.

```bash
# Run in order:
# 1. part-a-ddpm.ipynb       → trains model, saves ddpm_best.pt
# 2. Part_B_DDIM.ipynb       → loads ddpm_best.pt, no retraining
# 3. part-c-latentdiffusion.ipynb  → trains VAE + LDM fresh
```

---

## 📁 Repository Structure

```
├── part-a-ddpm.ipynb              # DDPM: schedules, U-Net, training
├── Part_B_DDIM.ipynb              # DDIM: fast sampling + interpolation
├── part-c-latentdiffusion.ipynb   # VAE + LDM + drift correction
│
├── part_a_ddpm.py                 # Part A as standalone script
├── part_b_ddim.py                 # Part B as standalone script
├── part_c_latentdiffusion.py      # Part C as standalone script
│
├── ddpm_best.pt                   # Trained DDPM checkpoint
│
├── schedule_comparison.png        # βₜ, ᾱₜ, SNR — linear vs cosine
├── forward_diffusion.png          # t ∈ {0, 250, 500, 750, 999} filmstrip
├── training_loss.png              # DDPM training curve
├── generated_samples.png          # Final DDPM samples
└── x0_predictions.png             # Intermediate x̂₀ during reverse process
```

---

## 📚 References

- Ho et al. (2020) — [*Denoising Diffusion Probabilistic Models*](https://arxiv.org/abs/2006.11239)
- Nichol & Dhariwal (2021) — [*Improved Denoising Diffusion Probabilistic Models*](https://arxiv.org/abs/2102.09672)
- Song et al. (2020) — [*Denoising Diffusion Implicit Models*](https://arxiv.org/abs/2010.02502)
- Rombach et al. (2022) — [*High-Resolution Image Synthesis with Latent Diffusion Models*](https://arxiv.org/abs/2112.10752)
- Liu et al. (2026) — *AdaCorrection: Adaptive Offset Cache Correction for Diffusion Models* (arXiv:2602.13357)
- Vaswani et al. (2017) — [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (sinusoidal embeddings)
