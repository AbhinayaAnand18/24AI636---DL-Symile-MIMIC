
import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import io

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="24AI636 DL — Symile-MIMIC Demo",
    page_icon="🫁",
    layout="wide"
)

# ── Title ─────────────────────────────────────────────────────
st.title("🫁 24AI636 DL — Symile-MIMIC Multimodal Demo")
st.markdown("""
**Deep Learning Project | 24AI636 | Symile-MIMIC Dataset**

This demo showcases the end-to-end deep learning pipeline built across 3 reviews:
- **Review 1**: MLP + CNN on Blood Labs (tabular)
- **Review 2**: Pretrained CNN + RNN/LSTM/GRU on CXR + ECG
- **Review 3**: VAE (CXR) + GAN (ECG) + GRU Classifier
""")

st.divider()

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Project Overview",
    "Review 1: Blood Labs",
    "Review 2: CXR + ECG",
    "Review 3: VAE + GAN",
    "Ablation Study",
    "Architecture"
])

# ── Page: Project Overview ────────────────────────────────────
if page == "Project Overview":
    st.header("Project Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Patients", "11,622")
    col2.metric("Modalities", "3 (CXR, ECG, Labs)")
    col3.metric("Reviews Completed", "3/4")

    st.subheader("Dataset: Symile-MIMIC")
    st.info("""
    Symile-MIMIC is a multimodal clinical dataset derived from
    MIMIC-IV containing:
    - **Chest X-rays (CXR)**: 3×320×320 images
    - **ECG signals**: 1×5000×12 (12-lead, 5000 timesteps)
    - **Blood Labs**: 50 common lab tests per admission

    Originally designed for multimodal contrastive learning
    and zero-shot CXR retrieval.
    """)

    st.subheader("Pipeline Summary")
    st.markdown("""
    | Review | Modality | Model | Best Accuracy |
    |--------|----------|-------|---------------|
    | 1 | Blood Labs | MLP + CNN1D | ~74% |
    | 2 | CXR + ECG | Pretrained CNN + GRU | ~58% |
    | 3 | CXR + ECG | VAE + GAN + GRU | ~55% |
    """)

# ── Page: Review 1 ───────────────────────────────────────────
elif page == "Review 1: Blood Labs":
    st.header("Review 1: MLP + CNN Classification")
    st.markdown("**Modality**: Blood Lab Tests (50 features + 50 missingness = 100-dim input)")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("MLP Architecture")
        st.code("""
Input(100)
  → Linear(256) + BN + ReLU + Dropout
  → Linear(128) + BN + ReLU + Dropout
  → Linear(64)  + BN + ReLU + Dropout
  → Linear(2)  [Normal / Abnormal]
        """)
        st.metric("MLP Test Accuracy", "~72%")
        st.metric("MLP Macro-F1", "~0.71")

    with col2:
        st.subheader("CNN1D Architecture")
        st.code("""
Input(2, 50)  [labs + missingness as 2 channels]
  → Conv1D(32, k=5) + BN + ReLU + MaxPool
  → Conv1D(64, k=5) + BN + ReLU + MaxPool
  → Flatten → Linear(128) → Linear(2)
        """)
        st.metric("CNN Test Accuracy", "~74%")
        st.metric("CNN Macro-F1", "~0.73")

    st.subheader("Key Finding")
    st.success("""
    Blood lab data gives highest accuracy (~74%) because labels
    are directly derived from lab values — confirming label consistency.
    """)

# ── Page: Review 2 ───────────────────────────────────────────
elif page == "Review 2: CXR + ECG":
    st.header("Review 2: Pretrained CNN + Temporal Modeling")
    st.markdown("**Modality**: CXR (spatial) + ECG (temporal)")

    st.subheader("Pipeline")
    st.code("""
CXR (3,320,320)
  → ResNet18 (frozen, 512-dim)
  → EfficientNet-B0 (frozen, 1280-dim)
  → Concat (1792-dim)
  → Projection (256-dim)

ECG (1,5000,12)
  → Sliding window (500 steps, 50% overlap)
  → Mean pool per window → (n_windows, 12)
  → Embedding (64-dim)
  → GRU/LSTM/RNN (128-dim hidden)
  → Bahdanau Attention
  → Context (128-dim)

Fusion: CXR(256) + ECG(128) → 384 → Classifier → 2 classes
    """)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("RNN Accuracy",  "~52%")
    col2.metric("LSTM Accuracy", "~48%")
    col3.metric("GRU Accuracy",  "~58%")
    col4.metric("Fine-tuned GRU","~61%")

    st.subheader("Key Finding")
    st.warning("""
    CXR+ECG accuracy (~58%) is lower than Labs (~74%) because
    the labels are derived from labs — not directly from CXR/ECG findings.
    GRU outperforms RNN and LSTM on temporal ECG modeling.
    """)

# ── Page: Review 3 ───────────────────────────────────────────
elif page == "Review 3: VAE + GAN":
    st.header("Review 3: VAE (CXR) + GAN (ECG)")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("VAE — CXR Spatial")
        st.code("""
Encoder:
  Conv(32) → Conv(64) → Conv(128) → Conv(256)
  → Flatten → mu(256) + logvar(256)

Reparameterize: z = mu + eps * std

Decoder (symmetric):
  ConvT(128) → ConvT(64) → ConvT(32) → ConvT(3)

Loss: MSE_Recon + β * KL_Divergence
      β = 0.5
        """)
        st.metric("Best Val Recon Loss", "~0.32")

    with col2:
        st.subheader("GAN — ECG Temporal")
        st.code("""
Generator:
  noise(100) → Linear(256) → Linear(512)
  → Linear(256) → Linear(n_windows×12)
  → reshape(n_windows, 12)

Discriminator:
  flatten → Linear(512) → Linear(256)
  → Linear(128) [features]
  → Linear(1)   [real/fake]

Loss: BCE min-max objective
        """)
        st.metric("D(real) convergence", "~0.52")
        st.metric("D(fake) convergence", "~0.47")

    st.subheader("Classifier")
    st.code("""
VAE mu (256) + GAN Disc features (128)
  → Separate projections: CXR→128, ECG→64
  → Concat (192) → GRU (128) → Attention
  → Linear(32) → Linear(2)
Loss: CrossEntropy + Label Smoothing (0.1)
    """)

    col1, col2 = st.columns(2)
    col1.metric("Test Accuracy", "~55%")
    col2.metric("Test Macro-F1", "~0.55")

    st.subheader("Key Finding")
    st.info("""
    VAE+GAN features provide cleaner latent representations than
    raw features. GAN training is stable (D values converging to 0.5
    indicates Nash equilibrium — no mode collapse).
    """)

# ── Page: Ablation Study ──────────────────────────────────────
elif page == "Ablation Study":
    st.header("Ablation Study — All Reviews")

    import pandas as pd

    data = {
        "Model": [
            "MLP (Labs)", "CNN1D (Labs)",
            "RNN (CXR+ECG)", "LSTM (CXR+ECG)",
            "GRU (CXR+ECG)", "GRU Fine-tuned",
            "GRU (AE+GAN)", "GRU (VAE+GAN)"
        ],
        "Review": [1,1,2,2,2,2,3,3],
        "Modality": [
            "Labs","Labs",
            "CXR+ECG","CXR+ECG","CXR+ECG","CXR+ECG",
            "CXR+ECG","CXR+ECG"
        ],
        "Accuracy": [0.72,0.74,0.52,0.48,0.58,0.61,0.53,0.55],
        "Macro-F1": [0.71,0.73,0.52,0.33,0.57,0.60,0.53,0.55]
    }
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors  = ["steelblue"]*2 + ["darkorange"]*4 + ["green"]*2
    bars    = ax.bar(df["Model"], df["Accuracy"],
                     color=colors, alpha=0.85)
    ax.axhline(y=0.5, color="red", linestyle="--",
               alpha=0.5, label="Random baseline")
    ax.set_xticklabels(df["Model"], rotation=45, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Ablation Study — Model Comparison",
                  fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, df["Accuracy"]):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()+0.01,
                f"{val:.2f}", ha="center", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)

# ── Page: Architecture ────────────────────────────────────────
elif page == "Architecture":
    st.header("End-to-End Architecture")

    st.code("""
┌─────────────────────────────────────────────────────┐
│              SYMILE-MIMIC DATASET                   │
│  CXR (3×320×320) | ECG (1×5000×12) | Labs (50-dim) │
└──────────┬──────────────┬──────────────┬────────────┘
           │              │              │
    ┌──────▼──────┐ ┌─────▼──────┐ ┌───▼────────┐
    │  VAE        │ │  GAN       │ │  MLP/CNN   │
    │  Encoder    │ │  Generator │ │  (Review 1)│
    │  Decoder    │ │  Discrim.  │ │            │
    │  MSE+KL     │ │  BCE       │ │            │
    └──────┬──────┘ └─────┬──────┘ └───┬────────┘
           │              │            │
    ┌──────▼──────┐ ┌─────▼──────┐    │
    │ mu (256)    │ │ feat (128) │    │
    └──────┬──────┘ └─────┬──────┘    │
           └──────┬────────┘          │
                  │                   │
         ┌────────▼────────┐         │
         │  GRU + Attention│         │
         │  (Review 2 & 3) │         │
         └────────┬────────┘         │
                  │                   │
         ┌────────▼───────────────────▼─┐
         │     Final Classification      │
         │     Normal / Abnormal         │
         └──────────────────────────────┘
    """)

    st.subheader("Design Justification")
    st.markdown("""
    | Choice | Reason |
    |--------|--------|
    | ResNet18 + EfficientNet-B0 | Two different inductive biases — residual connections vs efficient scaling |
    | GRU over LSTM | Fewer parameters, better performance on shorter sequences, Review 2 confirmed |
    | VAE over plain AE | KL divergence regularizes latent space → smoother features → better classification |
    | GAN for ECG | Learns distribution of real ECG → discriminator features capture real ECG statistics |
    | Median split labels | Only available proxy for severity given no disease labels in dataset |
    | Bahdanau attention | Soft alignment over ECG windows — interpretable, shows which time segment matters |
    """)
