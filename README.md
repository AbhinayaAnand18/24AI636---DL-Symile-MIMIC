
# 24AI636 DL — Symile-MIMIC Multimodal Deep Learning Project

## Dataset
Symile-MIMIC: Multimodal clinical dataset (CXR + ECG + Blood Labs)
Source: PhysioNet — https://physionet.org/content/symile-mimic/1.0.0/

## Project Structure
```
├── Review1_MLP_CNN_BloodLabs.ipynb   # MLP + CNN on blood labs
├── Review2_CXR_ECG_Temporal.ipynb    # Pretrained CNN + GRU
├── Review3_VAE_GAN.ipynb             # VAE + GAN + GRU
├── app.py                             # Streamlit demo
├── requirements.txt                   # Python dependencies
└── README.md
```

## Setup
```bash
# Clone repo
git clone https://github.com/YOUR_USERNAME/24AI636-DL-Symile-MIMIC

# Install dependencies
pip install -r requirements.txt
# OR
conda env create -f environment.yml
conda activate dl_symile

# Run Streamlit demo
streamlit run app.py
```

## Results Summary

| Review | Model | Modality | Accuracy | Macro-F1 |
|--------|-------|----------|----------|----------|
| 1 | MLP | Blood Labs | ~72% | ~0.71 |
| 1 | CNN1D | Blood Labs | ~74% | ~0.73 |
| 2 | GRU | CXR + ECG | ~58% | ~0.57 |
| 2 | GRU Fine-tuned | CXR + ECG | ~61% | ~0.60 |
| 3 | VAE+GAN+GRU | CXR + ECG | ~55% | ~0.55 |

## Architecture

- **Review 1**: MLP (100→256→128→64→2) + CNN1D (2ch×50→32→64→classifier)
- **Review 2**: ResNet18(512) + EfficientNet-B0(1280) → GRU → Attention
- **Review 3**: VAE(256-dim latent) + GAN(128-dim features) → GRU → Attention

## Key Findings

1. Blood lab features give highest accuracy because labels are derived from labs
2. GRU outperforms RNN and LSTM on ECG temporal modeling
3. VAE+GAN features provide cleaner representations than raw CNN features
4. Dataset is designed for retrieval — classification accuracy ceiling is ~60-65%

## Note on Dataset

Symile-MIMIC is a multimodal retrieval dataset with no ground truth
disease labels. Labels in this project are derived from blood lab
severity (median split) as a clinically motivated proxy.

## Citation
Saporta et al. (2025). Symile-MIMIC. PhysioNet.
https://doi.org/10.13026/3vvj-s428
