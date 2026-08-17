# PhytoGATE: Phytosanitary Gated Attention & Texture Ensemble for Automated Plant Disease Diagnosis

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow 2.15+](https://img.shields.io/badge/TensorFlow-2.15%2B-FF6F00.svg)](https://www.tensorflow.org/)
[![Paper Status](https://img.shields.io/badge/Paper-Under--Review-success.svg)](#citation)

**PhytoGATE** (**Phyto**sanitary **G**ated **A**ttention & **T**exture **E**nsemble) is a lightweight (12.3M parameters) hybrid deep learning framework designed for real-time automated plant leaf disease diagnosis. It unites domain-specific handcrafted chromatic, GLCM texture, and shape descriptors with a dual-spatial deep CNN ensemble (`EfficientNetB0` + `DenseNet121`) via independent **Sigmoid Cross-Attention Gates** ($g_A, g_B$).

---

## 🌟 Key Features & Innovations

- **Dual-Stream Topology**: Stream A extracts 104 domain-specific handcrafted descriptors (CLAHE LAB histograms, GLCM texture metrics, color moments, shape metrics) compressed via PCA (98% variance); Stream B extracts 512 deep spatial feature maps.
- **Dual Sigmoid Cross-Attention Gating**: Independent sigmoid gates ($g_A, g_B$) dynamically amplify discriminative lesion features while suppressing uninformative background noise.
- **State-of-the-Art Diagnostic Performance**:
  - **PlantVillage (Laboratory Benchmark)**: **97.01% Peak Accuracy** (96.31% ± 0.69% 3-seed mean).
  - **PlantDoc (In-the-Wild Field Benchmark)**: **83.77% ± 0.62% Mean Accuracy**, outperforming published literature field baselines (78.50%) by **+5.27%**.
  - **Multi-Crop (Foliar Benchmark)**: **99.72% ± 0.28% Mean Accuracy** (100.00% Peak).
- **Edge Deployment Ready**: Operating at **29.31 ms per image** latency with **12.3M parameters** ($7\times$ lighter than Vision Transformers ViT-Base).

---

## 📐 Architecture Topology

```text
                                      INPUT RGB IMAGE
                                    [ 224 x 224 x 3 ]
                                            │
                    ┌───────────────────────┴───────────────────────┐
                    ▼                                               ▼
     ┌─────────────────────────────┐                 ┌─────────────────────────────┐
     │          STREAM A           │                 │          STREAM B           │
     │ Domain Handcrafted Features │                 │ Dual Spatial Deep Ensemble  │
     ├─────────────────────────────┤                 ├─────────────────────────────┤
     │ 1. CLAHE LAB Normalization  │                 │ 1. Data Augmentation Layer  │
     │ 2. 80-bin Color Histograms  │                 │ 2. EfficientNetB0 Backbone  │
     │ 3. 9 Color Moments (1st-3rd)│                 │    (Channel Squeeze & Ex)  │
     │ 4. 10 GLCM Texture Metrics  │                 │ 3. DenseNet121 Backbone     │
     │ 5. 5 Contour Shape Metrics  │                 │    (Dense Feature Reuse)    │
     │ 6. StandardScaler + PCA(98%)│                 │ 4. Global Avg Pooling (GAP) │
     │ 7. Dense Projection (128-d) │                 │ 5. Ensemble Concat (512-d)  │
     └──────────────┬──────────────┘                 └──────────────┬──────────────┘
                    │ v_A in R^128                                  │ v_B in R^512
                    ▼                                               ▼
     ┌─────────────────────────────┐                 ┌─────────────────────────────┐
     │  Sigmoid Attention Gate A   │                 │  Sigmoid Attention Gate B   │
     │  g_A = σ( W_A · v_A + b_A ) │                 │  g_B = σ( W_B · v_B + b_B ) │
     └──────────────┬──────────────┘                 └──────────────┬──────────────┘
                    │ Hadamard Product (⊙)                          │ Hadamard Product (⊙)
                    ▼                                               ▼
     ┌─────────────────────────────┐                 ┌─────────────────────────────┐
     │ Gated Stream A: f_A in R^128│                 │ Gated Stream B: f_B in R^512│
     └──────────────┬──────────────┘                 └──────────────┬──────────────┘
                    └───────────────────────┬───────────────────────┘
                                            ▼
                             ┌─────────────────────────────┐
                             │ Joint Fused Vector (640-d)  │
                             └──────────────┬──────────────┘
                                            ▼
                             ┌─────────────────────────────┐
                             │ Classification Dense Head   │
                             │ Dense(384) -> Swish -> D0.4 │
                             │ Dense(192) -> Swish -> D0.3 │
                             │ Softmax Output Layer        │
                             └──────────────┬──────────────┘
```

---

## 📊 Benchmark Performance

### 1. Three-Dataset Triangulation Summary (3-Seed Zero-Leakage Evaluation)

| Dataset | Scope | Model Architecture | Peak Test Acc | 3-Seed Mean Acc | Macro F1-Score | Inference Latency |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **PlantVillage** | Solanaceae (Tomato/Potato) | **PhytoGATE (Proposed)** | **97.01%** | **96.31% ± 0.69%** | **0.9632** | **29.31 ms** |
| **PlantDoc** | In-the-Wild Outdoor Field | **PhytoGATE Dual-Stream**| **85.53%** | **83.77% ± 0.62%** | **0.6972** | **88.74 ms** |
| **Multi-Crop** | Corn, Grape, Peach | **PhytoGATE (Proposed)** | **100.00%** | **99.72% ± 0.28%** | **0.9972** | **29.44 ms** |

---

## 📁 Repository Structure

```text
PhytoGATE/
├── README.md                           # Project documentation & benchmark overview
├── LICENSE                             # MIT License
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Git ignore rules
├── src/                                # Core architecture module
│   ├── __init__.py
│   ├── phytogate.py                    # PhytoGATE network definition
│   └── utils.py                        # Feature extraction (CLAHE, GLCM, PCA) & evaluation
├── scripts/                            # Benchmark execution scripts
│   ├── train_plantvillage.py           # PlantVillage benchmark runner
│   ├── train_plantdoc.py               # PlantDoc field benchmark runner
│   └── train_multicrop.py              # Multi-Crop benchmark runner
├── docs/                               # PDF Reports & Learning Guides
│   ├── PhytoGATE_Benchmark_Report.pdf  # Technical benchmark report PDF
│   └── PhytoGATE_Master_Learning_Guide.pdf # Master technical learning guide PDF
└── paper/                              # Academic journal manuscript
    └── plant_leaf_disease_research_paper.md
```

---

## 🚀 Quickstart Guide

### 1. Installation

```bash
git clone https://github.com/totaliyahtrash/PhytoGATE.git
cd PhytoGATE
pip install -r requirements.txt
```

### 2. Run PlantVillage Benchmark

```bash
python scripts/train_plantvillage.py
```

### 3. Run PlantDoc Field Benchmark

```bash
python scripts/train_plantdoc.py
```

---

## 📖 PDF Reports & Documentation

- 📄 **[Technical Benchmark Report (PDF)](docs/PhytoGATE_Benchmark_Report.pdf)**
- 📄 **[Master Technical Learning Guide (PDF)](docs/PhytoGATE_Master_Learning_Guide.pdf)**
- 📄 **[Research Paper Manuscript (Markdown)](paper/plant_leaf_disease_research_paper.md)**

---

## 📝 Citation

If you use **PhytoGATE** or its benchmark suites in your research, please cite:

```bibtex
@article{phytogate2026,
  title={PhytoGATE: Phytosanitary Gated Attention & Texture Ensemble for Automated Plant Disease Diagnosis},
  author={Machine Learning and Computer Vision Research Team},
  journal={Computers and Electronics in Agriculture},
  year={2026},
  publisher={Elsevier}
}
```

---

## 📜 License

Distributed under the [MIT License](LICENSE).
