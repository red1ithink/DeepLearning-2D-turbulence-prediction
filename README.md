# convlstm-2d-turbulence-prediction

# ConvLSTM-based 2-D Turbulence Prediction  
**Autoregressive rollout with optional Scheduled Sampling**

<p align="center">
  <img src="docs/banner.png" alt="turbulence banner" width="600">
</p>

## 1. Overview
This repository contains three TensorFlow 2.x training scripts for predicting future vorticity fields in 2-D decaying turbulence:

| Script |  ε-schedule | Start ε (epoch 10) | Comment |
| ------ | ---------- | ------------------ | ------- |
| `train_baseline.py` | None | — | Teacher-forcing only |
| `train_sched09.py` | exp decay | 0.90 | Gentle sampling |
| `train_sched08.py` | exp decay | 0.80 | Aggressive sampling |

All models share the same lightweight **ConvLSTM → BN → Conv** backbone and roll out **5 frames** autoregressively from **3 input frames**.

---

## 2. Directory Layout
