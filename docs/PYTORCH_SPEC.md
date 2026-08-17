# PyTorch Implementation Specification for PhytoGATE

This document details the exact module definitions for porting **PhytoGATE** to **PyTorch / PyTorch Lightning**.

---

## 1. Dual Sigmoid Gating Module (`nn.Module`)

```python
import torch
import torch.nn as nn

class DualSigmoidGate(nn.Module):
    def __init__(self, dim_a=128, dim_b=512):
        super(DualSigmoidGate, self).__init__()
        self.gate_a = nn.Sequential(
            nn.Linear(dim_a, dim_a),
            nn.Sigmoid()
        )
        self.gate_b = nn.Sequential(
            nn.Linear(dim_b, dim_b),
            nn.Sigmoid()
        )

    def forward(self, v_a, v_b):
        g_a = self.gate_a(v_a)
        g_b = self.gate_b(v_b)
        
        f_a = v_a * g_a
        f_b = v_b * g_b
        
        return torch.cat([f_b, f_a], dim=-1) # 640-dim output
```

---

## 2. Classifier Head Module

```python
class PhytoGATEClassifierHead(nn.Module):
    def __init__(self, in_features=640, num_classes=5):
        super(PhytoGATEClassifierHead, self).__init__()
        self.fc1 = nn.Linear(in_features, 384)
        self.bn1 = nn.BatchNorm1d(384)
        self.act1 = nn.SiLU() # Swish
        self.drop1 = nn.Dropout(0.4)
        
        self.fc2 = nn.Linear(384, 192)
        self.act2 = nn.SiLU()
        self.drop2 = nn.Dropout(0.3)
        
        self.out = nn.Linear(192, num_classes)

    def forward(self, x):
        x = self.drop1(self.act1(self.bn1(self.fc1(x))))
        x = self.drop2(self.act2(self.fc2(x)))
        return self.out(x)
```
