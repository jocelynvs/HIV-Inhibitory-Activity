# HIV Activity Prediction Using Graph Neural Networks

This project predicts whether a molecule is **active against HIV** from its SMILES structure using Graph Neural Networks.

Three models are compared:

* GCN
* GAT
* GCN + GAT

Because the dataset is highly imbalanced, the models are trained using **class-weighted binary cross-entropy** instead of oversampling or undersampling.

---

## Project Purpose

The goal of this project is to:

* convert SMILES molecules into graph representations,
* generate atom-level features using Morgan fingerprints,
* train and compare GCN, GAT, and GCN-GAT models,
* handle class imbalance using class weights,
* evaluate performance using metrics suitable for imbalanced classification.

Main evaluation metrics:

* ROC-AUC
* PR-AUC
* Precision
* Recall
* F1-score

---

## Repository Structure

```text
.
├── notebook.ipynb                   # Data preprocessing, graph generation, training, and evaluation
├── dataset_hiv.csv                  # HIV dataset containing SMILES and activity labels
├── requirements.txt                 # Python dependencies
├── README.md                        # Project documentation
└── model/                           # Saved models and best validation checkpoints
    ├── gcn_model.keras              # Full trained GCN model
    ├── gcn_weights.weights.h5       # Best GCN weights
    ├── gat_model.keras              # Full trained GAT model
    ├── gat_weights.weights.h5       # Best GAT weights
    ├── gcn_gat_model.keras          # Full trained GCN + GAT model
    └── gcn_gat_weights.weights.h5   # Best GCN + GAT weights
```

---

## Dataset

The dataset contains:

* `smiles`: molecular structure
* `activity`: original activity label
* `HIV_active`: binary target

Target:

```text
0 = inactive
1 = active
```

The original dataset contains **41,127 molecules**. Molecules with more than **60 atoms** are removed due to limitation of computation resource.

After filtering:

```text
Inactive : 39,065
Active   : 1,319
Total    : 40,384
```

The dataset is therefore highly imbalanced.

---

## Data Split

Stratified splitting is used to preserve the class distribution.

```text
Training   : 25,845
Validation : 6,462
Test       : 8,077
```

The test set contains:

```text
Inactive : 7,813
Active   : 264
```

---

## Molecular Graph Representation

Each molecule is represented as a graph:

```text
Atoms  → nodes
Bonds  → edges
```

Three inputs are generated:

```text
X = atom feature matrix
A = adjacency matrix
M = atom mask
```

Graphs are padded to a maximum of **60 atoms**.

---

## Morgan Fingerprint Atom Features

The model does not use simple atom properties directly. Instead, each atom is represented using an **atom-centered Morgan fingerprint**, generated with RDKit.

The executed code uses:

```python
MORGAN_RADIUS = 1
MORGAN_NBITS = 256
```

The Morgan fingerprints describe the local chemical environment surrounding an atom. With `radius = 1`, the fingerprint captures structural information around the atom up to one bond away.

Each atom is represented by a:

```text
256-dimensional binary vector
```

Conceptually:

```text
SMILES
  │
  ▼
RDKit molecule
  │
  ▼
Morgan fingerprint generation
  │
  ▼
Fingerprint bits assigned to atom environments
  │
  ▼
Atom 1 → 256 features
Atom 2 → 256 features
Atom 3 → 256 features
...
```

These atom-level fingerprints become the node features used by the GNN models.

---

## Handling Class Imbalance

No oversampling or undersampling is used. Class weights calculated from the training data are:

```text
Class 0 : 0.5169
Class 1 : 15.3110
```

This increases the loss contribution of incorrectly classified HIV-active compounds.

---

## Workflow

```text
dataset_hiv.csv
        │
        ▼
Load SMILES and labels
        │
        ▼
Remove molecules with >60 atoms
        │
        ▼
Train / validation / test split
        │
        ▼
Convert SMILES into molecular graphs
        │
        ├── Morgan atom features
        ├── adjacency matrices
        └── atom masks
        │
        ▼
Calculate class weights
        │
        ▼
Train GCN / GAT / GCN+GAT
        │
        ▼
Evaluate on test set
        │
        ▼
Save trained models
```

---

## Models

### GCN

```text
GCN 64
   ↓
GCN 128
   ↓
GCN 256
   ↓
GCN 512
   ↓
Masked Global Average Pooling
   ↓
Dense 128
   ↓
Sigmoid
```

### GAT

The GAT model uses three attention layers with `64 channels` and `3 attention heads`:

```text
GAT
 ↓
GAT
 ↓
GAT
 ↓
Masked Global Average Pooling
 ↓
Dense 128
 ↓
Sigmoid
```

### GCN + GAT

The combined model processes the molecule using both branches:

```text
          Atom Features
          /           \
       GCN             GAT
        │               │
      Pooling         Pooling
          \           /
           Concatenate
               │
           Dense 128
               │
            Sigmoid
```

---

## Training

All models use:

```text
Optimizer     : Adam
Learning rate : 0.001
Loss          : Binary cross-entropy
Batch size    : 32
Maximum epochs: 50
```

Early stopping monitors:

```text
validation PR-AUC
```

The best model weights are saved in the `model/` directory.

---

## Results

| Model     |    ROC-AUC |     PR-AUC |  Precision + |   Recall + |       F1 + |
| --------- | ---------: | ---------: | -----------: | ---------: | ---------: |
| GCN       |     0.7887 |     0.2991 |       0.1465 | **0.6023** |     0.2357 |
| GAT       | **0.7966** |     0.2980 |       0.1442 |     0.5303 |     0.2267 |
| GCN + GAT |     0.7828 | **0.3285** |   **0.1648** |     0.5530 | **0.2539** |

### Main Findings

* **GCN** produced the highest recall.
* **GAT** produced the highest ROC-AUC.
* **GCN + GAT** produced the highest PR-AUC, precision, and F1-score.

However, positive-class precision remains low for all models, many compounds predicted as HIV-active are still false positives.

---

## Conclusion

The three models show some ability to separate HIV-active and inactive compounds, but the results are still limited. The main observations are:

* GCN has the highest recall at **0.6023**, but low precision.
* GAT has the highest ROC-AUC at **0.7966**.
* GCN + GAT has the highest PR-AUC at **0.3285** and F1-score at **0.2539**.

Although the combined model performs best on several minority-class metrics, its positive-class precision is still only **0.1648**. This means that a large proportion of predicted active compounds are false positives.

Therefore, the current models are best viewed as a **proof-of-concept for HIV-active compound screening**, with further optimization needed before practical application.

Future work could explore improved molecular features, model architectures, decision thresholds, and imbalance-handling strategies.

---

## Installation

To run the code, firstly install the dependencies:

```bash
pip install -r requirements.txt
```

Then open the notebook:

```bash
jupyter notebook notebook.ipynb
```

Keep the dataset in the same directory as the notebook and the trained models inside:

```text
model/
```
