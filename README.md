# ResNetInception-CNN-Classifier-For-TinyImageNetDataset
CNN-based image classification using a ResNet-Inception hybrid model on the TinyImageNet dataset, including extensive hyperparameter tuning and performance comparison
## 📦 Part 1: Dataset Preprocessing and Custom Dataloader

This section focuses on preparing the **TinyImageNet-200** dataset for training a CNN classifier. It includes downloading the dataset from Kaggle, parsing and organizing the files, converting them into a standard structure for PyTorch `ImageFolder`, and  performing initial visualization for  verification.

---

### 📁 Dataset Organization

After running the preprocessing script, the dataset is reorganized into a custom folder structure:
<pre> <code>custom_split/
├── train/
│   └── class_name/
│       └── images...
├── validation/
│   └── class_name/
│       └── images...
├── test/
│   └── unknown/
        └── images...</code> </pre> 


This format allows efficient loading using `torchvision.datasets.ImageFolder`.

---

### 📌 Processing Steps

- ✅ Download dataset using `kagglehub`
- ✅ Parse `words.txt` and map WordNet IDs (WNIDs) to human-readable labels
- ✅ Rebuild `val/` directory by splitting images by class using `val_annotations.txt`
- ✅ Group and copy all images into a new directory structure at `/content/custom_split`
- ✅ Display random sample images for visual verification of each split

---

### 📊 Dataset Summary

| Split         | Number of Images |
|---------------|------------------|
| **Train**     | 100,000           |
| **Validation**| 10,000            |
| **Test**      | 10,000            |
| **Total**     | 120,000           |



### 🎯 Purpose of This Step

This structured preprocessing step ensures that:

- The data is properly class-separated and labeled
- The format supports `ImageFolder`-based loading
- The dataset is compatible with data augmentation and dataloader batching
- Human-readable class names are available for visualization and analysis

---

<p align="center">
  <img src="https://github.com/user-attachments/assets/a693cd28-f912-49db-97e8-452a8c42ad6d" alt="ResNet-Inception Architecture" width="600"/>
</p>

<p align="center"><b>Figure:</b> ResNet-Inception Hybrid Architecture</p>


## 📊 Part 2: Learning Rate Tuning

In this stage, we aimed to find the optimal **learning rate (LR)** for training our custom **ResNet-Inception CNN** on the **TinyImageNet** dataset. We performed experiments under three conditions:

### ⚙️ Experimental Setup

- **Model**: `ResIncepCNNBase`
- **Optimizer**: SGD with momentum = 0.9
- **Loss**: CrossEntropyLoss
- **Batch size**: 64
- **Epochs**: 30
- **Hardware**: GPU (CUDA enabled if available)

### 🧪 Experiments

| Experiment | Learning Rate | Scheduler | Description |
|-----------|---------------|-----------|-------------|
| 1 | `0.01` | ❌ None | Baseline training |
| 2 | `0.001` | ❌ None | Lower learning rate |
| 3 | `0.01` | ✅ StepLR (step=5, gamma=0.1) | Learning rate decay applied |

---

### 📈 Results

#### 1: LR = 0.01 (No Scheduler)
![bi0 01](https://github.com/user-attachments/assets/1afc3b06-739f-4c29-b5d6-c6bdd6d8ee92)



- **Train Accuracy** improves steadily.
- **Validation Accuracy** plateaus early.
- Indication of **possible overfitting**.

---

#### 2: LR = 0.001 (No Scheduler)
![bi0 001](https://github.com/user-attachments/assets/993cf73b-ee20-4a66-a8bc-f9e766a2f830)



- Training is **very slow** due to low LR.
- Performance is **significantly worse** than other setups.
- Model struggles to converge.

---

#### 3: LR = 0.01 with StepLR Scheduler
![sch0 01](https://github.com/user-attachments/assets/c810b0db-d69e-4c0c-87ca-ee1d3066cb6d)



- **Best performance** overall.
- LR decay leads to **smoother convergence**.
- Validation accuracy improves with generalization.

---

### ✅ Conclusion

- The best result was obtained with a **learning rate of 0.01 using a StepLR scheduler**.
- Using a scheduler effectively controls the training dynamics and **prevents overfitting**.
- This setup will be used as the baseline for the next steps in model tuning.

# 🧪 Part 3: Investigating the Effect of Dropout and Batch Normalization

In this part, we explore the impact of incorporating **Dropout** and **Batch Normalization** into the ResIncepCNN model.

---

## 🔍 Experiment Overview

- **Model:** `ResIncepCNNWithBNDropout`
- **Dropout Probability:** 0.5
- **Batch Normalization:** Applied after the first fully connected layer
- **Learning Rate:** 0.01
- **Weight Decay:** 0.0
- **Scheduler:** StepLR (step size = 5, gamma = 0.1)
- **Epochs:** 30
- **Optimizer:** SGD with momentum = 0.9

---

## 🧠 Architecture Used

This model is an extension of the original `ResIncepCNNBase` by:
- Integrating **BatchNorm1d** after the first dense layer (`fc1`)
- Applying **Dropout (p=0.5)** before the final classification layer (`fc2`)

```python
self.bnFc1 = nn.BatchNorm1d(512)
self.dropout = nn.Dropout(p=0.5)
```

---

## 📈 Training and Validation Metrics

The following figure shows the **loss**, **Top-1 accuracy**, and **Top-5 accuracy** across the 30 training epochs:
![BN DO](https://github.com/user-attachments/assets/90aaa3a9-bfcd-4c5c-a9a1-a1ac3d75ed8b)

---

## ✅ Observations

- **Generalization improved**: Dropout helps prevent overfitting by randomly deactivating neurons during training.
- **Stabilized training**: BatchNorm improves convergence speed and reduces internal covariate shift.
- **Higher validation accuracy**: Compared to previous settings without regularization, the model achieved better validation Top-1 and Top-5 accuracies.

---

## 💡 Conclusion

Applying **Dropout** and **Batch Normalization** to our hybrid ResIncepCNN architecture effectively enhances model performance and generalization on the validation set. This setup can be a strong candidate for further experiments or deployment.

# 📦 Part 4: Integrating Early Stopping with ResNet-Inception and Dropout + Batch Normalization

## 🧠 Objective
This stage enhances the model training by introducing **Early Stopping** to the best architecture so far (a combined ResNet-Inception CNN enhanced with **Dropout** and **Batch Normalization**) to prevent overfitting and reduce unnecessary training epochs.

---

## 🛠 Model Architecture
The model used in this stage is `ResIncepCNNWithBNDropout_All`, which includes:

- 📚 **Residual Connections** (ResNet blocks) for better gradient flow
- 🔍 **Multi-scale feature extraction** via Inception modules
- 🧪 **Batch Normalization** for stabilizing the learning process
- 💧 **Dropout** (p=0.5) for regularization
- 🛑 **Early Stopping** triggered after 5 consecutive epochs with no improvement in validation loss

---

## ⚙️ Training Configuration

| Parameter              | Value                   |
|------------------------|-------------------------|
| Optimizer              | SGD                     |
| Learning Rate          | 0.01                    |
| Momentum               | 0.9                     |
| Weight Decay           | 0.0                     |
| LR Scheduler           | StepLR (γ=0.1, step=5)  |
| Epochs (max)           | 30                      |
| Early Stopping Patience| 5 epochs                |
| Batch Size             | 64                      |
| Input Size             | 64 × 64                 |

---

## 📊 Results

The training process was monitored with early stopping criteria. The performance plots for:

- **Loss** (Train vs Validation)
- **Top-1 Accuracy**
- **Top-5 Accuracy)

were visualized to track progress and convergence.

> 📌 *Training automatically stopped early once the model failed to improve for 5 consecutive validation epochs.*

---

## 🔁 Early Stopping Logic
Early stopping was implemented inside the  `trainModel()` function. The model stops training once validation loss fails to improve for **5 consecutive epochs**, ensuring better generalization and reduced overfitting.

---

## 🧾 Key Takeaways

- The model benefits from better generalization and reduced training time.
- Early Stopping avoided overfitting by monitoring validation performance.
- Final accuracy remained competitive with fewer epochs.

---
### 📈 Results
The following figure shows the **loss**, **Top-1 accuracy**, and **Top-5 accuracy** with EarlyStopping:
![image](https://github.com/user-attachments/assets/bdac1c0d-b35f-4913-b024-8d44a2bf3d5f)



