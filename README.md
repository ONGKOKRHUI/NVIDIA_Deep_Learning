# 🍏 Fruit Freshness Classification — VGG16 Transfer Learning Project

This project builds an image classification model that detects **fresh vs. rotten fruits** across 6 classes:

- fresh apples  
- fresh bananas  
- fresh oranges  
- rotten apples  
- rotten bananas  
- rotten oranges  

The goal was to train and fine-tune a model using **PyTorch**, **transfer learning**, **data augmentation**, and **GPU acceleration** to reach at least **92% accuracy**.  
My final model reached **91.79%**, slightly below the assessment threshold but demonstrating strong performance and solid understanding of deep learning workflows.

---

## 📌 **Skills Learned**

### **1. Building Custom Datasets in PyTorch**
- Learned to create a custom `Dataset` class  
- Loaded images using `torchvision.io.read_image`  
- Applied preprocessing and label assignment  
- Wrapped data with `DataLoader` for mini-batch training  

### **2. Using Pretrained Models (Transfer Learning)**
- Loaded `VGG16` with `ImageNet` weights  
- Froze base layers to prevent early overfitting  
- Extracted and repurposed parts of the VGG classifier  
- Learned proper architecture modification with `nn.Sequential`

### **3. Fine-Tuning a Pretrained Network**
- Unfroze VGG16’s convolutional blocks after initial training  
- Reduced learning rate (`1e-4`) for stable fine-tuning  
- Improved validation accuracy through additional updates  

### **4. GPU-based Training and `torch.compile`**
- Detected and used CUDA when available  
- Accelerated training with `torch.compile()`  
- Learned device-aware coding patterns (`.to(device)`)

### **5. Data Augmentation & Image Preprocessing**
Used a variety of augmentations to prevent overfitting:
- Random rotations  
- Random resized crops  
- Horizontal & vertical flips  
- Color jitter (brightness/contrast/saturation/hue)  
- Normalization with ImageNet stats  
- Random erasing  

This improved model robustness and real-world generalization.

### **6. Model Training Pipeline**
Implemented training loops using helper functions:
- Forward pass  
- Backpropagation  
- Loss computation using `CrossEntropyLoss`  
- Accuracy tracking  
- Validation loop separate from training loop  

### **7. Running and Interpreting Model Assessments**
- Evaluated model performance on a separate dataset  
- Interpreted loss and accuracy metrics  
- Understood assessment thresholds and failure cases  

---

## 🧠 **Model Architecture Overview**

VGG16 (pretrained, initially frozen)
│
├── Convolutional Feature Extractor (unchanged)
├── AdaptiveAvgPool2d
├── Flatten
├── First half of VGG classifier (4096 → 4096)
├── Custom classifier:
│ ├── Linear (4096 → 500)
│ ├── ReLU
│ └── Linear (500 → 6 classes)


The model’s final layer outputs **6 class logits** for multiclass classification using **CrossEntropyLoss**.

---

## 📂 **Dataset Structure**

data/fruits/
├── train/
│ ├── freshapples/
│ ├── freshbanana/
│ ├── freshoranges/
│ ├── rottenapples/
│ ├── rottenbanana/
│ └── rottenoranges/
└── valid/
└── (same folders)


---

## 🚀 **Training Results**

| Stage | Description | Accuracy |
|-------|-------------|----------|
| Initial Transfer Learning | Base model frozen | ~0.88 |
| After Adding Custom Layers | Stable improvements | ~0.91 |
| Fine-Tuning (Unfreeze VGG) | LR=0.0001 | **0.9179** |

Final score from assessment:

Accuracy required: 0.92
Your accuracy: 0.9179
Result: Just below passing threshold


---

## 📘 **Lessons Learned**

- Even small LR changes matter when fine-tuning pretrained networks  
- Data augmentation significantly reduces overfitting  
- Freezing/unfreezing must be timed correctly  
- Custom layers should be small to avoid overfitting on limited data  
- Validation accuracy can fluctuate — patience and tuning are essential  

---

## 🔧 **Future Improvements**

- Add more data or stronger augmentations  
- Use a more modern backbone (ResNet50, EfficientNet, ConvNeXt)  
- Use LR schedulers (CosineDecay / ReduceLROnPlateau)  
- Train longer epochs after unfreezing  
- Apply mixup or cutmix for further robustness  

---

## 🏁 **Final Notes**

This assessment project demonstrates practical skills in:
- Deep learning model construction  
- Transfer learning best practices  
- Fine-tuning pretrained architectures  
- Building full training pipelines from scratch  
- Evaluating and interpreting model performance  

Although the assessment accuracy was slightly below 92%, the project successfully shows competent understanding of modern computer vision workflows.

