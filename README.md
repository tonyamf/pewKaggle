markdown
# Pawpularity Score Prediction with Deep Learning

![Pet Image Example](https://example.com/pet-image.jpg) *Replace with actual image*

Predict the popularity score of pet images using hybrid features (handcrafted metrics + deep learning). Built with TensorFlow/Keras.

## 📌 Overview

This project predicts the **Pawpularity Score** (0-100) for pet images through:
- **Handcrafted Features**: Image sharpness, colorfulness, brightness
- **Deep Features**: EfficientNetB0/MobileNet/Vision Transformers
- **Metadata**: 12 binary flags (eyes, face, etc.)

## 🛠️ Features

### **1. Feature Engineering**
| Feature Type          | Metrics                          | Implementation File       |
|-----------------------|----------------------------------|---------------------------|
| Image Quality         | Tenengrad, Laplacian Variance    | `pawpularity.ipynb`       |
| Aesthetic Metrics     | Colorfulness, Brightness         | `image_quality.py`        |
| Deep Learning         | EfficientNetB0, ViT, MobileNet   | `models/`                 |

### **2. Model Architectures**
| Model                 | Key Components                   | Output Type               |
|-----------------------|----------------------------------|---------------------------|
| EfficientNet Hybrid    | Learnable Resizer + Metadata     | Regression (MSE Loss)     |
| Vision Transformer     | Patch Encoding + Multi-Head Attn | Classification (Custom)   |
| MobileNet Technical    | Pretrained on Technical Quality  | Feature Extraction        |

### **3. Advanced Techniques**
- **FixRes Training**: Train on 128px → Fine-tune on 224px
- **AngularGrad Optimizer**: Custom gradient adjustment
- **RandAugment**: Auto-augmentation policy

## 🚀 Installation

1. **Clone Repository**
```bash
git clone https://github.com/yourusername/pawpularity-prediction.git
cd pawpularity-prediction
Install Dependencies

bash
pip install -r requirements.txt
# Includes: tensorflow>=2.8, scikit-image, pandas, numpy
Download Data

Obtain dataset from PetFinder.my

Mount in Google Drive (Colab) or place in data/

🧠 Usage
1. Data Preparation
python
# Load metadata & images
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# Build tf.data pipeline
train_ds = build_dataset(train_df, batch_size=32, img_dir="data/train/")
2. Train Model
python
# Initialize EfficientNet hybrid
model = create_effnet_model()

# Train with AngularGrad
model.compile(
    optimizer=AngularGrad(learning_rate=1e-3),
    loss="mse",
    metrics=["mae"]
)
history = model.fit(train_ds, epochs=50)
**3. Evaluate
python
# Predict on test set
preds = model.predict(test_ds)

# Generate submission
submission = pd.DataFrame({"Id": test_df.Id, "Pawpularity": preds})
submission.to_csv("submission.csv", index=False)
📊 Results
Model	Validation MAE	Training Time (hrs)	Parameters
EfficientNetB0 (FixRes)	17.32	2.1	4.1M
Vision Transformer	18.91	4.7	12.8M
MobileNet + Metadata	19.45	1.4	2.3M
Results on PetFinder.my test set (2022)

🚧 Known Issues & Improvements
Current Limitations

Metadata converted to integer loses multi-label info

Dual-output model complicates training

ViT designed for classification, not regression

Proposed Fixes

Metadata Handling

python
# Use 12 input nodes instead of integer
metadata_input = layers.Input(shape=(12,), name="metadata")
Simplify Model

python
# Single regression output
outputs = layers.Dense(1, activation="linear")(x)
Enhance ViT

python
# Replace softmax with linear activation
logits = layers.Dense(1, activation="linear")(features)
📜 License
MIT License - See LICENSE for details.


---

**Key Files Structure**
├── models/
│ ├── effnet.py # EfficientNet hybrid model
│ ├── vit.py # Vision Transformer implementation
│ └── mobile_net.py # MobileNet technical/aesthetic
├── configs/ # Hyperparameters
├── data/ # Raw & processed data
├── utils/ # Preprocessing scripts
├── pawpularity.ipynb # Main workflow
└── requirements.txt # Dependencies


Replace `example.com/pet-image.jpg` with actual image URL. Adapt file paths as needed for your project structure
