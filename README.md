# 🍽️ Food101 Image Classification — ConvNeXt

A deep learning project that classifies food images into **101 categories** using ConvNeXt Base, PyTorch, and transfer learning.

## 🌐 Live Demo

**[Visit the Website](http://13.62.105.156:8000/docs)**

Upload a food image to get the model's predicted food category.

## 📊 Dataset

- **Dataset:** [Food101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)
- **Categories:** 101 food classes
- **Task:** Multi-class image classification

## 🧠 Model & Methodology

The project uses **ConvNeXt Base**, fine-tuned on Food101 to adapt its pretrained visual features to food recognition.

Key techniques include:
- **Transfer Learning:** Fine-tuning a pretrained model for food classification.
- **Data Augmentation:** Random resized cropping, horizontal flipping, rotation, and color jitter.
- **Optimization:** AdamW with different learning rates across model layers.
- **Learning Rate Scheduling:** ReduceLROnPlateau to reduce learning rates when validation loss stops improving.
- **Loss Function:** Cross-Entropy Loss.
- **Experiment Tracking:** MLflow for recording training configurations and results.

## 📈 Test Results

| Metric | Score |
|---|---:|
| Accuracy | 87.45% |
| Macro Precision | 87.68% |
| Macro Recall | 87.45% |
| Macro F1-Score | 87.44% |

These metrics evaluate the model's classification performance on the test set. Macro-averaged metrics give equal weight to each class.

## 🛠️ Technologies

- Python
- PyTorch and Torchvision
- TorchMetrics
- MLflow
- Docker
- AWS EC2

## 🔮 Future Improvements

- Analyze misclassified images using a confusion matrix.
- Compare additional model architectures.
- Improve inference speed and model generalization.

## 👨‍💻 Author

**JAWAD**

