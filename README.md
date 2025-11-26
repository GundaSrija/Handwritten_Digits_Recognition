# 📝 Handwritten Digit Recognition (MNIST Dataset)

A deep learning project that recognizes handwritten digits (0–9) using a Convolutional Neural Network (CNN).  
This project is built with **Python**, **TensorFlow/Keras**, and trained on the popular **MNIST dataset**.

---

## 📌 Project Overview

Handwritten digit recognition is a classic machine learning problem used to teach image classification and neural network concepts.  
This project builds a **CNN model** that can automatically recognize digits written by humans.

---

## 🎯 Features

- ✔️ Loads & preprocesses MNIST dataset  
- ✔️ Builds a Convolutional Neural Network  
- ✔️ Trains and evaluates the model  
- ✔️ Achieves high accuracy  
- ✔️ Predicts digits from custom images  
- ✔️ Includes visualization of training loss & accuracy  

---

## 🧠 Model Architecture

The CNN consists of:

- 2 Convolution layers  
- MaxPooling layer  
- Flatten layer  
- Dense hidden layers  
- Output layer with Softmax activation

This architecture helps the model learn spatial patterns in handwritten digits.

---

## 📂 Project Structure
Handwritten_Digits_Recognition/
│
├── dataset/ # MNIST dataset (loaded automatically)
├── models/ # Saved trained model (.h5)
├── images/ # Custom test images
├── handwritten_digit_recognition.py
├── prediction.py # For predicting custom digit images
├── requirements.txt
└── README.md

---

## 🚀 How to Run the Project

### **1️⃣ Install dependencies**

pip install -r requirements.txt

### **2️⃣ Train the model**


### **3️⃣ Predict a custom digit**
Place your image in the `images/` folder and run:

python prediction.py

---

## 📊 Output Example

- Training accuracy and loss graphs  
- Model accuracy displayed in terminal  
- Predicted digit printed with confidence  

---

## 🧪 Sample Predictions

The model can accurately classify digits from custom input images such as:

- 0 → correctly predicted as **0**  
- 7 → correctly predicted as **7**  
- 9 → correctly predicted as **9**

---

## 📈 Accuracy

Typical training results (may vary):

| Metric       | Value |
|--------------|-------|
| Training Acc | ~99%  |
| Test Acc     | ~98%  |

---

## 🛠️ Tech Stack

- **Python**
- **TensorFlow / Keras**
- **NumPy**
- **Matplotlib**
- **OpenCV** (optional for custom images)

---

## 📥 Dataset

This project uses the **MNIST digits dataset**, which contains **70,000 labeled images** of handwritten digits.  
It is automatically downloaded from Keras:


from tensorflow.keras.datasets import mnist
📌 Future Improvements

Add GUI for drawing digits

Deploy model to a web application

Improve accuracy with deeper CNN

Add training data augmentation
🤝 Contributing

Contributions are welcome!
Feel free to submit issues or pull requests.
**📄 License

This project is licensed under the MIT License.**

**💡 Author

Gunda Srija
🔗 GitHub: https://github.com/GundaSrija

🔗 Website: https://srija-gunda-xsysw4z.gamma.site/

🔗 LinkedIn: https://www.linkedin.com/in/srijagunda**
