📘 Handwritten Digit Recognition (MNIST Dataset)

A deep-learning project that recognizes handwritten digits (0–9) using a Convolutional Neural Network (CNN).
Built with Python, TensorFlow/Keras, and trained on the MNIST dataset.

🎯 Features

✔️ Loads & preprocesses MNIST dataset

✔️ Builds a Convolutional Neural Network

✔️ Trains and evaluates the model

✔️ Predicts digits from custom images

✔️ Visualizes training accuracy and loss

✔️ Achieves ~98–99% accuracy

🧠 Model Architecture

The CNN contains:

2 Convolution Layers

MaxPooling Layer

Flatten Layer

Dense Hidden Layers

Softmax Output Layer

This structure extracts spatial features from handwritten digits efficiently.

📂 Project Structure
Handwritten_Digits_Recognition/
│
├── dataset/                          # MNIST dataset (downloaded automatically)
├── models/                           # Saved trained model (.h5)
├── images/                           # Custom test images
│
├── handwritten_digit_recognition.py  # Main training script
├── prediction.py                     # Script for predicting digit images
├── requirements.txt                  
└── README.md

🚀 How to Run the Project
1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Train the Model
python handwritten_digit_recognition.py

3️⃣ Predict a Custom Digit

Place any digit image (28x28 or larger) into /images/ and run:

python prediction.py

📊 Example Output

Training Accuracy/Loss plots

Final test accuracy

Model predictions printed in terminal

Example prediction:

Predicted Digit: 7
Confidence: 98.32%

📈 Accuracy
Metric	Value
Training Acc	~99%
Test Acc	~98%
🛠️ Technologies Used

Python

TensorFlow / Keras

NumPy

Matplotlib

OpenCV (for image loading)

📥 Dataset Info

This project uses the MNIST dataset, which includes 70,000 handwritten digit images (28×28 grayscale).

Loaded directly using:

from tensorflow.keras.datasets import mnist

🔮 Future Enhancements

Add a GUI for drawing digits

Deploy using Flask / FastAPI

Improve performance with deeper CNN

Add data augmentation

🤝 Contributing

Contributions, issues, and pull requests are welcome!

📄 License

This project is licensed under the MIT License.

👩‍💻 Author

Gunda Srija
🔗 GitHub: https://github.com/GundaSrija

🔗 Website: https://srija-gunda-xsysw4z.gamma.site

🔗 LinkedIn: https://www.linkedin.com/in/srijagunda
