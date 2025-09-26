============================================================
# 🧠 HANDWRITTEN DIGIT RECOGNITION USING CNN AND FLASK
============================================================

This is a Flask web application that uses a trained 
Convolutional Neural Network (CNN) model to classify 
handwritten digits (0–9) from uploaded images.

------------------------------------------------------------

### 📁 PROJECT STRUCTURE
------------------------------------------------------------

app.py                → Main Flask application
mnist.hdf5            → Trained CNN model file (TensorFlow/Keras)
templates/
├── index.html        → Upload page for image input
└── result.html       → Result page to show prediction
static/               → (Optional) For CSS, JS, or image assets

------------------------------------------------------------

### 📦 DEPENDENCIES
------------------------------------------------------------

Install the following Python libraries:

- flask
- tensorflow
- numpy
- pillow

To install all at once:

> pip install flask tensorflow numpy pillow

------------------------------------------------------------

### 🚀 RUNNING THE APP (in VS Code Terminal)
------------------------------------------------------------

1. Open VS Code and this project folder.
2. Ensure your Python environment is activated.
3. Run the Flask app using:

> python app.py

4. Open your browser and go to:

> http://127.0.0.1:5000

------------------------------------------------------------

### 🔧 HOW THE MODEL WORKS
------------------------------------------------------------

- User uploads a handwritten digit image.
- The image is converted to grayscale, resized to 28x28,
  inverted, normalized, and reshaped for prediction.
- The pre-trained CNN model (mnist.hdf5) predicts the digit.

------------------------------------------------------------

