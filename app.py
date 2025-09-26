from flask import Flask, request, render_template
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model

app = Flask(__name__)

model = load_model('mnist.hdf5')

def prepare_image(image):
    image = image.convert("L")
    image = ImageOps.invert(image)
    image = image.resize((28, 28))
    image = np.array(image, dtype=np.float32) / 255.0
    image = image.reshape(1, 28, 28, 1)
    return image

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="No file uploaded")

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="No file selected")

        try:
            image = Image.open(file.stream)
            image = prepare_image(image)
            prediction = model.predict(image)
            predicted_class = int(np.argmax(prediction))
            return render_template('result.html', predicted_class=predicted_class)
        except Exception as e:
            return render_template('index.html', error=f"Error: {str(e)}")

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
