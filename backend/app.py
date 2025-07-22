from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import io
import tensorflow as tf
from tensorflow import keras
import os


MODEL_PATH = os.path.join('models', 'fashion_mnist_model.h5')

try:
    
    model = keras.models.load_model(MODEL_PATH)
    print(f"Model '{MODEL_PATH}' loaded successfully!")
except Exception as e:
    print(f"Error loading model from: {MODEL_PATH}")
    print("Please ensure the '.h5' file is in the correct location.")
    print(f"Error details: {e}")
    exit()


app = Flask(__name__)
CORS(app)  


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives an image file, processes it, and returns a prediction.
    """
    
    if 'file' not in request.files:
        return jsonify({'error': 'No image file sent'}), 400

    file = request.files['file']

    
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    if file:
        try:
            
            
            img = Image.open(io.BytesIO(file.read()))
            print(f"\n--- DEBUG RECEIVED IMAGE: Original: {img.mode}, {img.size} ---")

            
            
            img_grayscale = img.convert('L')
            print(f"DEBUG: After convert('L'): {img_grayscale.mode}, {img_grayscale.size}")

            
            
            img_resized = img_grayscale.resize((28, 28))
            print(f"DEBUG: After resize((28, 28)): {img_resized.size}")

            
            
            output_path = os.path.join('debug_processed_image.png')
            img_resized.save(output_path)
            print(f"DEBUG: Pre-processed image saved to {output_path}")

            
            
            img_array = np.array(img_resized)
            print(f"DEBUG: NumPy array shape: {img_array.shape}, Dtype: {img_array.dtype}")

            img_array = img_array / 255.0
            print(f"DEBUG: Normalized array min/max values: {np.min(img_array):.4f} / {np.max(img_array):.4f}")

            
            
            img_array_final = np.expand_dims(img_array, axis=0)
            img_array_final = np.expand_dims(img_array_final, axis=-1)
            print(f"DEBUG: Final array shape for model: {img_array_final.shape}")

            
            predictions = model.predict(img_array_final)
            print(f"DEBUG: Raw predictions: {predictions[0]}")

            
            predicted_class_index = np.argmax(predictions[0])
            predicted_class_name = class_names[predicted_class_index]
            confidence = float(np.max(predictions[0]))

            
            response = {
                'predicted_class': predicted_class_name,
                'confidence': confidence,
                'probabilities': predictions[0].tolist() 
            }
            return jsonify(response), 200

        except Exception as e:
            
            return jsonify({'error': f'Error processing image or making prediction: {e}'}), 500


if __name__ == '__main__':
    print("\nStarting Flask server...")
    
    app.run(debug=True, port=5000)
