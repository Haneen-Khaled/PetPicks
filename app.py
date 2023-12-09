from flask import  Flask, render_template, request, jsonify
from PIL import Image 
import numpy as np
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model('tensorflow-model.h5')
target_size =(224, 224)
def preprocess_image(input_photo):
    new_photo = Image.open(input_photo)
    new_photo = new_photo.resize(target_size)
    photo_array = np.array(new_photo) / 255.0
    photo_array = np.expand_dims(photo_array, axis=0)    
    return photo_array  
@app.route('/')
def home():
    return render_template('index.html')







@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'no file'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': "file name is empty"})
    
    
    
    
    
    
    
    try:
        photo_array = preprocess_image(file)
        numpy_number = model.predict(photo_array)
        class_index = np.argmax(numpy_number[0])
        if class_index == 0:
            result = 'Cat'
        else:
            result = 'Dog' 
        return jsonify({'class_name': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    
    
    
    
    
    
    
    
    app.run(debug=True)
    