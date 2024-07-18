from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

app = Flask(__name__)

# Load the models
category_model = load_model('eye_disease_category_model.h5')
type_model = load_model('eye_disease_type_model.h5')
grade_model = load_model('eye_disease_grade_model.h5')

# Define class labels
category_labels = ['Normal', 'Disease']  # Replace with your actual category labels
type_labels = ['Type 0', 'Type 1', 'Type 2', 'Type 3', 'Type 4']  # Replace with your actual type labels
grade_labels = ['Grade 0', 'Grade 1', 'Grade 2', 'Grade 3', 'Grade 4']  # Replace with your actual grade labels

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        
        if file:
            image_path = 'temp_image.jpg'
            file.save(image_path)
            processed_image = preprocess_image(image_path)

            category_pred = category_model.predict(processed_image)
            type_pred = type_model.predict(processed_image)
            grade_pred = grade_model.predict(processed_image)

            category_result = category_labels[np.argmax(category_pred)]
            type_result = type_labels[np.argmax(type_pred)]
            grade_result = grade_labels[np.argmax(grade_pred)]

            return jsonify({
                'category': category_result,
                'type': type_result,
                'grade': grade_result
            })

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)