from flask import Flask, request, jsonify
# from flask_uploads import UploadManager, configure_uploads
from werkzeug.utils import secure_filename
import face_recognition
import numpy as np
from typing import List

app = Flask(__name__)

# In-memory storage for face encodings and their labels
known_face_encodings = []
known_face_labels = []

@app.route('/', methods=['GET'])
def index():
    return jsonify({'message': 'Welcome to FaceApi'})

@app.route('/test', methods=['POST'])
def update_item():
    payload = request.get_json()
    return jsonify(payload)

@app.route('/train/', methods=['POST'])
def train_face():
    label = request.form['label']
    file = request.files['file']
    filename = secure_filename(file.filename)
    file.save(filename)
    image = face_recognition.load_image_file(filename)
    face_encodings = face_recognition.face_encodings(image)
    if len(face_encodings) == 0:
        return jsonify({'error': 'No face found in the image'}), 400
    face_encoding = face_encodings[0]
    known_face_encodings.append(face_encoding)
    known_face_labels.append(label)
    return jsonify({'message': f'Face with label {label} has been trained'})

@app.route('/recognize', methods=['POST'])
def recognize_face():
    file = request.files['file']
    filename = secure_filename(file.filename)
    file.save(filename)
    image = face_recognition.load_image_file(filename)
    face_encodings = face_recognition.face_encodings(image)
    if len(face_encodings) == 0:
        return jsonify({'message': 'No faces found in the image'})
    results = ''
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            label = known_face_labels[best_match_index]
            results=label
        else:
            results='Unknown'
    return jsonify({'results': results})

@app.route('/health', methods=['GET'])
def health_check():
    return ''

if __name__ == '__main__':
    app.run()