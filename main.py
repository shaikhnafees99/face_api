from typing import Any, Annotated
from fastapi import FastAPI,Body,UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
import face_recognition_models
import face_recognition
import numpy as np
import os
from typing import List
import uvicorn
from pydantic import BaseModel


app = FastAPI()

# In-memory storage for face encodings and their labels
known_face_encodings = []
known_face_labels = []


@app.get('/')
async def index():
    return {"message":"Welcome to FaceApi"}


@app.post('/test')
async def update_item(payload: Any = Body(None)):
    return payload


@app.post("/train/")
async def train_face(face_id: str, file: UploadFile = File(...)):
    # Load the uploaded image file
    image = face_recognition.load_image_file(file.file)
    # Find face encodings
    face_encodings = face_recognition.face_encodings(image)
    
    if len(face_encodings) == 0:
        return {'face_id': 'NA'}
    
    # Assuming the first face found is the one to be used for training
    face_encoding = face_encodings[0]
    
    # Add the face encoding and its label to the known faces
    known_face_encodings.append(face_encoding)
    known_face_labels.append(face_id)
    
    return {'face_id': face_id}

@app.post("/recognize/")
async def recognize_face(file: UploadFile = File(...)):
    # Load the uploaded image file
    image = face_recognition.load_image_file(file.file)
    # Find face encodings
    face_encodings = face_recognition.face_encodings(image)
    
    if len(face_encodings) == 0:
        return {'face_id': 'NA'}
    
    results = ''
    for face_encoding in face_encodings:
        # Compare this face encoding with known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        
        # Find the best match
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            label = known_face_labels[best_match_index]
            results=label
        else:
            results='NA'
    
    return {"results": results}
@app.get("/health")
async def chk():
    return ""
# Run the application
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
