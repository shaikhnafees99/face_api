Face Recognition API (FASTAPI / FLASK) Python
This is a simple Face Recognition API built using Flask. It provides two main endpoints: one for training the model with labeled images, and another for recognizing faces from images.

Features
Train the Model: Add labeled face images to train the recognition model.
Recognize Faces: Identify faces from input images using the trained model.
Endpoints
1. Train the Model
URL: http://localhost:5000/train
Method: POST
Parameters:

label (string): The label for the face (e.g., the person's name).
image (file): The image file containing the face.
Description: This endpoint takes a label and an image as input. It uses the image to train the face recognition model under the given label.

curl -X POST -F 'label=John Doe' -F 'image=@/path/to/image.jpg' http://localhost:5000/train


2. Recognize Faces
URL: http://localhost:5000/recognize
Method: POST
Parameters:

image (file): The image file containing the face to recognize.
Description: This endpoint takes an image as input and returns the label of the recognized face, if the face is in the trained model.

curl -X POST -F 'image=@/path/to/image.jpg' http://localhost:5000/recognize
