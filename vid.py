#This script will be used to swap faces in a existing video

# Import necessary libraries
import insightface
import cv2
from insightface.app import FaceAnalysis
import numpy as np

# Initialize the FaceAnalysis model
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 480))  # Set up the model for face detection

# Load the source image ("source.jpg") and extract the face
sourcep = cv2.imread("source.jpg")
source = np.array(sourcep, dtype=np.uint8)
sfaces = app.get(source)
sface = sfaces[0]  # Assuming there's only one face which will replace tall the faces in the live feed

# Initialize video capture from the existing video in the same folder
cap = cv2.VideoCapture("video.mp4")
ret = True

#Initializing VideoWriter function
output = cv2.VideoWriter( "output.mp4", cv2.VideoWriter_fourcc(*'MP4V'), 30, (1920, 1080))

while ret:
    # Read a frame from the camera
    ret, targetp = cap.read()
    if ret==False:
        output.release()
        print("Video Writing complete")
        break
    target = np.array(targetp, dtype=np.uint8)

    # Detect faces in the target frame
    faces = app.get(target)

    # Load the face-swapping model ("inswapper_128.onnx") here I hve downloaded the model ,and will share the file
    swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=False, download_zip=False)

    # Create a copy of the target frame for manipulation
    res = target.copy()

    # Perform face swapping for each detected face
    for face in faces:
        res = swapper.get(res, face, sface, paste_back=True)

    output.write(res)

