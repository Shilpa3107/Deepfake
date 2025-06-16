from flask import Flask, request
from flask_cors import CORS 
import cv2
import numpy as np
import base64
import re
import os
import matplotlib.pyplot as plt
import time
from skimage.feature import local_binary_pattern
from numpy.fft import fft2

# Create folders if not exist
os.makedirs("frames", exist_ok=True)
os.makedirs("processed", exist_ok=True)

app = Flask(__name__)
CORS(app)

@app.route('/frame', methods=['POST'])
def receive_frame():
    data = request.get_json()

    if not data or 'image' not in data:
        return "No image data", 400
    
    img_data = re.sub('^data:image/.+;base64,', '', data['image'])
    frame_bytes = base64.b64decode(img_data)
    np_arr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Save raw frame
    timestamp = int(time.time() * 1000)
    raw_frame_path = f"frames/frame_{timestamp}.jpg"
    cv2.imwrite(raw_frame_path, frame)
    print("Frame received and saved.")

     # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Load face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for i, (x, y, w, h) in enumerate(faces):
        face_roi = gray[y:y+h, x:x+w]
            
         # --- LBP Feature Extraction ---
        radius = 1
        n_points = 8 * radius
        lbp = local_binary_pattern(face_roi, n_points, radius, method='uniform')
        lbp_norm = cv2.normalize(lbp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # --- FFT Feature Extraction ---
        fft_image = fft2(face_roi)
        fft_magnitude = np.abs(fft_image)
        fft_log = np.log(1 + fft_magnitude)
        fft_norm = cv2.normalize(fft_log, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Save results
        lbp_path = f"processed/face_{timestamp}_{i}_lbp.jpg"
        fft_path = f"processed/face_{timestamp}_{i}_fft.jpg"
        cv2.imwrite(lbp_path, lbp_norm)
        cv2.imwrite(fft_path, fft_norm)

        print(f"Processed face {i} with LBP and FFT")

    return "Frame received", 200

if __name__ == '__main__':
    app.run(debug=True)

