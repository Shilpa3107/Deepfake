from flask import Flask, request
from flask_cors import CORS 
import cv2
import numpy as np
import base64
import re
import os
import matplotlib.pyplot as plt
import time

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

    cv2.imwrite("frame.jpg", frame)
    print("Frame received and saved.")

    filename = f"frames/frame_{int(time.time()*1000)}.jpg"
    cv2.imwrite(filename, frame)
    # plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # plt.title("Received Frame")
    # plt.axis('off')
    # plt.show()

    return "Frame received", 200

if __name__ == '__main__':
    app.run(debug=True)

