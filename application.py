# $ curl -XPOST -F "file=@f02.jpg" http://192.168.1.35:5001/
# http://35.206.106.108:5000/
#
# Returns:
#{"bg_white":true,"face_found":true,"height":300,"other_objects":true,"width":240,"width and height":"240x300"}

import face_recognition
from flask import Flask, jsonify, request, redirect
import cv2
import numpy as np
from PIL import Image
from numpy.core.fromnumeric import size

# You can change this to any folder on your system
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}

app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    # Check if a valid image file was uploaded
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        print(file)
        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            # The image file seems valid! Detect faces and return the result.
            return process_photo(file)

    # If no valid image file was uploaded, show the file upload form:
    return '''
    <!doctype html>
    <title>Photocheck</title>
    <h1>Cargar una foto</h1>
    <form method="POST" enctype="multipart/form-data">
      <input type="file" name="file">
      <input type="submit" value="Upload">
    </form>
    '''

def process_photo(file_stream):
    print('Processing stage...')
    face_found = False
    bg_white = False
    other_objects = False
    width = 0
    height = 0
    size = 0
 
    decoded_img = cv2.imdecode(np.frombuffer(file_stream.read(), np.uint8), -1)

    face_found = detect_faces_in_image(file_stream)
    bg_white = detect_bg(decoded_img)
    width, height, size = img_props(decoded_img)
    width_and_height = str(width)+'x'+str(height)

    result = {
        "face_found": face_found,
        "bg_white": bg_white,
        "other_objects": True,
        "width": width,
        "height": height,
        "width and height": width_and_height,
        "size": size
    }
    return jsonify(result)

def img_props(img):

    h, w, c = img.shape
    size = img.size
    print('width:  ', w)
    print('height: ', h)
    return w, h, size

def detect_bg(img):

    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    sensitivity = 15
    lower_white = np.array([0,0,255-sensitivity])
    upper_white = np.array([255,sensitivity,255])

    mask = cv2.inRange(hsv, lower_white, upper_white)
    ratio_white = cv2.countNonZero(mask)/(img.size/3)
    colorPercent = (ratio_white * (100))
    return True if colorPercent > 35.0 else False
    

def detect_faces_in_image(file_stream):
    
    # Load the uploaded image file
    img = face_recognition.load_image_file(file_stream)
    face_locations = face_recognition.face_locations(img)
    return True if len(face_locations) > 0 else False