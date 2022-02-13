# $ curl -XPOST -F "file=@f02.jpg" http://192.168.1.35:5001/
#http://192.168.1.35:5001/
# Returns:
#
# {
#  "face_found_in_image": true,
#  "is_picture_of_obama": true
# }
#
# This example is based on the Flask file upload example: http://flask.pocoo.org/docs/0.12/patterns/fileuploads/

# NOTE: This example requires flask to be installed! You can install it with pip:
# $ pip3 install flask

from dis import dis
import face_recognition
from flask import Flask, jsonify, request, redirect, render_template
import cv2
import numpy as np
from PIL import Image
from numpy.core.fromnumeric import size
import matplotlib.pyplot as plt
from PIL import Image
import statistics
import dlib
import math
from mouth_open_algorithm import get_lip_height, get_mouth_height


#OpenCV and Yolo Configurations **start here**
#Write down conf, nms thresholds,inp width/height
confThreshold = 0.25
nmsThreshold = 0.40
inpWidth = 416
inpHeight = 416

#Load names of classes and turn that into a list
classesFile = "coco.names"
classes = None

with open(classesFile,'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

#Model configuration
modelConf = 'yolov3.cfg'
modelWeights = 'yolov3.weights'

net = cv2.dnn.readNetFromDarknet(modelConf, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

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
    objects = []
    glasses_on = False
    open_mouth = False
 
    decoded_img = cv2.imdecode(np.frombuffer(file_stream.read(), np.uint8), -1)
    face_found = detect_faces_in_image(file_stream)
    bg_white = detect_bg(decoded_img)
    width, height, size = img_props(decoded_img)
    width_and_height = str(width)+'x'+str(height)
    objects = detect_obj(decoded_img)
    glasses_on = detect_glasses(file_stream)
    mouth_open = detect_mouth(file_stream)
    face_centered = detect_face_position(file_stream)

    result = {
        "face_found": face_found,
        "bg_white": bg_white,
        "other_objects": objects,
        "width": width,
        "height": height,
        "width and height": width_and_height,
        "size" : size,
        "glasses_on" : glasses_on,
        "mouth_open": mouth_open,
        "face_centered" : face_centered
     }
    return jsonify(result)

def detect_face_position(file_stream):
    
    # Load the uploaded image file
    img = face_recognition.load_image_file(file_stream)
    
    face_locations = face_recognition.face_locations(img)
    face_encodings = face_recognition.face_encodings(img, face_locations)
    face_landmarks = face_recognition.face_landmarks(img)
    
    h, w, c = img.shape

    x2, y2 = [h/2,w/2]
        
    x1, y1 = face_landmarks[0]['nose_bridge'][2]
    dist = math.hypot(x2 - x1, y2 - y1)
    print(dist)


    return True if dist < 40 else False

def detect_mouth(file_stream):

    # Load the uploaded image file
    img = face_recognition.load_image_file(file_stream)
    
    face_locations = face_recognition.face_locations(img)
    face_encodings = face_recognition.face_encodings(img, face_locations)
    face_landmarks = face_recognition.face_landmarks(img)

    top_lip = face_landmarks[0]['top_lip']
    bottom_lip = face_landmarks[0]['bottom_lip']

    top_lip_height = get_lip_height(top_lip)
    bottom_lip_height = get_lip_height(bottom_lip)
    mouth_height = get_mouth_height(top_lip, bottom_lip)
    
    # if mouth is open more than lip height * ratio, return true.
    ratio = 0.5
    print('top_lip_height: %.2f, bottom_lip_height: %.2f, mouth_height: %.2f, min*ratio: %.2f' 
          % (top_lip_height,bottom_lip_height,mouth_height, min(top_lip_height, bottom_lip_height) * ratio))
          
    if mouth_height > min(top_lip_height, bottom_lip_height) * ratio:
        print("Boca Abierta")
        return True
    else:
        print("Boca Cerrada")
        return False


def img_props(img):

    h, w, c = img.shape
    size = img.size
    print('width:  ', w)
    print('height: ', h)
    return w, h, size

def detect_bg(img):

    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    sensitivity = 25
    lower_white = np.array([0,0,255-sensitivity])
    upper_white = np.array([255,sensitivity,255])

    mask = cv2.inRange(hsv, lower_white, upper_white)

    ratio_white = cv2.countNonZero(mask)/(img.size/3)

    colorPercent = (ratio_white * (100))

    print(colorPercent)
    return True if colorPercent > 35.0 else False
    

def detect_faces_in_image(file_stream):
    
    # Load the uploaded image file
    img = face_recognition.load_image_file(file_stream)
    img2 = Image.open(file_stream)
    face_locations = face_recognition.face_locations(img)
    
    print("I found {} face(s) in this photograph.".format(len(face_locations)))
    return True if len(face_locations) > 0 else False

def detect_glasses(file_stream):
    # Load the uploaded image file
    img = face_recognition.load_image_file(file_stream)
    img2 = Image.open(file_stream)
    face_locations = face_recognition.face_locations(img)

    # test code
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    
    rect = detector(img)[0]
    sp = predictor(img, rect)
    landmarks = np.array([[p.x, p.y] for p in sp.parts()])

    nose_bridge_x = []
    nose_bridge_y = []
    for i in [28,29,30,31,33,34,35]:
            nose_bridge_x.append(landmarks[i][0])
            nose_bridge_y.append(landmarks[i][1])
            
            
    ### x_min and x_max
    x_min = min(nose_bridge_x)
    x_max = max(nose_bridge_x)
    ### ymin (from top eyebrow coordinate),  ymax
    y_min = landmarks[20][1]
    y_max = landmarks[31][1]
    
    img2 = img2.crop((x_min,y_min - 10,x_max,y_max -10))
    
    img_blur = cv2.GaussianBlur(np.array(img2),(3,3), sigmaX=0, sigmaY=0)
    
    edges = cv2.Canny(image =img_blur, threshold1=100, threshold2=200)

    edges_center = edges.T[(int(len(edges.T)/2))]

    return True if 255 in edges_center else False


def postprocess(img, outs):
    imgHeight = img.shape[0]
    imgWidth = img.shape[1]
    obj = []
    classIDs = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            
            scores = detection [5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > confThreshold:
                centerX = int(detection[0] * imgWidth)
                centerY = int(detection[1] * imgHeight)

                width = int(detection[2]* imgWidth)
                height = int(detection[3]*imgHeight )

                left = int(centerX - width/2)
                top = int(centerY - height/2)

                classIDs.append(classID)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    indices = cv2.dnn.NMSBoxes (boxes,confidences, confThreshold, nmsThreshold )
    #print(classIDs)
    
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]

        label = '%.2f' % confidence

        if classes:
            assert (classIDs[i] < len(classes))
            obj.append(classes[classIDs[i]])
            
    #print(obj)
    return obj
        #drawPred(classIDs[i], confidences[i], left, top, left + width, top + height)

def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
   
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def detect_obj(img):
    blob = cv2.dnn.blobFromImage(img, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop = False)

    #Set the input the the net
    net.setInput(blob)
    outs = net.forward (getOutputsNames(net))
   
    return postprocess (img, outs)