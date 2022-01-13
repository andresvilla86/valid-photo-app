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

import face_recognition
from flask import Flask, jsonify, request, redirect
import cv2
import numpy as np
from PIL import Image
from numpy.core.fromnumeric import size


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
 
    decoded_img = cv2.imdecode(np.frombuffer(file_stream.read(), np.uint8), -1)
    face_found = detect_faces_in_image(file_stream)
    bg_white = detect_bg(decoded_img)
    width, height, size = img_props(decoded_img)
    width_and_height = str(width)+'x'+str(height)
    objects = detect_obj(decoded_img)

    result = {
        "face_found": face_found,
        "bg_white": bg_white,
        "other_objects": objects,
        "width": width,
        "height": height,
        "width and height": width_and_height,
        "size" : size
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

    print(colorPercent)
    return True if colorPercent > 35.0 else False
    

def detect_faces_in_image(file_stream):
    
    # Load the uploaded image file
    img = face_recognition.load_image_file(file_stream)
    face_locations = face_recognition.face_locations(img)

    print("I found {} face(s) in this photograph.".format(len(face_locations)))
    
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
    return True if len(face_locations) > 0 else False

def detect_obj(img):
    blob = cv2.dnn.blobFromImage(img, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop = False)

    #Set the input the the net
    net.setInput(blob)
    outs = net.forward (getOutputsNames(net))
   
    return postprocess (img, outs)

