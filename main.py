from flask import Flask, render_template, request
from google.cloud import vision
from google.cloud.vision import types
from PIL import Image, ImageDraw
import tempfile
from os.path import join
import io
from base64 import b64encode

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return  render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file_handler = request.files['picture']
    mime = file_handler.mimetype
    image_file_location = join(tempfile.gettempdir(), file_handler.filename)
    file_handler.save(image_file_location)
    file_handler.close()
    
    with io.open(image_file_location, 'rb') as file_handler:
        g_image = types.Image(content=file_handler.read())

    client = vision.ImageAnnotatorClient()

    label_detection_response = client.label_detection(image=g_image)
    face_detection_response = client.face_detection(image=g_image)

    p_image = Image.open(image_file_location)
    draw = ImageDraw.Draw(p_image)
    
    for face in face_detection_response.face_annotations:
        bounding_box = tuple([(i.x, i.y) for i in face.bounding_poly.vertices])
        print(bounding_box)
        draw.polygon(bounding_box, outline='blue')

    p_image.save(image_file_location)


    with io.open(image_file_location, 'rb') as file_handler:
        data = b64encode(file_handler.read()).decode('UTF-8')

    return render_template('display.html', mime=mime, data=data, labels=label_detection_response.label_annotations)