from flask import Flask, request, jsonify

from faceRecognition import faceRecognition

face_recognition = faceRecognition()
app = Flask(__name__)


@app.route('/face_recognition', methods=['POST'])
def face_recongnition():
    img_path = request.form['imagePath']
    result = face_recognition.compare(img_path)
    return result


app.run(port=5000)
