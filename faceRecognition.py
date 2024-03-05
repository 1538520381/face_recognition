import face_recognition
import glob
import os
import cv2


class faceRecognition():
    tolerance = 0.48
    images_path = 'images'
    images_locations = []
    images_encodings = []
    images_names = []

    def __init__(self):
        images = glob.glob(os.path.join(self.images_path, '*.jpg')) + glob.glob(os.path.join(self.images_path, '*.png'))
        for image in images:
            self.images_names.append(os.path.splitext(os.path.basename(image))[0])

            image = face_recognition.load_image_file(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image_location = face_recognition.face_locations(image)[0]
            image_encoding = face_recognition.face_encodings(image)[0]

            self.images_locations.append(image_location)
            self.images_encodings.append(image_encoding)

    def compare(self, path):
        image = face_recognition.load_image_file(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # image_location = face_recognition.face_locations(image)[0]
        image_encodings = face_recognition.face_encodings(image)
        matches = []
        for image_encoding in image_encodings:
            result = face_recognition.compare_faces(self.images_encodings, image_encoding, tolerance=self.tolerance)
            if True in result:
                index = result.index(True)
                matches.append(self.images_names[index])
        return matches
