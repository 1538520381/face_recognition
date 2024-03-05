import glob
import os
import cv2
import face_recognition
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 照片路径
images_path = 'images'
# 照片编码
images_encoding = []
images_name = []
# 容忍度
tolerance = 0.48


# 初始化照片编码
def load_images():
    images = glob.glob(images_path + '/*.jpg') + glob.glob(images_path + '/*.png')
    for image in images:
        images_encoding.append(face_recognition.face_encodings(face_recognition.load_image_file(image))[0])
        images_name.append(os.path.splitext(os.path.basename(image))[0])


# 标记人脸
def mark(frame, name, position, fill):
    font = ImageFont.truetype('simsun.ttc', 30)
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, name, font=font, fill=fill)
    return np.array((img_pil))


load_images()

# 摄像机
camera = cv2.VideoCapture(0)
while True:
    # 摄像头读取图像帧
    # flag:是否读取到图像
    # frame:图像帧
    flag, frame = camera.read()

    # 将图像转换为RGB格式
    rgb = frame[:, :, ::-1]

    # 检测图像中的人脸
    face_locations = face_recognition.face_locations(rgb)
    face_encodings = face_recognition.face_encodings(rgb, face_locations)

    # 人脸比对
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        distance = face_recognition.face_distance(images_encoding, face_encoding)
        min_distance_index = np.argmin(distance)
        min_distance = distance[min_distance_index]

        name = "unknow"
        if min_distance <= tolerance:
            name = images_name[min_distance_index]

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        frame = mark(frame, name, (left, top - 38), (0, 0, 255))

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
