
"""
./dataset : 학습 동영상을 받아오는 폴더
./dataset_split/{uuid} : 학습 동영상을 프레임 단위로 분할
./get_eye : 프레임으로 나눈 사진에서 눈을 탐지해 따로 추출
./eye_model : 동영상 눈 사진을 기반으로 keras 학습모델을 작성

./inference_img : 요청받은 사진을 저장
./infer_eye : 요청받은 사진을 복사한뒤 이름뒤에 시간 정보를 기입
./final_eye : 요청받은 사진에서 눈을 추출
"""

import datetime as dt
import json
import os
import shutil
import cv2
import tensorflow as tf
from flask import Flask, request
import base64
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image
from tqdm import tqdm


face_cascade = cv2.CascadeClassifier('./haarcascade_eye.xml')
eye_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

app = Flask(__name__)

"""
json 타입의 POST 요청으로 uuid와 base64이미지를 받은 뒤 keras 학습모델을 거쳐 
시선이 어디를 보는지를 0~3으로 리턴
"""
@app.route('/inference', methods=['POST'])
def inference():
    # img : 이미지 파일
    # uuid : 사용자식별자
    # if request.method == "POST":

    get_img = request.json['file']
    address = request.json['uuid']
    decode = base64.b64decode(get_img)
    current = str(dt.datetime.now()).replace(" ", "_").replace(":", "").replace("-", "").replace(".", "_")
    with open(f'./inference_img/{address}_{current}.jpg', 'wb') as f:
        f.write(decode)

    if not os.path.exists(f"infer_eye/{address}"):
        os.makedirs(f"infer_eye/{address}")
    shutil.copy(os.path.join("inference_img", f"{address}_{current}.jpg"),
                os.path.join(f"infer_eye/{address}", f"{address}_{current}.jpg"))

    infercam(os.path.join(f"infer_eye/{address}", f"{address}_{current}.jpg"))

    load = tf.keras.models.load_model(f'eye_model/{address}.keras')

    file = os.path.join(f"final_eye", f'{address}_{current}.jpg')

    # for idx, file in enumerate(image_list):

    img = image.load_img(file, target_size=(160, 80))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    result = load.predict(img)[0]
    print(result.argmax())
    # result_list.append([max(result[0]), int(result.argmax())])
    try:
        # print("result list : ", result_list)
        response = int(result.argmax())
    except IndexError:
        response = 4
    shutil.rmtree(f"infer_eye/{address}")
    result = json.dumps({"result": response})
    print(result)
    return result


"""
json 타입의 POST 요청으로 uuid와 base64이미지를 받은 뒤 keras 학습모델을 만들기위해
동영상을 쪼갠뒤 눈검출을 하고 케라스 모델생성
"""


@app.route('/shat', methods=['POST'])
def shat():
    if request.method == "POST":
        try:
            data = request.json['file']
            decode = base64.b64decode(data)
            target = request.json['uuid']
            # print(target)
            with open(f'./dataset/{target}.avi', 'wb') as f:
                f.write(decode)
            videocut(target + ".avi")
            cam(target)
            mkmodel(target)

            result = json.dumps({"result": '200'})
            return result
        except Exception as e:
            print(e)
            print("video date not found")
        result = json.dumps({"result": "200"})
        return result


"""
모델 작성
"""


def mkmodel(target):
    # 데이터세트 전처리 및 준비
    train_generator = ImageDataGenerator(
        rescale=1. / 255,
        validation_split=0.2,
        rotation_range=0.2,
    )

    train_flow = train_generator.flow_from_directory(
        directory=f"./get_eye/{target}",
        shuffle=True,
        target_size=(160, 80),
        class_mode="categorical",
        batch_size=8,
        subset="training"  # training, test, validation
    )

    val_flow = train_generator.flow_from_directory(
        directory=f"./get_eye/{target}",
        shuffle=True,
        target_size=(160, 80),
        class_mode="categorical",
        batch_size=8,
        subset="validation"  # training, test, validation
    )

    model = Sequential()
    model.add(Conv2D(8, kernel_size=(5, 5), activation='relu',
                     padding="valid", input_shape=(160, 80, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(8, kernel_size=(5, 5), activation='relu',
                     padding="valid"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    hist = model.fit(train_flow, epochs=30,
                     validation_data=val_flow,
                     validation_split=0.2)

    model.save(f"./eye_model/{target}.keras")


"""
비디오를 사용구간만 프레임으로 저장
"""


def videocut(videoname):
    filepath = rf'./dataset/{videoname}'
    video = cv2.VideoCapture(filepath)
    filesplit = rf'./dataset_split/{videoname}'

    if not video.isOpened():
        print("Could not Open :", filepath)
        exit(0)

    # 불러온 비디오 파일의 정보 출력
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)

    print("length :", length)
    print("width :", width)
    print("height :", height)
    print("fps :", fps)

    # 프레임을 저장할 디렉토리를 생성
    try:
        if not os.path.exists(filesplit[:-4]):
            os.makedirs(filesplit[:-4])
            os.makedirs(os.path.join(filesplit.rsplit(".", maxsplit=1)[0], 'lt'))
            os.makedirs(os.path.join(filesplit.rsplit(".", maxsplit=1)[0], 'lb'))
            os.makedirs(os.path.join(filesplit.rsplit(".", maxsplit=1)[0], 'rt'))
            os.makedirs(os.path.join(filesplit.rsplit(".", maxsplit=1)[0], 'rb'))
    except OSError:
        print('Error: Creating directory. ' + filesplit.rsplit(".", maxsplit=1)[0])

    while (video.isOpened()):
        ret, image = video.read()
        if int(video.get(1)) < 120:
            cv2.imwrite(filesplit.rsplit(".", maxsplit=1)[0] + "/lt/%d.jpg" % int(video.get(1)), image)
        elif int(video.get(1)) < 150:
            pass

        elif int(video.get(1)) < 270:
            cv2.imwrite(filesplit.rsplit(".", maxsplit=1)[0] + "/rt/%d.jpg" % int(video.get(1)), image)
        elif int(video.get(1)) < 300:
            pass

        elif int(video.get(1)) < 420:
            cv2.imwrite(filesplit.rsplit(".", maxsplit=1)[0] + "/lb/%d.jpg" % int(video.get(1)), image)
        elif int(video.get(1)) < 450:
            pass

        elif int(video.get(1)) < 570:
            cv2.imwrite(filesplit.rsplit(".", maxsplit=1)[0] + "/rb/%d.jpg" % int(video.get(1)), image)
        elif int(video.get(1)) == 570:
            break

    video.release()


"""
눈 추출
"""


def cam(current):
    empty_eye = 0
    for i in ['lt', 'lb', 'rt', 'rb']:
        os.makedirs(os.path.join('./get_eye', current, i))
    for folder in ('lb', 'lt', 'rb', 'rt'):
        for imgfile in tqdm(os.listdir(os.path.join('./dataset_split/' + current, folder))):
            cap = cv2.VideoCapture(os.path.join('./dataset_split/' + current, folder, imgfile))
            ret, img = cap.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            min_x = min_y = 99999
            max_x = max_y = 0
            for idx, (x, y, w, h) in enumerate(faces[:2]):
                if x < min_x:
                    min_x = x
                if y < min_y:
                    min_y = y
                if x + w > max_x:
                    max_x = x + w
                if y + h > max_y:
                    max_y = y + h

            cv2.rectangle(img, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)
            roi_gray = gray[min_y:max_y, min_x: max_x]

            if len(roi_gray):
                cv2.imwrite(f'./get_eye/{current}/{folder}/{os.path.basename(imgfile)}', roi_gray)
            else:
                empty_eye+=1
                # print("empty eye")
    print("empty eye :", empty_eye)


def infercam(current):
    cap = cv2.VideoCapture(current)
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    min_x = min_y = 99999
    max_x = max_y = 0
    for idx, (x, y, w, h) in enumerate(faces[:2]):
        if x < min_x:
            min_x = x
        if y < min_y:
            min_y = y
        if x + w > max_x:
            max_x = x + w
        if y + h > max_y:
            max_y = y + h

    cv2.rectangle(img, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)
    roi_gray = gray[min_y:max_y, min_x: max_x]
    try:
        # shutil.copy(current,
        #             os.path.join(f"infer_eye/{current}"))
        cv2.imwrite(
            os.path.join(
                'final_eye',
                os.path.basename(current)
            ), roi_gray)
    except Exception as e:
        print(e)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9123)
