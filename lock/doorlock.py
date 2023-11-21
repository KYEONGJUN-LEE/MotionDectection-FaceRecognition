from flask import Flask, render_template, send_file
from firebase_admin import credentials, initialize_app
from firebase_admin import storage
import os
import cv2
import numpy as np
import picamera
import picamera.array
import os
import time
import threading
import RPi.GPIO as GPIO
import signal  # 시그널 모듈을 임포트합니다.
import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage  
from uuid import uuid4
import schedule
from datetime import datetime
import requests



app = Flask(__name__)


PROJECT_ID = "fir-storage-ea381"

# Firebase 서비스 계정 키를 로드합니다.
cred = credentials.Certificate("/home/KHM/HomeCamera_FaceOpenDoorLock/Artifical Intelligence/serviceAccount.json")
firebase_app = initialize_app(cred, {
    'storageBucket': 'fir-storage-ea381.appspot.com'
})

# Firebase Storage 클라이언트를 초기화합니다. 
bucket  = storage.bucket()

# def fileUpload(file):
#     blob = bucket.blob('image_store/'+file) #저장한 사진을 파이어베이스 storage의 image_store라는 이름의 디렉토리에 저장
#     #new token and metadata 설정
#     new_token = uuid4()
#     metadata = {"firebaseStorageDownloadTokens": new_token} #access token이 필요하다.
#     blob.metadata = metadata
 
#     #upload file
#     blob.upload_from_filename(filename='/home/KHM/HomeCamera_FaceOpenDoorLock/'+file, content_type='image/png') #파일이 저장된 주소와 이미지 형식(jpeg도 됨)
#     #debugging hello
#     print("저장완료 ")
#     print(blob.public_url)

    #흑백사진 업로드
def upload_image_to_firebase(image_path, remote_path):
    # 이미지를 Firebase Cloud Storage에 업로드
    bucket = storage.bucket()
    blob = bucket.blob(remote_path)
    blob.upload_from_filename(image_path)
    
    # 업로드 후 로컬 파일 삭제 (선택 사항)
    os.remove(image_path)
# GPIO 핀 번호 설정
TRIG_PIN = 14
ECHO_PIN = 15

# GPIO 핀 모드 설정
GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIG_PIN, GPIO.OUT)
GPIO.setup(ECHO_PIN, GPIO.IN)

prev_frame = None
min_area = 1000  # 윤곽선을 감지하기 위한 최소 영역 크기 (조절 가능)

capture_directory = 'lock_captures/'  # 이미지 캡처를 저장할 디렉토리
output_directory = "test_images"  # 흑백 사진을 저장할 폴더 경로 ,도어락 카메라가 찍은 사진

# 이미지 캡처를 저장할 디렉토리 생성
if not os.path.exists(capture_directory):
    os.makedirs(capture_directory)
# 도어락 흑백 사진 저장할 디렉토리 생성
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

last_capture_time = 0  # 마지막으로 사진을 찍은 시간을 저장하는 변수
capture_interval = 5  # 사진 찍기 간격 (초)

 

# 초음파 센서로부터 거리를 측정하는 함수
def measure_distance():
    # 초음파 센서 트리거 핀을 10us 동안 활성화
    GPIO.output(TRIG_PIN, True)
    time.sleep(0.00001)
    GPIO.output(TRIG_PIN, False)

    start_time = time.time()
    stop_time = time.time()

    # 에코 핀에서 신호가 들어올 때까지의 시간 측정
    while GPIO.input(ECHO_PIN) == 0:
        start_time = time.time()

    while GPIO.input(ECHO_PIN) == 1:
        stop_time = time.time()

    # 초음파 센서로부터 거리 계산 (음속: 343m/s)
    elapsed_time = stop_time - start_time
    distance = (elapsed_time * 34300) / 2  # 거리(cm) 계산
    return distance




def run_camera():
    with picamera.PiCamera() as camera:
        camera.resolution = (320, 240)  # 낮은 해상도
        camera.framerate = 15  # 낮은 프레임 속도

        camera.rotation = 180
        raw_capture = picamera.array.PiRGBArray(camera, size=(320, 240))

        for _ in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
            frame = raw_capture.array



            distance = measure_distance()


             # 거리가 15cm 미만이고 모션 감지된 경우 사진 캡처
            if distance < 5 :
                capture_time = datetime.now().strftime("%Y%m%d%H%M%S")  # 현재 날짜와 시간을 문자열로 포맷팅
                capture_file_path = os.path.join(capture_directory, f'{capture_time}.jpg')
                cv2.imwrite(capture_file_path, frame)
                time.sleep(capture_interval)
                face_image_path = detect_faces(capture_file_path)

                if face_image_path:  # 얼굴 검출이 제대로 되었을 때만 처리
                    remote_path = f'image_store/lock_captures/{capture_time}.jpg'
                    upload_image_to_firebase(face_image_path, remote_path)

                    bw_image_path = process_and_save_image(capture_file_path)  # 블랙으로 바꾸고 얼굴 검출
                    if bw_image_path:  # 얼굴 검출이 실패하지 않았을 때만 서버로 전송
                        remote_path = f'image_store/lock_blackcaptures/{capture_time}.jpg'  # Firebase Cloud Storage에 업로드될 경로 및 파일 이름
                        upload_image_to_firebase(bw_image_path, remote_path)


            # OpenCV 창에 비디오 스트림 표시
            cv2.imshow("Motion Detection", frame)

            # 'q' 키를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            raw_capture.truncate(0)

    # 비디오 캡처 종료 후 OpenCV 창 닫기  
    cv2.destroyAllWindows()

# Haar Cascade 얼굴 검출기 초기화
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_cascade2 = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(image_path):
    # 이미지 불러오기
    image = cv2.imread(image_path)
    
    # 얼굴 검출
    faces = face_cascade2.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # output_path 변수 초기화
    output_path = None
    
    if len(faces) == 0:
        # 얼굴이 검출되지 않은 경우에는 output_path를 초기화하고 아무 작업도 수행하지 않습니다.
        pass
    else:
        for i, (x, y, w, h) in enumerate(faces):
            # 원본 이미지에서 얼굴 부분 추출
            face = image[y:y+h, x:x+w]
            
            # 새로운 파일 이름 생성
            current_time = datetime.now().strftime("%Y%m%d%H%M%S")
            new_filename = f"{current_time}{i}.jpg"
            
            # 얼굴 이미지 저장
            output_path = os.path.join(output_directory, new_filename)
            cv2.imwrite(output_path, face)

    return output_path

def process_and_save_image(image_path):
    # 이미지 불러오기
    image = cv2.imread(image_path)
    
    # 이미지를 흑백으로 변환
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 얼굴 검출
    faces = face_cascade.detectMultiScale(grayscale_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # output_path 변수 초기화
    output_path = None
    
    if len(faces) == 0:
        # 얼굴이 검출되지 않은 경우에는 output_path를 초기화하고 아무 작업도 수행하지 않습니다.
        pass
    else:
        for i, (x, y, w, h) in enumerate(faces):
            # 원본 이미지에서 얼굴 부분 추출
            face = image[y:y+h, x:x+w]
            
            # 추출된 얼굴 이미지를 흑백으로 변환
            grayscale_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            
            # 새로운 파일 이름 생성
            current_time = datetime.now().strftime("%Y%m%d%H%M%S")
            new_filename = f"{current_time}{i}.jpg"
            
            # 흑백 얼굴 이미지 저장
            output_path = os.path.join(output_directory, new_filename)
            cv2.imwrite(output_path, grayscale_face)

            send_bw_image_to_server2(output_path)
    
    return output_path


def send_bw_image_to_server2(image_path):
    with open(image_path, 'rb') as image_file:
        files = {'image': (image_path, image_file)}

        # RESTful API 엔드포인트 및 데이터 설정
        api_url = 'http://192.168.1.5:9097/receive_image'
        payload = {'description': '흑백 이미지 설명'}
        
        # RESTful POST 요청 보내기
        response = requests.post(api_url, data=payload, files=files)

        if response.status_code == 200:
            print("흑백 이미지가 성공적으로 서버 2로 전송되었습니다.")
        else:
            print("흑백 이미지 전송 실패:", response.status_code)

    # 이미지 파일을 명시적으로 닫습니다.
    image_file.close()

# 이미지를 다운로드할 로컬 경로 설정
download_folder = '/home/KHM/HomeCamera_FaceOpenDoorLock/lock/static/image'

# 이미 다운로드한 파일들을 추적할 집합(set)을 생성
downloaded_files = set()
# 이미지 파일 경로 리스트
IMG_LIST = os.listdir("/home/KHM/HomeCamera_FaceOpenDoorLock/lock/static/image")
IMG_FOLDER = os.path.join("static", "image")
app.config["UPLOAD_FOLDER"] = IMG_FOLDER
IMG_FOLDER = "image"


@app.route('/')
def download_images():
    # 'train' 폴더에 있는 모든 파일 목록 가져오기
    IMG_LIST = os.listdir("/home/KHM/HomeCamera_FaceOpenDoorLock/lock/static/image")
    blobs = bucket.list_blobs(prefix='image_store/lock_captures/')

    for blob in blobs:
        # 각 파일을 다운로드할 로컬 경로 설정
        file_name = os.path.basename(blob.name)

        # 이미 다운로드한 파일인지 확인
        if file_name not in downloaded_files:
            download_path = os.path.join(download_folder, file_name)
            print("다운로드 주소:",download_path)
            # 이미지 다운로드 및 로컬 저장
            blob.download_to_filename(download_path)
            downloaded_files.add(file_name)  # 다운로드한 파일을 집합에 추가

            print(f"{file_name} 이미지가 로컬 다운로드 폴더에 저장되었습니다.")
        else:
            print(f"{file_name} 이미지는 이미 다운로드되었습니다.")


        
    IMG_LIST = sorted(IMG_LIST, key=lambda x: os.path.getmtime(os.path.join(download_folder, x)), reverse=True)
    IMG_LIST = [os.path.join(IMG_FOLDER, i) for i in IMG_LIST]

    return render_template('visit.html',image_files=IMG_LIST)


if __name__ == '__main__':
    # 카메라 스레드 시작
    camera_thread = threading.Thread(target=run_camera)
    camera_thread.daemon = True
    camera_thread.start()
    app.run(host='0.0.0.0', port=9092)

