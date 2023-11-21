from flask import Flask,render_template, request, send_from_directory,jsonify,session
import firebase_admin
from firebase_admin import credentials, auth, db
import os
from datetime import datetime
import cv2
import numpy as np
import numpy as np
import os
from datetime import datetime
import base64
import tempfile
import requests

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # 비밀 키 설정


# Firebase 초기화

cred = credentials.Certificate("C:\\Users\\kj100\\OneDrive\\바탕 화면\\HomeCamera_FaceOpenDoorLock\\HomeCamera_FaceOpenDoorLock\\service_key\\serviceAccount.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://fir-storage-ea381-default-rtdb.firebaseio.com'
})



@app.route('/')
def index():
    return render_template('start.html')

@app.route('/register',methods=['POST', 'GET'])
def register_index():
    return render_template('register.html')

@app.route('/login')
def logout_index():
    return render_template('login.html')

@app.route('/homecam')
def homacam_index():
    return render_template('homecam.html')

@app.route('/choice')
def homeback_index():
    return render_template('choice.html')

@app.route('/homecam')
def homacamback_index():
    return render_template('homecam.html')

@app.route('/faceid')
def faceid_index():
    return render_template('faceid.html')

# @app.route('/user_image/<filename>')
# def user_image(filename):
#     return send_from_directory('train', f'{filename}.jpg')


#s



# 비디오 스트림을 위한 라우트와 핸들러 함수 설정
@app.route('/video_feed')
def video_feed():
    return app.response_class(train_face(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/login', methods=['POST'])
def login():
    try:
        user_data = request.get_json()  # 클라이언트에서 전송한 사용자 정보를 받음
        email = user_data['email']
        uid = user_data['uid']

        # 사용자 정보를 세션에 저장
        session['uid'] = uid
        session['email'] = email

        # 여기에서 사용자 정보를 처리하거나 필요한 동작을 수행할 수 있습니다.

        return jsonify({'success': True, 'message': '로그인 성공'})  # 성공 응답
    except Exception as e:
        return jsonify({'success': False, 'message': '로그인 실패: ' + str(e)})  # 실패 응답

# 사용자 엔드포인트
@app.route('/user')
def user_index():
    try:
        # 세션에서 사용자 정보를 가져옵니다.
        uid = session.get('uid')
        email = session.get('email')

        if uid is None:
            return jsonify({'error': '사용자 정보를 가져올 수 없습니다.'})

        # 파이어베이스에서 사용자 정보 가져오기
        user = auth.get_user(uid)

        created_timestamp = user.user_metadata.creation_timestamp / 1000  # 밀리초를 초로 변환
        created_datetime = datetime.utcfromtimestamp(created_timestamp).strftime('%Y-%m-%d %H:%M:%S')
        user_data = {
            'uid': user.uid,
            'email': user.email,
            'name': user.display_name,  # Firebase에서 설정한 사용자 이름
            'created_at': created_datetime  # 변환된 생성 시간
        }

        return render_template('user.html', user_data=user_data)

    except Exception as e:
        return jsonify({'error': '사용자 정보를 가져올 수 없습니다.', 'message': str(e)})




@app.route('/register_success', methods=['POST', 'GET'])
def register():
    data = request.json
    username = data['username']
    password = data['password']
    name = data['name']
    phoneNumber = data['phoneNumber']
    email = data['username']
    print("사용자 정보:", username, password, name, phoneNumber, email)
   

    try:
        # Firebase Authentication을 사용하여 사용자 등록
        user = auth.create_user(
            email=email,
            password=password,
            display_name=name
        )

        # 사용자 정보를 Firebase Realtime Database에 저장
        user_data = {
            'username': username,
            'name': name,
            'phoneNumber': phoneNumber
        }
        print("사용자 정보:", user)
        print("사용자 정보:", user_data)
        db.reference('/users/' + user.uid).update(user_data)  # update 메서드를 사용하여 업데이트

        return "회원가입 성공!"
    except Exception as e:
        print(str(e))
        return "회원가입 실패: " + str(e)
    
# DNN 모델과 가중치 파일 경로 설정
prototxt = 'deploy.prototxt.txt'
model = 'res10_300x300_ssd_iter_140000_fp16.caffemodel'
net = cv2.dnn.readNetFromCaffe(prototxt, model)
os.makedirs('train', exist_ok=True)

# 얼굴 이미지를 저장할 디렉토리 설정
TRAIN_IMAGE_DIR = 'train'

# 얼굴 이미지 학습
@app.route('/faceid', methods=['POST'])
def train_face():
    try:
        # 이미지 처리 및 출력 루프
        face_images = []

        capture = cv2.VideoCapture(0)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

        max_images = 5  # 최대 저장할 얼굴 이미지 수
        capturing = False  # 이미지 캡처 상태

        while len(face_images) < max_images:
            ret, frame = capture.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # DNN 모델을 통한 얼굴 감지
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (200 , 200)), 1.0, (320, 720), (104.0, 177.0, 123.0))
            net.setInput(blob)
            detections = net.forward()

            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([320, 240, 320, 240])
                    (startX, startY, endX, endY) = box.astype(int)

                    # 감지된 얼굴 주변에 큰 사각형 표시
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 4)

                    if capturing:
                        if len(face_images) < max_images:
                            face_image = gray[startY:endY, startX:endX]
                            face_image = cv2.resize(face_image, (320,320))
                            if not face_image is None and not face_image.size == 0:
                                # 얼굴 이미지를 저장
                                current_image_count = len(face_images)
                                image_filename = os.path.join(TRAIN_IMAGE_DIR, f'{current_image_count}.jpg')
                                cv2.imwrite(image_filename, face_image)
                                print(f'얼굴 이미지 저장: {image_filename}')
                                face_images.append(face_image)  # 얼굴 이미지 저장

                               

                    if len(face_images) >= max_images:
                        break

            cv2.imshow('image', frame)
            key = cv2.waitKey(1)

            if key > 0:
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # 이미지 캡처 시작 또는 종료
                    capturing = not capturing

        # 카메라 종료
        capture.release()
        cv2.destroyAllWindows()
        

        # train 폴더에 저장된 얼굴 이미지를 사용하여 학습 수행
        if len(face_images) > 0:
            
            labels = np.array([0] * len(face_images))
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.train(face_images, labels)
            model_file = 'trained_model.yml'
            recognizer.save(model_file)
            return jsonify({'message': '사진 촬영이 완료되었습니다. trained_model.yml이 생성되었습니다.'}), 200
            
        else:
            # 에러 메시지도 JSON 형식으로 포함합니다.
            print("2")
            return jsonify({'message': '학습을 위한 얼굴 이미지가 캡처되지 않았습니다.'}), 400

    except Exception as e:
        print(e)
        return jsonify({'error': f'학습 도중 오류가 발생했습니다: {str(e)}'}), 400
    

# 이미지 폴더 생성
image_directory = 'test_images'
os.makedirs(image_directory, exist_ok=True)

@app.route('/receive_image', methods=['POST'])
def receive_image():
    # POST 요청에서 이미지 데이터 가져오기
    image_data = request.files['image']
    current_time = datetime.now().strftime('%Y%m%d%H%M%S')
    image_filename = f'received_image_{current_time}.jpg'

    # 이미지를 파일로 저장
    image_path = os.path.join(image_directory, image_filename)
    image_data.save(image_path)

    print("이미지를 받아 저장했습니다:", image_path)

    # 저장한 이미지를 테스트 이미지로 사용하고 결과를 반환
    result = predict(image_path)
    print('결과', result)

    send_result_to_server(result)

    # 테스트 이미지 삭제
    os.remove(image_path)
    print("테스트 이미지를 삭제했습니다:", image_path)

    return '성공'



def predict(image_path):
    try:
        # 테스트 이미지 로드
        test_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # 얼굴 인식 모델 생성
        recognizer = cv2.face_LBPHFaceRecognizer.create()

        # 학습된 모델 로드
        model_file = 'trained_model.yml'
        recognizer.read(model_file)
        print("모델코드:", model_file)

        # 이미지를 얼굴 인식 모델에 적용
        label, confidence = recognizer.predict(test_image)
        accuracy = int(100 - confidence)
        print("정확도:", accuracy)

        # 결과를 JSON 형식으로 반환
        if confidence < 150:
            return accuracy
        else:
            return jsonify({'error': '얼굴이 감지되지 않았거나 신뢰도가 너무 낮습니다.', 'confidence': confidence}), 400

    except Exception as e:
        return jsonify({'error': f'얼굴 인식 중 오류 발생: {str(e)}'}), 500

def send_result_to_server(number):
    # RESTful API 엔드포인트 설정
    api_url = 'http://220.69.203.19:9092/face_test_result'
    
    # 숫자 데이터를 딕셔너리로 묶어서 전송
    payload = {'number': number}
    
    # RESTful POST 요청 보내기
    response = requests.post(api_url, json=payload)

    if response.status_code == 200:
        print(f"결과 {number}가 성공적으로 서버로 전송되었습니다.")
    else:
        print(f"결과 {number} 전송 실패:", response.status_code)


@app.route('/choice')
def choice():
    return render_template('choice.html')

@app.route('/faceid')
def face():
    return render_template('faceid.html')

if __name__ == '__main__':

    app.run(host='0.0.0.0', port=5124)
