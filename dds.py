
from flask import Flask, jsonify
from threading import Thread
import cv2
import dlib
import imutils
from imutils import face_utils
from scipy.spatial import distance as dist
from picamera2 import Picamera2

app = Flask(__name__)

ear_value = 0.0
mar_value = 0.0

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[13], mouth[19])  # vertical distance
    B = dist.euclidean(mouth[14], mouth[18])  # vertical distance
    C = dist.euclidean(mouth[15], mouth[17])  # vertical distance
    D = dist.euclidean(mouth[12], mouth[16])  # horizontal distance
    mar = (A + B + C) / (2.0 * D)
    return mar

@app.route('/data', methods=['GET'])
def get_data():
    return jsonify({'EAR': ear_value, 'MAR': mar_value})

def camera_loop():
    global ear_value, mar_value
    facial_landmark_predictor = "/home/pi/Downloads/shape_predictor_68_face_landmarks (1).dat"
    minimum_ear = 0.25
    maximum_mar = 0.6

    face_detector = dlib.get_frontal_face_detector()
    landmark_finder = dlib.shape_predictor(facial_landmark_predictor)

    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration())
    picam2.start()

    (left_eye_start, left_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (right_eye_start, right_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    (mouth_start, mouth_end) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

    try:
        while True:
            frame = picam2.capture_array()
            frame = imutils.resize(frame, width=800)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_detector(gray_frame, 0)

            for face in faces:
                face_landmarks = landmark_finder(gray_frame, face)
                face_landmarks = face_utils.shape_to_np(face_landmarks)

                left_eye = face_landmarks[left_eye_start:left_eye_end]
                right_eye = face_landmarks[right_eye_start:right_eye_end]
                mouth = face_landmarks[mouth_start:mouth_end]

                left_ear = eye_aspect_ratio(left_eye)
                right_ear = eye_aspect_ratio(right_eye)
                ear_value = (left_ear + right_ear) / 2.0
                mar_value = mouth_aspect_ratio(mouth)
           
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    thread = Thread(target=camera_loop)
    thread.start()
    app.run(host='0.0.0.0', port=5000)

