import os
import torch
import cv2
import numpy as np
import pandas as pd
import csv
#from scipy.spatial import distance as dist
#import mediapipe as mp
#from mediapipe.python.solutions.face_mesh import FACEMESH_TESSELATION
import torch.utils.data.dataset
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image

def get_lists_mediapipe():
    root_dir='/mnt/c/Users/charu/VS/ViT/DDD/'
    sub_dir=['Drowsy', 'Non Drowsy']
    img_f=[]
    labels=[]
    ear=[]
    files_none=[]
    for i in sub_dir:
        for f in os.listdir(root_dir+i):
            
            image = cv2.imread(root_dir+i+"/"+f)
            with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
                with mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
                    
                        rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        rgb_frame.flags.writeable = False

                        detection_results = face_detection.process(rgb_frame)

                        if detection_results.detections:
                            for detection in detection_results.detections:
                                
                                face_mesh_results = face_mesh.process(rgb_frame)

                                if face_mesh_results.multi_face_landmarks:
                                    for face_landmarks in face_mesh_results.multi_face_landmarks:
                                        left_eye_landmarks = [face_landmarks.landmark[i] for i in range(133, 144)]
                                        left_eye = [(landmark.x * image.shape[1], landmark.y * image.shape[0]) for landmark in left_eye_landmarks]
                                        left_ear = calculate_ear(left_eye)
                                        right_eye_landmarks = [face_landmarks.landmark[i] for i in range(362, 374)]
                                        right_eye = [(landmark.x * image.shape[1], landmark.y * image.shape[0]) for landmark in right_eye_landmarks]
                                        right_ear = calculate_ear(right_eye)
                                        ear_val=(right_ear+left_ear)/2.0
                                        img_f.append(root_dir+i+"/"+f)
                                        labels.append(i)
                                        ear.append(ear_val)

                        else:
                            files_none.append(root_dir+i+"/"+f)
    return img_f,ear,labels,files_none
def get_lists_dlib():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # Define eye landmarks indices for left and right eye
    (left_eye_start, left_eye_end) = (42, 48)
    (right_eye_start, right_eye_end) = (36, 42)

    # Function to calculate Eye Aspect Ratio (EAR)
    def calculate_ear(eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    root_dir='/mnt/c/Users/charu/VS/ViT/DDD/'
    sub_dir=['Drowsy', 'Non Drowsy']
    img_f=[]
    labels=[]
    ear=[]
    files_none=[]
    for i in sub_dir:
        for f in os.listdir(root_dir+i):
            
            image = cv2.imread(root_dir+i+"/"+f)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = detector(gray, 0)
            if (len(faces)!=0):
                for face in faces:
                    shape = predictor(gray, face)
                    shape = np.array([[p.x, p.y] for p in shape.parts()])
                    
                    left_eye = shape[left_eye_start:left_eye_end]
                    right_eye = shape[right_eye_start:right_eye_end]

                    left_ear = calculate_ear(left_eye)
                    right_ear = calculate_ear(right_eye)
                    ear_val = (left_ear + right_ear) / 2.0
                    print(f,ear_val)
                    img_f.append(root_dir+i+"/"+f)
                    labels.append(i)
                    ear.append(ear_val)
    return img_f,ear,labels

def display_save(file):
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh

    
    def calculate_ear(eye_landmarks):
        A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
        B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
        C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
        ear = (A + B) / (2.0 * C)
        return ear
    image = cv2.imread(file)
    #cv2.imshow("im",image)
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        with mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
            
                rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                rgb_frame.flags.writeable = False

                detection_results = face_detection.process(rgb_frame)

                if detection_results.detections:
                    for detection in detection_results.detections:
                        
                        mp_drawing.draw_detection(image, detection)

                        face_mesh_results = face_mesh.process(rgb_frame)

                        if face_mesh_results.multi_face_landmarks:
                            for face_landmarks in face_mesh_results.multi_face_landmarks:
                                mp_drawing.draw_landmarks(
                                image=image,
                                landmark_list=face_landmarks,
                                connections=mp_face_mesh.FACEMESH_TESSELATION,
                                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1))
                                
                                                            
                                left_eye_landmarks = [face_landmarks.landmark[i] for i in range(133, 144)]
                                left_eye = [(landmark.x * image.shape[1], landmark.y * image.shape[0]) for landmark in left_eye_landmarks]
                                left_ear = calculate_ear(left_eye)
                                right_eye_landmarks = [face_landmarks.landmark[i] for i in range(362, 374)]
                                right_eye = [(landmark.x * image.shape[1], landmark.y * image.shape[0]) for landmark in right_eye_landmarks]
                                right_ear = calculate_ear(right_eye)
                                print(right_ear, left_ear)

                                cv2.putText(image, f'L-EAR: {left_ear:.2f}', (5, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                cv2.putText(image, f'R-EAR: {right_ear:.2f}', (120, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                fname=file.split("/")[-1]
                                cv2.namedWindow(fname,cv2.WINDOW_NORMAL)
                                cv2.resizeWindow(fname, 500,500)
                                cv2.imshow(fname,image)
                                cv2.imwrite('/mnt/c/Users/charu/VS/ViT/'+ fname, image)

                
                cv2.imshow('MediaPipe Face Detection and Mesh', image)

def create_csv(op_file):
    img_files,ear,labels,none_files=get_lists()
    data={
        "image_file_paths":img_files,
        "EAR":ear,
        "labels":labels
    }
    with open(op_file,'w',newline='') as f:
        writer=csv.write(f)
        writer.writerow(data.keys())
        rows=zip(*data.values())
        for row in rows:
            writer.writerow(row)

class DrowsinessDataset(torch.utils.data.Dataset):
    def __init__(self,annotations,transform=None):
        self.annotations=annotations
        self.transform=transform
    def __len__(self):
        return len(self.annotations)
    def __getitem__(self,idx):
        path=self.annotations.iloc[idx, 1]
        #print(self.annotations)
        #print(path)
        #print(self.annotations.iloc[idx, 1])
        #print(self.annotations.iloc[idx, 2])
        image=cv2.imread(path)
        
        if image is None:
            raise FileNotFoundError(f"Error loading image: {self.annotations.iloc[idx, 1]}")
           
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        if self.transform:
            image=self.transform(image)
        label = self.annotations.iloc[idx, 3]
        ear=self.annotations.iloc[idx,2]
        return (image,torch.tensor(ear, dtype=torch.float),torch.tensor(label, dtype=torch.float))
        



                