import face_recognition
import cv2
import numpy as np
from config import Config
from utils.helpers import load_json
import os

class FaceRecognitionService:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_student_ids = []
        self.load_known_faces()
    
    def load_known_faces(self):
        """Load known face encodings from students.json"""
        students = load_json(Config.STUDENTS_JSON)
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_student_ids = []
        
        for student in students:
            self.known_face_encodings.append(np.array(student['encoding']))
            self.known_face_names.append(student['name'])
            self.known_student_ids.append(student['student_id'])
    
    def process_frame(self, frame):
        """Process a video frame and return recognized faces"""
        # Resize frame for faster face recognition
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        
        # Convert BGR to RGB
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Find faces in frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        recognized_faces = []
        
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(
                self.known_face_encodings, 
                face_encoding,
                tolerance=Config.FACE_RECOGNITION_TOLERANCE
            )
            
            if True in matches:
                first_match_index = matches.index(True)
                name = self.known_face_names[first_match_index]
                student_id = self.known_student_ids[first_match_index]
                recognized_faces.append({
                    'student_id': student_id,
                    'name': name
                })
        
        return recognized_faces
    
    def register_new_student(self, student_id, name, image_path):
        """Register a new student with their face encoding"""
        try:
            # Load and encode the face
            image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(image)
            
            if not face_encodings:
                raise ValueError("No face found in the image")
            
            # Get the first face encoding
            encoding = face_encodings[0]
            
            # Load existing students
            students = load_json(Config.STUDENTS_JSON)
            
            # Add new student
            student_data = {
                "student_id": student_id,
                "name": name,
                "encoding": encoding.tolist()
            }
            
            students.append(student_data)
            
            # Save updated students list
            from utils.helpers import save_json
            if save_json(Config.STUDENTS_JSON, students):
                # Update in-memory encodings
                self.load_known_faces()
                return True
                
            return False
            
        except Exception as e:
            print(f"Error registering student: {str(e)}")
            return False