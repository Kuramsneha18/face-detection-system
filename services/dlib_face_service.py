import dlib
import cv2
import numpy as np
from config import Config
from utils.helpers import load_json
import os

class DlibFaceService:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        self.face_rec_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
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
    
    def get_face_encoding(self, image):
        """Get face encoding from image using dlib"""
        # Convert to RGB if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = self.detector(image)
        if not faces:
            return None
            
        # Get the first face
        face = faces[0]
        
        # Get face landmarks
        shape = self.shape_predictor(image, face)
        
        # Get face encoding
        face_encoding = np.array(self.face_rec_model.compute_face_descriptor(image, shape))
        
        return face_encoding
    
    def process_frame(self, frame):
        """Process a video frame and return recognized faces"""
        # Resize frame for faster face recognition
        height, width = frame.shape[:2]
        small_frame = cv2.resize(frame, (width//4, height//4))
        
        # Get face encoding
        face_encoding = self.get_face_encoding(small_frame)
        
        recognized_faces = []
        
        if face_encoding is not None:
            # Compare with known faces
            for i, known_encoding in enumerate(self.known_face_encodings):
                # Calculate Euclidean distance
                distance = np.linalg.norm(face_encoding - known_encoding)
                
                # If distance is below threshold (similar faces)
                if distance < Config.FACE_RECOGNITION_TOLERANCE:
                    name = self.known_face_names[i]
                    student_id = self.known_student_ids[i]
                    recognized_faces.append({
                        'student_id': student_id,
                        'name': name
                    })
                    break  # Stop after first match
        
        return recognized_faces
    
    def register_new_student(self, student_id, name, image_path):
        """Register a new student with their face encoding"""
        try:
            # Load the image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not load image")
                
            # Get face encoding
            encoding = self.get_face_encoding(image)
            if encoding is None:
                raise ValueError("No face found in the image")
            
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