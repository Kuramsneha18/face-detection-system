from flask import Flask, render_template, jsonify, request
from config import Config
from routes.student_routes import student_bp
from routes.admin_routes import admin_bp
from services.dlib_face_service import DlibFaceService
from services.attendance_service import AttendanceService
import cv2
import threading
import time
import numpy as np
import base64

app = Flask(__name__)
app.config.from_object(Config)
Config.init_app(app)

# Initialize services
face_service = DlibFaceService()
attendance_service = AttendanceService()

# Register blueprints
app.register_blueprint(student_bp)
app.register_blueprint(admin_bp)

@app.route('/')
def index():
    """Main page with live camera feed"""
    return render_template('index.html')

@app.route('/api/process-frame', methods=['POST'])
def process_frame():
    """Process video frame for face recognition"""
    try:
        # Get frame data from request
        frame_data = request.get_json()
        if not frame_data or 'frame' not in frame_data:
            raise ValueError("No frame data provided")

        # Decode base64 frame
        frame_bytes = base64.b64decode(frame_data['frame'])
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise ValueError("Could not decode frame")
        
        # Process frame
        recognized_faces = face_service.process_frame(frame)
        
        # Update attendance for recognized faces
        for face in recognized_faces:
            student_id = face['student_id']
            name = face['name']
            
            # Update last seen time
            attendance_service.update_last_seen(student_id)
            
            # Mark login if not already logged in
            attendance_service.mark_login(student_id, name)
        
        return jsonify({
            'success': True,
            'recognized_faces': recognized_faces
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

def check_timeouts():
    """Background task to check for session timeouts"""
    while True:
        attendance_service.check_and_update_timeouts()
        time.sleep(60)  # Check every minute

if __name__ == '__main__':
    # Start timeout checker in background thread
    timeout_thread = threading.Thread(target=check_timeouts, daemon=True)
    timeout_thread.start()
    
    # Run the Flask application
    app.run(host='0.0.0.0', port=5000, debug=True)