# FaceAttendAI - Automated Face Recognition Attendance System

FaceAttendAI is a production-ready Python web application that automatically performs face recognition attendance tracking using real-time camera feed. The system detects student faces automatically without requiring any manual clicks and records attendance in local JSON files.

## Features

- 🎥 Real-time face detection and recognition
- 📊 Automatic attendance tracking with login/logout times
- 💾 Local JSON storage (no database required)
- 🔄 Automatic session management
- 📱 Responsive web interface
- 📈 Admin dashboard with attendance analytics
- 👤 Student dashboard with personal attendance history
- 📄 Export attendance records to CSV

## Prerequisites

- Python 3.8 or higher
- Webcam
- Modern web browser

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/face-detection-system.git
cd face-detection-system
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Start the application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

## Registering a New Student

1. Navigate to Admin Dashboard and click "Register Student"
2. Fill in the student details:
   - Student ID
   - Full Name
   - Upload a clear face photo
3. Submit the form
4. The system will automatically extract and store face encodings

## How Automatic Login/Logout Works

1. **Login Process:**
   - When a registered face is detected in the camera feed
   - System automatically records login time
   - Visual and audio confirmation is provided
   
2. **Active Session:**
   - System continuously monitors presence
   - Updates "last seen" timestamp
   
3. **Automatic Logout:**
   - If face not detected for 5+ minutes
   - If session exceeds 8 hours
   - When browser/tab is closed

## Exporting Attendance Records

1. Go to Admin Dashboard
2. Use filters to select desired records:
   - Date range
   - Specific student
3. Click "Export to CSV"
4. Save the downloaded file

## Project Structure

```
FaceAttendAI/
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── config.py             # Configuration settings
│
├── /data/                # JSON storage
│   ├── students.json     # Student records
│   └── attendance.json   # Attendance logs
│
├── /services/            # Business logic
├── /routes/             # URL routing
├── /templates/          # HTML templates
├── /static/             # Assets (CSS, JS, images)
└── /utils/              # Helper functions
```

## Configuration

Key settings in `config.py`:

- `AUTO_LOGOUT_TIME`: Maximum session duration (default: 8 hours)
- `FACE_TIMEOUT`: Time until auto-logout when face missing (default: 5 minutes)
- `FACE_RECOGNITION_TOLERANCE`: Recognition strictness (default: 0.6)
- `TIMEZONE`: Local timezone for timestamps (default: 'Asia/Kolkata')

## Security Notes

1. Keep `students.json` and `attendance.json` backed up
2. Use HTTPS in production
3. Implement user authentication for admin access
4. Regular backups of attendance data recommended

## License

MIT License - feel free to use for any purpose

## Support

For issues and feature requests, please create an issue on GitHub.