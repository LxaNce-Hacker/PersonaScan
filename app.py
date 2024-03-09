# created by : LxaNce

from flask import Flask, render_template, jsonify, request
import subprocess
import json

app = Flask(__name__)

# Define a global variable for storing attendance log
attendance_log = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recognize_faces')
def recognize_faces():
    try:
        # Use subprocess to run your recognition script
        result = subprocess.run(['python', 'recognizer_info.py'], capture_output=True, text=True)

        if result.returncode == 0:
            # Successful recognition
            return jsonify({'message': result.stdout, 'attendance': attendance_log})
        else:
            # Recognition error
            return jsonify({'error': result.stderr})

    except Exception as e:
        # Other unexpected errors
        return jsonify({'error': str(e)})

@app.route('/capture_faces')
def capture_faces():
    try:
        person_name = request.args.get('name')
        result = subprocess.run(['python', 'capture_images.py', person_name], capture_output=True, text=True)

        return jsonify({'message': 'Faces captured successfully'})

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/train_faces')
def train_faces():
    try:
        result = subprocess.run(['python', 'train_face_recognition.py'], capture_output=True, text=True)

        return jsonify({'message': 'PersonaScan trained successfully'})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=5000, debug=True)












































# created by : LxaNce
# created by : LxaNce
# created by : LxaNce
# created by : LxaNce
# created by : LxaNce
