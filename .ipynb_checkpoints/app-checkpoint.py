from flask import Flask, render_template, request, redirect, jsonify, session
import os
import subprocess
import time
import speech_recognition as sr
from live_speech_recognition import rt_stt
import nbformat
from nbconvert import PythonExporter
import pdb

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'aiff', 'au', 'avr', 'caf', 'flac', 'htk', 'svx', 'mat4', 'mat5', 'mpc2k', 'ogg', 'paf', 'pvf', 'raw', 'rf64', 'sd2', 'sds', 'ircam', 'voc', 'w64', 'wav', 'nist', 'wavex', 'wve', 'xi'}
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
processing = False  # Variable to track processing status
mic_on = False  # Variable to track microphone status

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/livespeech')
def livespeech():
    return render_template('live_speech.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global processing

    if processing:
        return jsonify({"error": "Processing in progress. Please wait."}), 400

    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided."}), 400
        
        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No selected file."}), 400
        
        if file and allowed_file(file.filename):
            processing = True

            if not os.path.exists('data'):
                os.makedirs('data')
            
            filepath = os.path.join('data', file.filename)
            file.save(filepath)

            time.sleep(5)

            notebook_path = os.path.join(os.path.dirname(__file__), '211-speech-to-text.ipynb')

            result = run_jupyter_notebook(notebook_path)

            return render_template('index.html', result=result)
        else:
            return render_template('index.html', result='Invalid file type! Allowed file types are: .aiff, .au, .avr, .caf, .flac, .htk, .svx, .mat4, .mat5, .mpc2k, .ogg, .paf, .pvf, .raw, .rf64, .sd2, .sds, .ircam, .voc, .w64, .wav, .nist, .wavex, .wve, .xi')
    finally:
        processing = False


@app.route('/toggle_microphone')
def toggle_microphone():
    mic_status = not session.get('mic_on', False)
    session['mic_on'] = mic_status
    return jsonify({"mic_status": mic_status})

@app.route('/get_transcription')
def get_transcription():
    global processing

    if processing:
        return jsonify({"error": "Processing in progress. Please wait."}), 400

    try:
        transcribed_text = rt_stt(2)
        if transcribed_text is not None:
            return transcribed_text
        else:
            return jsonify({"error": "Could not understand audio"}), 500

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": f"Unexpected error: {e}"}), 500
        pdb.set_trace()

    finally:
        processing = False

def run_jupyter_notebook(notebook_path):
    try:
        notebook_path = os.path.abspath(notebook_path)

        # Convert Jupyter Notebook to Python script
        subprocess.check_call(['jupyter', 'nbconvert', '--to', 'script', notebook_path])

        # Execute the generated Python script
        script_path = os.path.splitext(notebook_path)[0] + '.py'
        result = subprocess.run(['python', script_path], capture_output=True, text=True)

        if result.returncode == 0:
            # Successful execution
            transcription_path = os.path.abspath(os.path.join(os.path.dirname(notebook_path), 'transcription.txt'))

            with open(transcription_path, 'r', encoding='utf-8') as file:
                transcription = file.read()

            return transcription
        else:
            # Execution failed
            error_message = result.stderr.strip()
            return jsonify({"error": f"Error running Jupyter Notebook:\n{error_message}"}), 500

    except subprocess.CalledProcessError as e:
        # Error during subprocess.call
        return jsonify({"error": f"Error running Jupyter Notebook: {e}"}), 500
    except Exception as e:
        # Other unexpected errors
        return jsonify({"error": f"Unexpected error: {e}"}), 500




if __name__ == '__main__':
    app.run(debug=False)
