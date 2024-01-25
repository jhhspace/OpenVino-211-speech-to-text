# OpenVino-211-speech-to-text
ITE College West X Busan Computer Science High School Social Impact Project

# Pre-requisite
- Anaconda Prompt: Run the commands below after starting the prompt with administrator
    - (base) C:\WINDOWS\system32>cd\
    - (base) C:\mkdir BrainAI
    - (base) C:\cd BrainAI
    - (base) C:\BrainAI>conda create --name BrainAI python=3.11 anaconda
    - (base) C:\BrainAI>conda activate BrainAI
    - (BrainAI) C:\BrainAI>

- Installing packages: Do it in the anaconda prompt after you're in the BrainAI Environment
    - (BrainAI) C:\BrainAI>python.exe -m pip install --upgrade --user pip
    - (BrainAI) C:\BrainAI>pip install openvino
    - (BrainAI) C:\BrainAI>pip install opencv-python
    - (BrainAI) C:\BrainAI>pip install flask
    - (BrainAI) C:\BrainAI>pip install flask_wtf
    - (BrainAI) C:\BrainAI>jupyter lab

# How to set microphone
- Run `live_speech_recognition.py` and find your microphone number
- Head to `app.py` line __**81**__ and replace the __**2**__ with your microphone number
- Head to `live_speech_recognition.py` line __**43**__ and replace the variable `selected_mic` to your microphone number

# How to run?
- Go to the file `211-speech-to-text.ipynb`. uncomment the first cell and run it. Afterwards, comment the first cell again.
- Open Command Prompt and run `python app.py`, go to the link shown from Flask.
- Enjoy.

# Bugs/Issues
- Please open a bug or issue report if there's any found, this project was made from 23/1/24 - 25/1/24 during ITE College West X Busan Computer Science High School exchange.

# Credits
[BrainAI Co. Ltd](https://brainai.kr/)

[Busan Computer Science High School](https://school.busanedu.net/pcs-h/main.do)

[ITE College West](https://www.ite.edu.sg/colleges/ite-college-west/)
