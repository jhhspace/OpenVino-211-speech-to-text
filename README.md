# OpenVino-211-speech-to-text
ITE College West X Busan Computer Science High School Social Impact Project

## About this project
This project was selected by the CEO and CTO of BrainAI Co. Ltd for showcasing to Busan Computer Science High School and ITE College West Principal and Management.

Our team, Team HoYo, created this AI Application for a sole purpose: Elevating social interactions for individuals with hearing impairment.
<br>Why?
This Speech to Text AI acts as a budget friendly substitute for conventional hearing aids. The average amount for a hearing aid is about $650 to $7000 ([Source](https://www.sgh.com.sg/patient-care/specialties-services/hearing-aid-devices), General question 7), and this AI is free for use. Thanks to Intel OpenVINO pretrained model, we managed to make this a possibility.
<br>
### Data
- In 2022, South Korea recorded the highest number of people with hearing disabilities since 2003, reaching a whopping __425.2 thousand__, with a continuous upwards trend since 2015.
- As of 13 September 2022, around __500,000__ Singaporeans experienced hearing impairment and permanent deaf.
- Currently, __430 million__ people needs rehabilitation due to just hearing loss.
- It is also predicted that __by 2050__, __700 million (1/10 people) will have hearing impairment.
[Source](https://www.who.int/news-room/fact-sheets/detail/deafness-and-hearing-loss)

# Pre-requisite
- Anaconda Prompt: Run the commands below after starting the prompt with administrator
    - (base) C:\WINDOWS\system32> cd\
    - (base) C:\ mkdir BrainAI
    - (base) C:\ cd BrainAI
    - (base) C:\BrainAI> conda create --name BrainAI python=3.11 anaconda
    - (base) C:\BrainAI> conda activate BrainAI
    - (BrainAI) C:\BrainAI>

- Installing packages: Do it in the anaconda prompt after you're in the BrainAI Environment
    - (BrainAI) C:\BrainAI> python.exe -m pip install --upgrade --user pip
    - (BrainAI) C:\BrainAI> pip install openvino
    - (BrainAI) C:\BrainAI> pip install opencv-python
    - (BrainAI) C:\BrainAI> pip install flask
    - (BrainAI) C:\BrainAI> pip install flask_wtf
    - (BrainAI) C:\BrainAI> jupyter lab

# How to set microphone
- Run `python live_speech_recognition.py` and find your microphone number
- Head to `app.py` line __**81**__ and replace the __**2**__ with your microphone number
- Head to `live_speech_recognition.py` line __**43**__ and replace the variable `selected_mic` to your microphone number

# How to run?
- Go to the file `211-speech-to-text.ipynb`. uncomment the first cell and run it. Afterwards, comment the first cell again.
- Open Command Prompt and run `python app.py`, go to the link shown from Flask.
- Enjoy.

# Bugs/Issues
- Please open a bug or issue report if there's any found, this project was made from 23/1/24 - 25/1/24 during ITE College West X Busan Computer Science High School exchange.

# Credits
Our Facilitator, [Irene Wong](https://www.linkedin.com/in/irene-wong-56a88b237/)

[BrainAI Co. Ltd](https://brainai.kr/)

[Busan Computer Science High School](https://school.busanedu.net/pcs-h/main.do)

[ITE College West](https://www.ite.edu.sg/colleges/ite-college-west/)

[Intel OpenVINO](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html)
