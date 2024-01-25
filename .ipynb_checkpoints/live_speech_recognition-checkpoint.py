import speech_recognition as sr

def get_all_your_mic():
    microphones = sr.Microphone.list_microphone_names()
    print("Available Microphones:")
    for i, mic in enumerate(microphones):
        print(f"{i}: {mic}")

def rt_stt(microphone_index):
    recognizer = sr.Recognizer()

    # get all your mics
    microphones = sr.Microphone.list_microphone_names()

    if 0 <= microphone_index < len(microphones):
        selected_microphone = microphones[microphone_index]
        print(f"Selected Microphone: {selected_microphone}")

    # use mic as the source for your voice
    with sr.Microphone(device_index=microphone_index) as source:
        print("paiseh, adjusting the ambient voice...")
        recognizer.adjust_for_ambient_noise(source)

        print("lai lai talk now")

        # error handling
        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio)
            print("You said:", text)
            return text

        except sr.UnknownValueError:
            print("woi, cannot understand your voice la deh")
            return None

        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
            return None

if __name__ == "__main__":
    get_all_your_mic()
    selected_mic = 2
    rt_stt(selected_mic)
