from gtts import gTTS
from io import BytesIO
import pygame 
import threading

def speak(text):
    tts = gTTS(text=text, lang='en')
    fp = BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    pygame.mixer.init()
    pygame.mixer.music.load(fp)
    pygame.mixer.music.play()

def audio_thread(text):
    thread = threading.Thread(target=speak, args=(text,))
    thread.start()