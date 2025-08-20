import pyttsx3

engine = pyttsx3.init()

engine.setProperty('rate', 200)
engine.setProperty('volume', 1.0)
voices = engine.getProperty('voices')
engine.setProperty('voices', voices[0].id)
print(len(voices))

engine.say("안녕하세요")
# engine.save_to_file("welcome to my home", "save.mp3")

engine.runAndWait()