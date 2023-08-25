import os
import sys
import time
try:
    from kivy.graphics.svg import Svg
    from kivy.uix.relativelayout import RelativeLayout
    from kivy.properties import StringProperty, ListProperty
    from kivy.core.window import Window
    from kivy.animation import Animation

    from kivy.lang import Builder
    from kivymd.app import MDApp
    from kivymd.uix.card import MDCard
    from kivymd.uix.widget import MDWidget
    from kivymd.uix.button import MDIconButton
    import speech_recognition as sr
    import components
    from queue import Queue
    from threading import Thread
    from chatbot import Chatbot
    import pyttsx3
except Exception as e:
    print(f"Error al iniciar la IA: {e}\nInstalando las independencias.")
    os.system("pip3 install -r requirements.txt")
    print("Reiniciando en 3 segundos")
    time.sleep(3)
    os.execv(sys.executable, ['python', 'main.py'])


chat_bot = Chatbot()

retroalimentacion = False
mode = 0
question = ""
responses = []

class Chat(MDWidget):
    volume = 1
    mute = False
    messages = ListProperty([])

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.requests = Queue()
        self.tts = pyttsx3.init()
        self.tts.setProperty("rate", 150)
        self.volume = self.tts.getProperty("volume")

        def chatter():

            while True:
                item = self.requests.get()
                if item is None:
                    break

                response = chat_bot.chat(item)
                self.receive(response)

        self.background_thread = Thread(target=chatter)
        self.background_thread.start()

    def __del__(self):
        self.requests.put(None)
        self.background_thread.join()

    def send(self, text):
        if not text:
            return
        global mode, retroalimentacion, question, responses, key
        if retroalimentacion:
            print("retroalimentacion en curso, modo: ", mode)
            if mode == 1:
                question = text
                mode += 1
            elif mode == 2:
                responses.append(text)
                mode += 1
            elif mode == 3:
                if text.lower() == "ciita":
                    self.receive("Clave correcta, realizando retroalimentación.")
                    chat_bot.retroalimentacion(question, responses)
                    self.receive("Retroalimentación lista, reiniciando a Ruby en 5 segundos.")
                    time.sleep(5)
                    os.execv(sys.executable, ['python', 'main.py'])
                else:
                    self.receive("Clave incorrecta, deshaciendo los cambios.")
                mode = 0
                retroalimentacion = False
            print(text)

        self.messages.append({"text": text, "sent": True, "pos_hint": {"right": 1}})

        # scroll to bottom
        rv = self.ids.chat_rv
        box = self.ids.chat_box
        if rv.height < box.height:
            Animation.cancel_all(rv, "scroll_y")
            Animation(scroll_y=0, t="out_quad", d=0.5).start(rv)

        self.requests.put(text)

    def receive(self, text):
        if text == "Lo siento, no cuento con dicha informacion o no pude entender bien su pregunta. Intenta reformular tu pregunta o realizar una retroalimentacion.":
            print("fallo encontrado, texto: ", text)
            global retroalimentacion, mode, key
            if not retroalimentacion:
                print('retroalimentando')
                retroalimentacion = True
                if mode == 0:
                    self.messages.append({"text": "Lo siento, no cuento con dicha informacion o no pude entender bien su pregunta. Por favor vuelve a enviar tu pregunta para realizar la retroalimentación", "sent": False})
                    mode += 1
            if retroalimentacion:
                if mode == 2:
                    self.messages.append({"text": "Por favor vuelve a enviar la respuesta para realizar la retroalimentación", "sent": False})
                elif mode == 3:
                    self.messages.append({"text": "Por favor digita la clave para realizarla.", "sent": False})
        else:
            self.messages.append({"text": text, "sent": False})


    def record_message(self):
        def record_and_transcribe():
            recognizer = sr.Recognizer()
            microphone = sr.Microphone()
            with microphone as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
            self.receive("Puedes hablar...")
            with microphone as source:
                audio = recognizer.listen(source, timeout=15)
            try:
                text = recognizer.recognize_google(audio, language="es")
                response = chat_bot.chat(text)
                self.receive(response)
                self.tts.say(response)
                self.tts.runAndWait()
                self.receive("Espero que mi respuesta te sea de mucha ayuda")
            except sr.RequestError as e:
                self.receive(f"Error en la solicitud a la API de Google Speech Recognition: {e}")
            except sr.UnknownValueError:
                self.receive("Google Speech Recognition no pudo entender el audio")
            except Exception as e:
                self.receive(f"Error durante la grabación o transcripción: {str(e)}")
        recording_thread = Thread(target=record_and_transcribe)
        recording_thread.start()

    def toggle_mute(self, button: MDIconButton):
        self.mute = not self.mute
        if self.mute:
            button.text_color = MDApp.get_running_app().theme_cls.accent_color
        else:
            button.text_color = MDApp.get_running_app().theme_cls.icon_color
        self.tts.setProperty("volume", int(self.mute) * self.volume)


class ChatApp(MDApp):
    def on_stop(self):
        self.chat.__del__()

    def build(self):
        self.title = "La Rodadora"
        self.theme_cls.theme_style = "Light"
        self.theme_cls.primary_palette = "LightBlue"
        self.chat = Chat()
        return self.chat


if __name__ == "__main__":
    Window.size = (1000, 700)
    ChatApp().run()
