from kivymd.uix.textfield import MDTextField
from kivy.app import App

class EnterTextField(MDTextField):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def keyboard_on_key_down(self, window, keycode, text, modifiers):
        key, key_str = keycode
        if key == 13:
            if "shift" in modifiers:
                self.insert_text('\n')
            else:
                self.dispatch('on_text_validate')
        else:
            super().keyboard_on_key_down(window, keycode, text, modifiers)