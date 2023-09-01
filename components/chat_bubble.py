from kivy.lang import Builder
from kivy.properties import BooleanProperty

from components.label import PLabel

Builder.load_string(
    """
<ChatBubble>
    adaptive_height: True
    padding: [dp(8), dp(8)]
    text_color: 1, 1, 1, 1
    text_size: self.width, None

    canvas.before:
        Color:
            rgba:
                app.theme_cls.primary_dark if self.sent \
                else app.theme_cls.primary_color
        RoundedRectangle:
            size: self.size
            pos: self.pos
            radius:
                [dp(8), dp(8), (dp(-5), dp(5)), dp(8)] if self.sent \
                else [dp(8), dp(8), dp(8), (dp(-5), dp(5))]
"""
)


class ChatBubble(PLabel):
    sent = BooleanProperty()

    def get_pos_hint(self):
        return {"right": 1 if self.sent else 0}