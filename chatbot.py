import random
import json
import pickle
import unicodedata
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

from pathlib import Path

from train import train

parent_dir  = Path(__file__).parent

def lector(lenguaje: str = "es"):
    if lenguaje == "es":
        jsons = 'data_es.json'
    else:
        jsons = 'data_en.json'
    with open(parent_dir / jsons, encoding='utf-8') as file:
        intents = json.load(file)
    return intents

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

class Chatbot:
    def __init__(self, language: str = "es"):
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        self.lemmatizer = WordNetLemmatizer()
        self.retroalimentacion_status = False
        self.language = language
        if self.language == "es":
            self.model = load_model(parent_dir / 'chat_es.h5')
        else:
            self.model = load_model(parent_dir / 'chat_en.h5')
        print("Encendido.")

    def clean_up_sentence(self, sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [self.lemmatizer.lemmatize(word) for word in sentence_words]
        return sentence_words

    def bag_of_word(self, sentence, words):
        sentence_words = self.clean_up_sentence(sentence)
        bag = [0] * len(words)
        for w in sentence_words:
            for i, word in enumerate(words):
                if word == w:
                    bag[i] = 1
        return np.array(bag)

    def predict_class(self, sentence):
        if self.language == "es":
            classe = 'classes_es.pkl'
            word = 'words_es.pkl'
        else:
            classe = 'classes_en.pkl'
            word = 'words_en.pkl'
        print(self.model)
        classes = pickle.load(open(parent_dir / classe, 'rb'))
        words = pickle.load(open(parent_dir / word, 'rb'))
        msg = ''.join((c for c in unicodedata.normalize('NFD', sentence) if unicodedata.category(c) != 'Mn'))
        bow = self.bag_of_word(msg.lower(), words)
        res = self.model.predict(np.array([bow]))[0]
        ERROR_THRESHOLD = 0.4
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
        return return_list

    def retroalimentacion(self, text: str, responses: list):
        intents = lector(self.language)
        datas = intents['intents']
        dicts = {'tag': text, 'patterns': [text], 'responses': responses}
        datas.append(dicts)
        dicts = {"intents": datas, "error": intents['error']}
        if self.language == "es":
            jsons = 'data_es.json'
        else:
            jsons = 'data_en.json'
        with open(parent_dir / jsons,'w' ,encoding='utf-8') as file:
            json.dump(dicts,file, indent=4)
            file.close()
        self.retroalimentacion_status = True
        train()

    def chat(self, message: str):
        # try:
            intents = lector(self.language)
            ints = self.predict_class(message)
            if ints:
                return get_response(ints, intents)
            else:
                return intents['error']
        # except Exception as e:
        #     print(f"Error en funcion chatbot: {e}")