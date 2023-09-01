import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

from pathlib import Path

parent_dir  = Path(__file__).parent

def train():
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    lemmatizer = WordNetLemmatizer()
    with open(parent_dir / 'data.json', encoding='utf-8') as file:
        intents = json.load(file)
    words = []
    classes = []
    documents = []
    ignore_letters = ['!', '?', '.', ',']
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            word_list = nltk.word_tokenize(pattern)
            words.extend(word_list)
            documents.append((word_list, intent['tag']))
            if intent['tag'] not in classes:
                classes.append(intent['tag'])
    words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
    words = sorted(set(words))
    classes = sorted(set(classes))
    pickle.dump(words, open(parent_dir / 'words.pkl', 'wb'))
    pickle.dump(classes, open(parent_dir / 'classes.pkl', 'wb'))
    training = []
    output_empty = [0] * len(classes)
    for document in documents:
        bag = []
        word_patterns = document[0]
        word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
        for word in words:
            bag.append(1) if word in word_patterns else bag.append(0)
        output_row = list(output_empty)
        output_row[classes.index(document[1])] = 1
        training.append([bag, output_row])
    random.shuffle(training)
    training = np.array(training, dtype=object)
    train_x = list(training[:, 0])
    train_y = list(training[:, 1])
    model = Sequential()
    model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_y[0]), activation='softmax'))
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.fit(np.array(train_x), np.array(train_y), epochs=100, batch_size=8, verbose=1)
    model.save(parent_dir / 'chat.h5')

if __name__ == "__main__":
    train()