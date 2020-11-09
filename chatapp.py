from chat_model import ChatModel as chatModel
import nltk
import pickle
import numpy as np
from keras.models import load_model
import json
import random
import utils as u

class ChatApp:

    def __init__(self):
        self.cM = chatModel()
        self._lemmatizer = nltk.stem.WordNetLemmatizer()
        self._model = load_model('chatbot_model.h5')
        self._intents = self.cM.get_intents() 
        self._words = u.load_pickle('pickles\words.pkl')
        self._classes = u.load_pickle('pickles\classes.pkl')
        
    def clean_up_sentence(self,sentence):
        # tokenize the pattern - split words into array
        sentence_words = nltk.word_tokenize(sentence)
        # stem each word - create short form for word
        sentence_words = [self._lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words

    # return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
    def bow(self, sentence, words, show_details=True):
        # tokenize the pattern
        sentence_words = self.clean_up_sentence(sentence)
        # bag of words - matrix of N words, vocabulary matrix
        bag = [0]*len(words) 
        for s in sentence_words:
            for i,w in enumerate(words):
                if w == s: 
                    # assign 1 if current word is in the vocabulary position
                    bag[i] = 1
                    if show_details:
                        print ("found in bag: %s" % w)
        return(np.array(bag))

    def predict_class(self, sentence, model):
        ERROR_THRESHOLD = 0.25
        # filter out predictions below a threshold
        p = self.bow(sentence, self._words, show_details=False)
        res = self._model.predict(np.array([p]))[0]
        
        results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
        # sort by strength of probability
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({"intent": self._classes[r[0]], "probability": str(r[1])})
        return return_list


    def getResponse(self, ints, intents_json):
        tag = ints[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if(i['tag']== tag):
                result = random.choice(i['responses'])
                break
        return result

    def chatbot_response(self, text):
        ints = self.predict_class(text, self._model)
        res = self.getResponse(ints, self._intents)
        return res



