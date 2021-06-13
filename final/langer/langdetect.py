# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 18:28:31 2020

@author: Sanket SS
"""
import re
import pandas as pd
import numpy as np

filename = '_cleaned.txt'
langnames = ['en','fr','it','es','de']
langs = ['English','French', 'Italian','Spanish', 'German']
#%%
for i in range(len(langnames)):
    file = str(langnames[i] +str(filename))
    f = open(file,'r',encoding='utf-8').read()
    # cleaning data
    f = re.sub(r'[\\\\/:*«`\'?¿";!<>,()|\-]', '',f.lower())
    sent = []
    count = 0
    for x in f.split('.'):
        if len(x)>25 and count<1250:
            sent.append(x)
            count+=1
    #for j in range(0,100000,100):
    #    sent.append(f[j*offset: j*offset +100])
    # list of DataFrames
    (langnames[i]) = pd.DataFrame(sent)
    (langnames[i])['language'] = langs[i]
    (langnames[i]).columns = ['sentence','language']
#%%
all_words = []
x = langnames[0]
for i in range(1,len(langnames)):
    x = x.append(langnames[i],ignore_index=True)
    
    
del langnames
#%%
print(x.shape)

#%%
X = x['sentence']
Y = x['language']
#%%
from sklearn.utils import shuffle
X, Y = shuffle(X,Y)

#%%
allwords = (' '.join([sentence for sentence in X])).split()

#%%
allwords = list(set(allwords))
#%%
from sklearn.model_selection import train_test_split
X_train, X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25)

#%%
languages = set(Y)
langs = list(languages)[:]
#%%

def create_lookup_tables(text):
    
    word_num= {word: i for i, word in enumerate(text)}
    num_word = {value:key for key, value in word_num.items()}
    
    return word_num, num_word
#%%
allwords.append('<UNK>')
#%%

vocabint , intvocab = create_lookup_tables(allwords)
langint, intlang = create_lookup_tables(languages)

#%%
def convert_to_int(data, data_int):
    """Converts all our text to integers
    :param data: The text to be converted
    :return: All sentences in ints
    """
    all_items = []
    for sentence in data:
        all_items.append([data_int[word] if word in data_int else data_int["<UNK>"] for word in sentence.split()])
        
    print(all_items)
    return all_items
#%%
    
X_test_encoded = convert_to_int(X_test, vocabint)
X_train_encoded = convert_to_int(X_train, vocabint)
#%%
# categorical coding for 
Y_data = convert_to_int(Y_test, langint)
#%%
from sklearn.preprocessing import OneHotEncoder
ohc = OneHotEncoder()
ohc.fit(Y_data)
#%%
Y_train_encoded = ohc.fit_transform(convert_to_int(Y_train, langint)).toarray()
Y_test_encoded = ohc.fit_transform(convert_to_int(Y_test, langint)).toarray()

#%%
import tensorflow as tf
import time
#%%
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence    
#%%    
dropout_factor = 0.5

#%% observe values for padding
l = []
for i in X_train_encoded:
    l.append(len(i))

print(max(l),l.index(max(l)))
#%%
maxstrlen = max(l) + 5
X_train_pad = sequence.pad_sequences(X_train_encoded, maxlen=100)
X_test_pad = sequence.pad_sequences(X_test_encoded, maxlen=100)
#%%

model = Sequential()

model.add(Embedding(len(vocabint), 300, input_length=100))
model.add(LSTM(256, return_sequences=False,activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32,activation='sigmoid'))
model.add(Dense(len(languages), activation='softmax'))

#%%    
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

#%%    
hist = model.fit(X_train_pad, Y_train_encoded, validation_data=(X_test_pad, Y_test_encoded) ,epochs=5, batch_size=512)
    #%%
def process_sentence(sentence):
    '''Removes all special characters from sentence. It will also strip out
    extra whitespace and makes the string lowercase.
    '''
    return re.sub(r'[\\\\/:*«`\'?¿";!<>,.|]', '', sentence.lower().strip())
#%%

def predict_sentence(sentence):
    """Converts the text and sends it to the model for classification
    :param sentence: The text to predict
    :return: string - The language of the sentence
    """
    if len(sentence)<75:
        print('Minimum length of string required is 75')
        return ''
    else:
        
        # Clean the sentence
        sentence = process_sentence(sentence)
        
        # Transform and pad it before using the model to predict
        x = np.array(convert_to_int([sentence], vocabint))
        x = sequence.pad_sequences(x, maxlen=100)
        
        prediction = model.predict(x)
        print(prediction[0][1])
        # Get the highest prediction
        #lang_index = np.argmax(prediction)
        #return prediction
        temp = []
        for i in range(len(prediction)):
            temp.append([langs[i],prediction[0][i]])
        return temp
        #return intlang[lang_index]

#%%
predict_sentence('Wir sind eine ganz normale Familie. Ich wohne zusammen mit meinen Eltern, meiner kleinen Schwester Lisa und unserer Katze Mick. Meine Großeltern wohnen im gleichen Dorf wie wir. Oma Francis arbeitet noch. ')
#%%

scores = model.evaluate(X_test_pad, Y_test_encoded, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

model.metrics


#%%
model.save('C:\\Users\\sahas\\Desktop\\newlang.h5')
#%%