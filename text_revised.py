# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 00:44:10 2019

@author: DELL
"""

import numpy as np
from sklearn.datasets import load_files
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import re
import nltk
#from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
lemmatizer = WordNetLemmatizer()
def clean_str(string):
    string = re.sub(r"[^A-Za-z]", " ", string)
    return string.strip().lower()
def load_data_labels(datasets):
    """
    Load data and labels
    :param datasets:
    :return:
    """
    # Split by words
    x_text = datasets['data']
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels one hot encoding
    labels = []
    for i in range(len(x_text)):
        label = [0 for j in datasets['target_names']]
        label[datasets['target'][i]] = 1
        labels.append(label)
    y = np.array(labels)
    return [x_text, y]
def text_clean(text_list):
    for i in range(len(text_list)):
        words = nltk.word_tokenize(text_list[i])
        words=[word for word in words if len(word)>3]
        words = [lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
        text_list[i] = ' '.join(words)
    return text_list
#VALIDATION_SPLIT=0.2
EMBEDDING_DIM=200
MAX_SEQUENCE_LENGTH=1000
train_path='C:/Users/DELL/Desktop/Classification/new_multimodal/new_text/train'
val_path='C:/Users/DELL/Desktop/Classification/new_multimodal/new_text/val'
train_datasets = load_files(container_path=train_path,categories=None, load_content=True,encoding='utf-8', shuffle=True, random_state=42)
val_datasets = load_files(container_path=val_path,categories=None, load_content=True,encoding='utf-8', shuffle=True, random_state=42)
train_text, train_labels =load_data_labels(train_datasets)
val_text, val_labels = load_data_labels(val_datasets)
train_text = text_clean(train_text)
val_text = text_clean(val_text)
total_text=train_text+val_text    
tokenizer =Tokenizer(num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ')
tokenizer.fit_on_texts(total_text)
train_sequences = tokenizer.texts_to_sequences(train_text)
val_sequences = tokenizer.texts_to_sequences(val_text)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
train_data=pad_sequences(train_sequences, maxlen=1000, dtype='int32', padding='pre', truncating='pre', value=0.0)
val_data=pad_sequences(val_sequences, maxlen=1000, dtype='int32', padding='pre', truncating='pre', value=0.0)
print('Shape of train data tensor:', train_data.shape)
print('Shape of  train label tensor:', train_labels.shape)
print('Shape of val data tensor:', val_data.shape)
print('Shape of  train label tensor:', val_labels.shape)
#true_label=datasets['target']
indices = np.arange(train_data.shape[0])
np.random.seed(10)
np.random.shuffle(indices)
#true_label=true_label[indices]
x_train= train_data[indices]
y_train= train_labels[indices]
val_indices=np.arange(val_data.shape[0])
np.random.seed(10)
np.random.shuffle(val_indices)
x_val = val_data[val_indices]
y_val = val_labels[val_indices]
#y_true=true_label[-nb_validation_samples:]
embeddings_index = {}
glove_data = 'C:/Users/DELL/Desktop/Classification/glove.6B.200d.txt'
f = open(glove_data,encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))
#preparing a embedding matrix
embedding_matrix = np.zeros((len(word_index) + 1,EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
from keras.layers import Embedding
embedding_layer = Embedding(len(word_index) + 1,EMBEDDING_DIM,weights=[embedding_matrix],input_length=MAX_SEQUENCE_LENGTH,trainable=False)        
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.initializers import Constant
from keras import regularizers
text_input = Input(shape=(1000,), dtype='int32')
embedded_sequences = embedding_layer(text_input) #check better convolutional layer for training
conv_0 = Conv1D(256,3, padding='valid', kernel_initializer='normal', activation='relu')(embedded_sequences)
conv_1 = Conv1D(256,4, padding='valid', kernel_initializer='normal', activation='relu')(embedded_sequences)
conv_2 = Conv1D(256,5, padding='valid', kernel_initializer='normal', activation='relu')(embedded_sequences)

maxpool_0 = MaxPooling1D(998,padding='valid')(conv_0)
maxpool_1 = MaxPooling1D(997, padding='valid')(conv_1)
maxpool_2 =  MaxPooling1D(996,padding='valid')(conv_2)

concatenated_tensor = keras.layers.concatenate([maxpool_0, maxpool_1, maxpool_2],axis=1)
flatten = keras.layers.Flatten()(concatenated_tensor)
text_out= Dense(256, activation='relu')(flatten)
dropout =keras.layers.Dropout(0.5)(text_out)
preds = Dense(units=6, activation='softmax')(dropout)
model = Model(text_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])        
model.fit(x_train,y_train,epochs=25, batch_size=64,validation_data=(x_val,y_val))
model.save('text_multimodal_rev.h5')
.
#test------------------------------------------------------------------------
from keras.models import load_model
#see here----------------------------
text_model=load_model('text_multimodal_rev.h5')
loss,accuracy=text_model.evaluate(x_val,y_val, verbose=0)
print('text model loss:',loss)
print('text model accuracy:',accuracy)
#y_prob = text_model.predict(x_val)
#y_pred = y_prob.argmax(axis=-1)
#import sklearn
#from sklearn.metrics import classification_report
#target_names=datasets['target_names']
#print(classification_report(y_true, y_pred, target_names=target_names))
#print(list(datasets['target']))




    