# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 11:33:17 2019

@author: DELL
"""
import keras
import numpy as np
#from sklearn.datasets import load_files
#image-------------------------------------
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Input, GlobalMaxPooling2D
from keras.layers import Conv2D, MaxPooling2D, Embedding,Activation,Flatten
img_train_path='C:/Users/DELL/Desktop/Classification/new_multimodal/new_image/train'
img_val_path='C:/Users/DELL/Desktop/Classification/new_multimodal/new_image/val'
img_height=299
img_width=299
train_datagen = ImageDataGenerator(
 rescale=1./255,
 shear_range=0.2,
 zoom_range=0.2,
 horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(
    img_train_path,
    target_size=(img_height, img_width),
    batch_size=4663,shuffle=False,
    class_mode='categorical')
x_train_image,y_train_image=train_generator.next()
indices = np.arange(x_train_image.shape[0])
np.random.seed(10)
np.random.shuffle(indices)
x_train_image= x_train_image[indices]
y_train_image= y_train_image[indices]
val_datagen = ImageDataGenerator(rescale=1./255)
val_generator=val_datagen.flow_from_directory(img_val_path,
                                            target_size = (img_height,img_width),
                                            batch_size =1168,shuffle=False,
                                            class_mode = 'categorical')
x_val_image, y_val_image = val_generator.next()
val_indices = np.arange(x_val_image.shape[0])
np.random.seed(10)
np.random.shuffle(val_indices)
x_val_image= x_val_image[val_indices]
y_val_image= y_val_image[val_indices]

from keras.layers import Input
input_img = Input(shape = (299,299, 3),dtype='float32')
from keras.layers import Conv2D, MaxPooling2D
tower_1 = Conv2D(256, (1,1), padding='same', activation='relu')(input_img)
tower_1 = Conv2D(256, (3,3), padding='same', activation='relu')(tower_1)
tower_2 = Conv2D(256, (1,1), padding='same', activation='relu')(input_img)
tower_2 = Conv2D(256, (5,5), padding='same', activation='relu')(tower_2)
tower_3 = MaxPooling2D((3,3), strides=(1,1), padding='same')(input_img)
tower_3 = Conv2D(256, (1,1), padding='same', activation='relu')(tower_3)
output = keras.layers.concatenate([tower_1, tower_2, tower_3], axis = 3)
from keras.layers import Flatten, Dense
img_out = Flatten()(output)
out    = Dense(6, activation='softmax')(img_out)
#text---------------------------------------------------------
from sklearn.datasets import load_files
text_train_path='C:/Users/DELL/Desktop/Classification/new_multimodal/new_text/train'
text_val_path='C:/Users/DELL/Desktop/Classification/new_multimodal/new_text/val'
train_datasets = load_files(container_path=text_train_path,categories=None, load_content=True,encoding='utf-8', shuffle=False, random_state=42)
val_datasets = load_files(container_path=text_val_path,categories=None, load_content=True,encoding='utf-8', shuffle=False, random_state=42)
#import keras
#from keras.preprocessing.text import Tokenizer
#from keras.preprocessing.sequence import pad_sequences
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
#EMBEDDING_DIM=200
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
EMBEDDING_DIM=200
MAX_SEQUENCE_LENGTH=1000
train_text,train_labels=load_data_labels(train_datasets)   
val_text, val_labels = load_data_labels(val_datasets)
train_text=text_clean(train_text)
val_text = text_clean(val_text)
total_text=train_text+val_text    
tokenizer =Tokenizer(num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ', char_level=False, oov_token=None, document_count=0)
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
#np.random.seed(10)
#np.random.shuffle(val_indices)
indices = np.arange(train_data.shape[0])
np.random.seed(10)
np.random.shuffle(indices)
x_train_text= train_data[indices]
y_train_text= train_labels[indices]
val_indices=np.arange(val_data.shape[0])
np.random.seed(10)
np.random.shuffle(val_indices)
#x_val = val_data[val_indices]
x_val_text = val_data[val_indices]
y_val_text = val_labels[val_indices]
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
 #-------------df---------------------------------------
#model=(text_acc)*(text_model)+(img_acc)*(img_model)
#from keras.models import load_model
#text_model=load_model('text_multimodal_rev.h5')
#text_loss,text_acc=text_model.evaluate(x_val_text,y_val_text, verbose=0)
#img_model=load_model('image_classification.h5')
#img_loss,img_acc=image_model.evaluate(x_val_img,y_val_img,verbose=0)
 
#final=preds*text_acc+out*img_acc
final=keras.layers.Add()([out,preds])
#preds+out
final=Dense(units=6, activation='softmax')(final)
model = Model([input_img,text_input], final)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc']) 
train=([x_train_image,x_train_text],y_train_image)
val=([x_val_image,x_val_text],y_val_image)
model.fit([x_train_image,x_train_text],y_train_image,epochs=40, batch_size=64,validation_data=([x_val_image,x_val_text],y_val_image))
model.save('multimodal_df.h5')
#---------------------df report---------------------------------------------------------------------------
from keras.models import load_model
multi_model=load_model('multimodal_df.h5')
loss,accuracy=multi_model.evaluate([x_val_image,x_val_text],y_val_image, verbose=0)
print('decision fusion model loss:',loss)
print('decision fusion model accuracy:',accuracy)









