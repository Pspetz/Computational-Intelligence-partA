

from tensorflow import keras
import tensorflow as tf
from sklearn.model_selection import KFold
from keras.layers import Dense
from keras import backend as K
from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM
from keras.layers import GlobalMaxPooling1D
from keras.models import Model
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.layers import Input
from keras.layers.merge import Concatenate
import pandas as pd
import numpy as np
import re
from matplotlib import pyplot as plt



#TRAIN DATA
path = 'train-data.dat'


clean_files = []
df = pd.DataFrame()

file = open(path).readlines()
len(file)


X_train=[]
for i in range(len(file)):
    sentence_vectors=re.sub('<\d+>','',file[i])
    X_train.append(sentence_vectors)


    
        
#TEST-DATA
path = 'test-data.dat'

clean_files = []
df = pd.DataFrame()

file1 = open(path).readlines()
len(file)
           

X_test=[]
for i in range(len(file1)):
    sentence_vectorss=re.sub('<\d+>','',file[i])
    X_test.append(sentence_vectorss)

tokenizer = Tokenizer(num_words=8251)
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

vocab_size = len(tokenizer.word_index) + 1

maxlen = 80

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

labels_fnames = [
            'train-label.dat',
            'test-label.dat'
            ]

Y_train = pd.read_csv(labels_fnames[0] , delimiter = ' ', header = None)
Y_test= pd.read_csv(labels_fnames[1], delimiter = ' ', header = None)

from numpy import array
from numpy import asarray
from numpy import zeros

embeddings_dictionary = dict()

glove_file = open("vocabs.txt", encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary[word] = vector_dimensions
glove_file.close()

embedding_matrix = zeros((vocab_size, 100))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector



print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)



def create_model():
    deep_inputs = Input(shape=(maxlen,))
    embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], trainable=False)(deep_inputs)
    LSTM_Layer_1 = LSTM(512)(embedding_layer)
    LSTM_Layer_2 = LSTM(256)(embedding_layer)
    dense_layer_1 = Dense(20, activation='sigmoid')(LSTM_Layer_2)
    model = Model(inputs=deep_inputs, outputs=dense_layer_1)
    opt = tf.keras.optimizers.SGD(learning_rate=0.001, momentum = 0)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model 
def create_model_mse():
    deep_inputs = Input(shape=(maxlen,))
    embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], trainable=False)(deep_inputs)
    LSTM_Layer_1 = LSTM(512)(embedding_layer)
    LSTM_Layer_2 = LSTM(256)(embedding_layer)
    dense_layer_1 = Dense(20, activation='sigmoid')(LSTM_Layer_2)
    model = Model(inputs=deep_inputs, outputs=dense_layer_1)
    opt = tf.keras.optimizers.SGD(learning_rate=0.001, momentum = 0)
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
    return model 

def evaluate_model(X_train_new,Y_train,X_test_new,Y_test):            
    fold_number2=0
    fold_number = 0
    sum_of_acc2 =0
    sum_of_loss2 = 0
    sum_of_acc=0
    sum_of_loss=0
    losses,scores,histories = list(),list(),list()
    losses2,scores2,histories2 = list(),list(),list()
    kfold = KFold(n_splits=5, shuffle=False, random_state=None)
    epochs = 30
    
    for train_index, test_index in kfold.split(X_train_new,Y_train):  
        shallow_mlp_model = create_model()
        mse_model = create_model_mse()
        #es=EarlyStopping(monitor='val_loss' , mode='min' , verbose=1)
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X_train_new[train_index,:], X_train_new[test_index,:]
        y_train, y_test = Y_train.iloc[train_index],Y_train.iloc[test_index]
    
        #MODEL for cross-entropy

        history = shallow_mlp_model.fit(X_train_new[train_index,:],Y_train.iloc[train_index] , epochs=epochs ,batch_size=512, validation_data=(X_test, y_test) ,verbose=1)
        loss, val_acc = shallow_mlp_model.evaluate(X_test_new,Y_test,verbose=1)

        #MODEL 2 for Mse
       
        history2 = mse_model.fit(X_train_new[train_index,:],Y_train.iloc[train_index] , epochs=epochs ,batch_size=128, validation_data=(X_test, y_test)  ,verbose=1)
        loss2, val_acc2 = mse_model.evaluate(X_test_new,Y_test,verbose=1)
    
    


        print("-"*80)
        ###########################
        fold_number +=1 
        fold_number2 +=1
        ##########################
        print(" for cross entropy fold",(fold_number),"\n|  loss:" , loss, "Accuracy:",val_acc)
        print(" for Mse fold",(fold_number2),"\n|  loss:" , loss2, "Accuracy:",val_acc2)
        ##########################
        sum_of_acc += val_acc
        sum_of_loss += loss
        #########################
        sum_of_loss2 +=loss2
        sum_of_acc2 += val_acc2

        scores.append(val_acc)
        histories.append(history)
        scores2.append(val_acc2)
        histories2.append(history2)

        print("-"*80)
        print("\n Cross-Entropy:the average of the loss and acc is: \n","loss:" , sum_of_loss/fold_number, "\n" , "Accuracy" , sum_of_acc/fold_number,"\n")
        print("\n MSE:the average of the lose and acc is: \n","loss:" , sum_of_loss2/fold_number2, "\n" , "Accuracy" , sum_of_acc2/fold_number2,"\n")
        
    return history,history2


    
def create_model_plots(history,history2):
        plt.figure(0)
        plt.subplot(2, 2, 1)
        plt.plot(history.history['accuracy'], label='Accuracy (train)')
        plt.plot(history.history['val_accuracy'], label='Accuracy (test)')
        plt.title("Accuracy with Cross Entropy loss")
        plt.ylabel("Accuracy")
        plt.xlabel("Epochs")
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(history2.history['accuracy'], label='Accuracy (train)')
        plt.plot(history2.history['val_accuracy'], label='Accuracy (test)')
        plt.title("Accuracy with MSE loss")
        plt.ylabel("Accuracy")
        plt.xlabel("Epochs")
        plt.tight_layout()
        plt.show()

    # plot the cross entropy loss
        plt.figure(1)
        plt.subplot(2, 2, 1)
        plt.plot(history.history['loss'], label='Cross entropy (train)')
        plt.plot(history.history['val_loss'], label='Cross entropy (test)')
        plt.title('Cross Entropy Evaluated')
        plt.xlabel('Epochs')
        plt.ylabel('Error value')
        plt.legend()

    # plot the mse loss
        plt.subplot(2, 2, 2)
        plt.plot(history2.history['loss'], label='MSE (train)')
        plt.plot(history2.history['val_loss'], label='MSE (test)')

        plt.title('MSE Evaluated')
        plt.xlabel('Epochs')
        plt.ylabel('Error value')
        plt.legend()
        plt.tight_layout()
        plt.show()

def run_test():
   h1,h2 = evaluate_model(X_train,Y_train,X_test,Y_test)
   create_model_plots(h1,h2)
run_test()









