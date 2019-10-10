import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import random
np.random.seed(0)

mots = pd.read_csv('mots.txt',sep='\t')

masc = mots[mots['Étiquettes'].str.contains('mas')]
fem = mots[mots['Étiquettes'].str.contains('fem')]

def clean(word): return word.lower().replace('-','_').replace('’','_').replace(' ','_').replace("'",'_')

_masc = set(clean(mot) for mot in set(masc['Flexion'].tolist()))
_fem = set(clean(mot) for mot in set(fem['Flexion'].tolist()))

_masc_mots = list(_masc.difference(_fem))
_fem_mots = list(_fem.difference(_masc))
#still need to remove english imports

char_list = list("abcdefghijklmnopqrstuvwxyz" #standard alphabet
                 "àáâäæ" #various accent marks
                 "ç" #...
                 "èéêęë" #...
                 "îï" #...
                 "ôöœ" #...
                 "ûüùú" #...
                 "_") #unknown char

characters = sorted(list(set(letter for word in _masc_mots + _fem_mots for letter in word)))
c = [c for c in characters if c not in char_list]
m = [mot for mot in _masc_mots if any(ch in mot for ch in c)]
f = [mot for mot in _fem_mots if any(ch in mot for ch in c)]

masc_mots = [mot for mot in _masc_mots if mot not in m]
fem_mots = [mot for mot in _fem_mots if mot not in f]
characters = sorted(list(set(letter for word in masc_mots + fem_mots for letter in word)))

pad_len = len(max(masc_mots + fem_mots, key=len))
def pad(data): return np.pad(data, (0,pad_len-len(data)), 'constant', constant_values=(0))

def word_encoder(word): return pad([characters.index(l) if l in characters else 0 for l in clean(word)[::-1]])
def word_decoder(word): return ''.join([characters[i] for i in word[::-1]]).strip('_')

##def one_hot(data): return to_categorical(word_encoder(data),num_classes=len(characters))
##def one_cold(data): return word_decoder(np.argmax(data,axis=1)) #reverses one_hot()

def bin2gen(n): return 'masc' if n==1 else 'fem'

from sklearn import svm
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix


def prep_data(train_len = 2500, test_len = 150):
    np.random.seed(17)
    if train_len + test_len > min([len(masc_mots),len(fem_mots)]):
        print('Warning: Overlap of training and testing sets!')
    
    X_train = np.array([word_encoder(x).tolist() for x in masc_mots[:train_len] + fem_mots[:train_len]])
    y_train = np.array([1]*train_len + [0]*train_len)

    X_test = np.array([word_encoder(x).tolist() for x in masc_mots[-test_len:] + fem_mots[-test_len:]])
    y_test = np.array([1]*test_len + [0]*test_len)

    X_train, y_train = shuffle(X_train, y_train)

    return (X_train, y_train.flatten(), X_test, y_test.flatten())

def model(data):
    np.random.seed(0)
    X_train, y_train, X_test, y_test = data

    clf = svm.SVC(gamma='scale',random_state=0,probability=True)
    clf.fit(X_train, y_train)
    
    #print(clf.score(X_test,y_test))
    y_pred = clf.predict(X_test)
    print('Accuracy: %f%%' % (100 - 100*sum(abs(y_pred - y_test))/len(y_pred)))
    print('Confusion Matrix:\n',confusion_matrix(y_test, y_pred))
    for i in random.sample(range(len(X_test)), 10):
        word = word_decoder(X_test[i])
        print('Word:',word,
              ' '*(pad_len - len(word)),'Guess:',bin2gen(y_pred[i]),
              '\tAnswer:',bin2gen(y_test[i]))
    dely = abs(y_test - y_pred)
    y_test_list = list(y_test)
    return (clf, sorted([word_decoder(X_test[i]) for i in range(len(X_test)) if dely[i]]))#[i for i in y_test if dely[y_test_list.index(i)]])

def run_model(trials,train_len=10000//2,test_len=1000):
    print('Train:',train_len,'\nTest:',test_len,'\n')
    np.random.seed(0)
    data = prep_data(train_len,test_len)
    for trial in range(trials):
        clf, wrong = model(data)
    return (clf, wrong)
