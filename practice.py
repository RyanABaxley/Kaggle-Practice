import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import random
np.random.seed(17)

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
