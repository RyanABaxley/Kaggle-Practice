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
