import matrix_concentration as mc
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
import pandas as pd

n, l, p = 30, 3, 2
print(f'Parameter (n, l, p) = {(n, l, p)}')

name = 'moments_3crossings_10'
print('File name : '+name)

sample_size = 16
print(f'Sample size = {sample_size}')

pairing_size = 10
print(f'Pairing size = {pairing_size}')
pairing_list = mc.enumerate_pairings([i for i in range(pairing_size)])

detected = []
for pairing in pairing_list:
    if len(mc.k_crossings(3, pairing)) > 0 and len(mc.k_crossings(4, pairing)) == 0:
        detected.append(pairing)
print(f'Number of selected pairings = {len(detected)}')

res = np.zeros((len(detected), sample_size))

func = lambda  pairing: mc.pairing_moment(30, 3, 2, pairing)

for i in range(sample_size):
    print(f'Sample number = {i}')
    moment_list = Parallel(n_jobs=-1, verbose=5)(delayed(func)(pairing) for pairing in detected)
    res[:,i] = np.array(moment_list)

res = res.mean(axis=1)

idx = np.argsort(res)
dict = {'pairing': [detected[i] for i in idx], 'moment': [res[i] for i in idx]}
df = pd.DataFrame(dict)
df.to_csv(name+'.csv')
plt.hist(res, bins=30)
plt.savefig('histogram_'+name+'.png')

#pairing_list = enumerate_pairings([i for i in range(8)])
#print(f'Number of pairings = {len(pairing_list)}')
#tot = 0
#func = lambda pairing: pairing_moment(30, 2, 2, pairing)
#moment_list = Parallel(n_jobs=-1, verbose=10)(delayed(func)(pairing) for pairing in pairing_list)
#for i, moment in enumerate(moment_list):
#    if moment < 0.1:
#        print(pairing_list[i])
#        tot += 1
#print(f'Number of detected pairings = {tot}')
#plt.hist(moment_list, bins=30)
#plt.show()