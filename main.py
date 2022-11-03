import matrix_concentration as mc
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
import pandas as pd

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

pairing_list = mc.enumerate_pairings([i for i in range(10)])
detected = []
for pairing in pairing_list:
    if len(mc.k_crossings(3, pairing)) > 0 and len(mc.k_crossings(4, pairing)) == 0:
        detected.append(pairing)
res = np.zeros(len(detected))
for i in tqdm(range(len(detected))):
    moment_list = Parallel(n_jobs=-1)(delayed(mc.pairing_moment)(30, 3, 2, detected[i]) for _ in range(8))
    res[i] = np.array(moment_list).mean()
idx = np.argsort(res)
dict = {'pairing': [detected[i] for i in idx], 'moment': [res[i] for i in idx]}
df = pd.DataFrame(dict)
df.to_csv('moments_3crossing_10.csv')
plt.hist(res, bins=30)
plt.savefig('histogram_moment_3crossings_10.png')