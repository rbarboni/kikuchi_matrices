import matrix_concentration as mc
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import os
from tqdm import tqdm

plt.rcParams.update({"pdf.fonttype": 42, 'text.usetex': True})
r = 0.5
fig, ax = plt.subplots(figsize=(15 * r, 8 * r))

pairing1 = [(0, 1), (2, 3), (4, 5)]
pairing2 = [(0, 2), (1, 3), (4, 5)]
pairing3 = [(0, 2), (1, 4), (3, 5)]
pairing4 = [(0, 3), (1, 4), (2, 5)]

pairing_list = [pairing1, pairing2, pairing3, pairing4]

l = 2
p = 2
n_list = np.arange(5, 31, 5)

res = np.zeros((len(n_list), 4))

for i, n in enumerate(n_list):
    print(f'n = {n}')
    for j, pairing in tqdm(enumerate(pairing_list), leave=False):
        func = lambda: mc.pairing_moment(n, l, p, pairing)
        moment_list = Parallel(n_jobs=-1, verbose=False)(delayed(func)() for _ in range(4*os.cpu_count()))
        res[i, j] = np.array(moment_list).mean()

ax.plot(n_list, res, label=pairing_list)
#ax.set_xlabel('n')
ax.set_ylabel('$\lambda(\pi) = \sigma^{-6} m(\pi)$')
ax.legend()
#fig.savefig('pairing_l2_p2_n50.pdf', bbox_inches='tight', pad_inches=0., )
fig.show()

