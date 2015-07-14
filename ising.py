
from concurrent.futures import ThreadPoolExecutor
import itertools
from math import exp, log, e, sqrt
import time

import numba
import numpy as np

kT = 2 / log(1 + sqrt(2), e)

#@numba.jit(nopython=True)
def update_one_element(x, i, j):
    n, m = x.shape
    assert n > 0
    assert m > 0
    dE = 2 * x[i, j] * (
                     x[(i-1)%n, (j-1)%m]
                   + x[(i-1)%n,  j     ]
                   + x[(i-1)%n, (j+1)%m]

                   + x[ i     , (j-1)%m]
                   + x[ i     , (j+1)%m]

                   + x[(i+1)%n, (j-1)%m]
                   + x[(i+1)%n,  j     ]
                   + x[(i+1)%n, (j+1)%m]
                   )
    if dE <= 0 or exp(-dE / kT) > np.random.random():
        x[i, j] = -x[i, j]

#@numba.jit(nopython=True, nogil=True)
def update_one_frame(x, n, m):
    for i in range(n):
        for j in range(0, m, 2):  # Even columns first to avoid overlap
            update_one_element(x, j, i)
    for i in range(n):
        for j in range(1, m, 2):  # Odd columns second to avoid overlap
            update_one_element(x, j, i)


@numba.jit(nopython=True, nogil=True)
def update_one_block(x, i, m):
    for j in range(0, m, 2):  # Even columns first to avoid overlap
        update_one_element(x, j, i)
    for j in range(1, m, 2):  # Odd columns second to avoid overlap
        update_one_element(x, j, i)


if __name__ == '__main__':
    x = np.random.randint(2, size=(200, 200)).astype('i1')
    x[x == 0] = -1
    clock = time.perf_counter
    n, m = x.shape
    exe = ThreadPoolExecutor(max_workers=4)
    for i in range(15):
        t1 = clock()
        if 1:
            update_one_frame(x, n, m)
        else:
            it = exe.map(update_one_block, itertools.repeat(x),
                         range(n), itertools.repeat(m))
            list(it)
        t2 = clock()
        print(t2 - t1)


