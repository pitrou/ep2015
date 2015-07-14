"""
Compute and plot the Mandelbrot set using matplotlib.
"""

from time import perf_counter

import math
import numpy as np
import pylab

from numba import jit, cuda

@jit
def mandel(x, y, max_iters):
    """
    Given the real and imaginary parts of a complex number,
    determine if it is a candidate for membership in the Mandelbrot
    set given a fixed number of iterations.
    """
    c = complex(x,y)
    z = 0j
    for i in range(max_iters):
        z = z*z + c
        if z.real * z.real + z.imag * z.imag >= 4:
            return 255 * i // max_iters

    return 255

@jit(nopython=True)
def create_fractal(min_x, max_x, min_y, max_y, image, iters):
    height = image.shape[0]
    width = image.shape[1]

    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height
    for x in range(width):
        real = min_x + x * pixel_size_x
        for y in range(height):
            imag = min_y + y * pixel_size_y
            color = mandel(real, imag, iters)
            image[y, x] = color


def mandel_np(min_x, max_x, min_y, max_y, image, iters):
    height = image.shape[0]
    width = image.shape[1]
    x = np.linspace(min_x, max_x, width).reshape((1, width))
    y = np.linspace(min_y, max_y, height).reshape((height, 1))
    c = x + 1j * y

    z = np.zeros_like(c)

    for i in range(iters):
        z *= z
        z += c
        image += (np.abs(z) <= 2)



mandel_gpu = cuda.jit(device=True)(mandel.py_func)


@cuda.jit#("(float32, float32, float32, float32, uint8[:,:], uint8)")
def mandel_kernel(min_x, max_x, min_y, max_y, image, iters):
    height = image.shape[0]
    width = image.shape[1]

    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height

    x, y = cuda.grid(2)
    if x < width and y < height:
        real = min_x + x * pixel_size_x
        imag = min_y + y * pixel_size_y
        image[y, x] = mandel_gpu(real, imag, iters)


xsize = 1400
ysize = 700

bsize = 16

f = mandel_np
f = create_fractal
#f = mandel_kernel[(math.ceil(xsize / bsize), math.ceil(ysize / bsize)), (bsize, bsize)]


for i in range(4):
    image = np.zeros((ysize, xsize), dtype=np.uint8)
    t = perf_counter()
    f(-2.0, 1.0, -1.0, 1.0, image, 20)
    dt = perf_counter() - t
    print(dt)

pylab.imshow(image)
pylab.gray()
pylab.show()
