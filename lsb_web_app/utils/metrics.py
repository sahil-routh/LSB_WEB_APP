import numpy as np
import math

def mse(img1, img2):
    return np.mean((img1.astype(float) - img2.astype(float)) ** 2)

def psnr(img1, img2):
    m = mse(img1, img2)
    if m == 0:
        return float("inf")
    return 10 * math.log10((255 ** 2) / m)
