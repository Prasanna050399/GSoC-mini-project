#from skimage.measure import structural_similarity as ssim
from skimage import measure as ms
import matplotlib.pyplot as plt
import numpy as np
import cv2

def mse(I1,I2):
    err=np.sum((I1.astype("float")-I2.astype("float"))**2)
    err/=float(I1.shape[0]*I1.shape[1])
    return err

def comp_image(I1,I2):
    m=mse(I1,I2)
    s=ms.compare_ssim(I1,I2)
    return m,s

new_image=cv2.imread("captured.jpg",0)
base_image=cv2.imread("base.jpg",0)

#new_image=cv2.cvtColor(new_image,cv2.COLOR_BGR2GRAY)
#base_image=cv2.cvtColor(base_image,cv2.COLOR_BGR2GRAY)

mean_sq_er,SSIM=comp_image(base_image,new_image)

print(mean_sq_er,SSIM)
