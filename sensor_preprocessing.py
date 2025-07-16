from lsm6ds3 import LSM6DS3
import time
import numpy as np

lsm = LSM6DS3()
sent = 0.000061

def read_acc():
	ax, ay, az, gx, gy, gz = lsm.get_readings()
	ax = sent * ax
	ay = sent * ay
	az = sent * az
	
	return ax, ay, az

def moving_avg(data, window_size=3):
	kernel = np.ones(window_size) / window_size
	padded_data = np.convolve(data, kernel, mode='same')
	return padded_data

