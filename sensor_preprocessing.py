from lsm6ds3 import LSM6DS3
import time
import numpy as np
import math

lsm = LSM6DS3()
sent = 0.00006

def raw_to_dps(raw_val, sent=8.75):
	return raw_val/sent

def dps_to_rads(dps):
	return dps * (math.pi / 180)
	
def raw_to_rads(raw_val, sent=8.75):
	return dps_to_rads(raw_to_dps(raw_val, sent))
	

def read_acc():
	ax, ay, az, gx, gy, gz = lsm.get_readings()
	ax = sent * ax
	ay = sent * ay
	az = sent * az
	gx = raw_to_rads(gx)
	gy = raw_to_rads(gy)
	gz = raw_to_rads(gz)
	
	return ax, ay, az, gx, gy, gz

def moving_avg(data, window_size=3):
	kernel = np.ones(window_size) / window_size
	padded_data = np.convolve(data, kernel, mode='same')
	return padded_data
	
	
	
