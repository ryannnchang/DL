#Importing files
from model import model_v2
from predict import predict, predict2
from sensor_preprocessing import read_acc
from screen import off, display

#importing libraries
import time
import RPi.GPIO as GPIO
import numpy as np


#Define GPIO button
button = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(button, GPIO.IN)

#Prediction Labels
data_labels = {0.0:"Walking", 1.0: "WalkUp", 2.0: "WalkDw", 3.0: "Sitting", 4.0: "Standing", 5.0: "Laying"}

#Collecting (ax,ay, az)
while GPIO.input(button) != False:
	window = [[],[],[]]
	window2 = [[],[],[],[],[],[]]
	
	while len(window[0]) < 128:
		ax, ay, az, gx, gy, gz = read_acc()
		window[0].append(ax)
		window[1].append(ay)
		window[2].append(az)
		
		window2[0].append(ax)
		window2[1].append(ay)
		window2[2].append(az)
		window2[3].append(gx)
		window2[4].append(gy)
		window2[5].append(gz)
		
		time.sleep(0.02)
	
	window = np.array(window)
	guess_num = predict(window)
	guess = data_labels[guess_num]
	
	window2 = np.array(window2)
	guess_num2 = predict2(window2)
	guess2 = data_labels[guess_num2]
	
	print(f"Prediction: {guess} \t PredictionGyro: {guess2}")
	display(guess, guess2)
	
off()

