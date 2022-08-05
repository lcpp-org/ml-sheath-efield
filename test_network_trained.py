import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import Dense
import pdb
import time
import matplotlib.pyplot as plt

# load json and create model
model_name = 'testing_with_one_input'
json_file = open(model_name+'.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(model_name+".h5")
print("Loaded model from disk")

# Load training data and remove NaN
training_set = np.load('training_particle.npy')
training_set = training_set[~np.any(np.isnan(training_set), axis=1)]


# Use the model trained on the training data for visual verification
major = np.zeros(len(training_set))
minor = np.zeros(len(training_set))

# Inverse minmax normalization factor
max1  =  np.max(training_set[:,-2])
min1 = np.min(training_set[:,-2])
max2  =  np.max(training_set[:,-1])
min2 = np.min(training_set[:,-1])

for i in range(len(training_set)):
    prediction = loaded_model.predict(np.asarray([training_set[i,0]]))
    major[i] = prediction[0][0] * (max1-min1) + min1
    minor[i] = prediction[0][1] * (max2-min2) + min2



plt.figure()
plt.plot(major,label='Major radius')
plt.plot(minor, label = 'Minor radius')
plt.ylabel('r (m)', fontsize = 18)
plt.xlabel('i-th time step', fontsize = 18)
plt.tick_params(axis = 'both', which = 'major', labelsize = 16)
plt.legend()
plt.show()
