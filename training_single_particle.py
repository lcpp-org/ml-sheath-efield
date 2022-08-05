import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import Dense
import matplotlib.pyplot as plt
import pdb
import time

def get_model(n_inputs, n_outputs):
	model = Sequential()
	model.add(Dense(1024, input_dim=n_inputs, kernel_initializer='random_normal', activation='relu'))
	model.add(Dense(512))
	model.add(Dense(256))
	model.add(Dense(128))
	model.add(Dense(n_outputs))
	model.compile(loss='mse', optimizer='adam',metrics=['mse'])
	model.summary()

	return model


model_name = 'testing_with_one_input'
start = time.time()

# Read and remove rows containing NaN
# col 0: v parallel
# col 1: v perpendicular
# col 2: Electric field
# col 3: Electric field gradient
# col 4: 2nd derivative of electric field
# col 5: Bx
# col 6: By
# col 7: major radius of ellipse fitted
# col 8: minor radius of ellipse fitted
training_set = np.load('training_particle.npy')
training_set = training_set[~np.any(np.isnan(training_set), axis=1)]

# Extract v_parallel and major/minor radius from the training particle
X = training_set[:,0]
y = training_set[:,7:]

# Normalized each feature by minmax normalization ( [x-min_x] / [max_x - min_x] )
max_in = np.max(X,axis=0)
min_in = np.min(X,axis = 0)
max_out = np.max(y,axis=0)
min_out = np.min(y,axis=0)

X = (X - min_in) / (max_in - min_in)
y = (y - min_out) / (max_out - min_out)

# Training
model = get_model(1,2)
history = model.fit(X,y, verbose = 2 ,batch_size = 128, epochs = 3)
score = model.evaluate(X, y)
print("Testing on training set, %s: %.2f" % (model.metrics_names[1], score[1]))

# Serialize model to JSON
model_json = model.to_json()
with open(model_name + ".json", "w") as json_file:
    json_file.write(model_json)
# Serialize weights to HDF5
model.save_weights(model_name + ".h5")
print("Saved model to disk")


end = time.time()
print("Ellapsed time(s): ",end - start)

# Plot history of loss function
plt.semilogy(history.history["loss"])
plt.xlabel("Epoch",fontsize = 18)
plt.title("History of loss function",fontsize=18)
plt.tick_params(axis = 'both', which = 'major', labelsize = 16)
plt.show()
