
################################################################################
################################################################################
# KERAS for REGRESSION #########################################################
################################################################################


# libraries
import numpy as np
import pandas as pd
import keras
from keras.layers import Dense
from keras.models import Sequential


# 0. DATA ###############################

# Reading data
data = pd.read_csv("", delimiter=",")
target = data.y
predictors = data.drop(["y"], axis=1).as_matrix()
n_cols = predictors.shape[1]  # input layer equal number of features (=n_cols)



# 1. ARCHITECTURE ###############################
# -> Choose Number of Nodes, Connections and Layers

# Set up the model: model
model = Sequential()
# First hidden layer plus input layer
model.add(Dense(50, activation="relu", input_shape=(n_cols, )))
# Second hidden layer
model.add(Dense(32, activation="relu"))
# Output layer
model.add(Dense(1))



# 2. COMPILATION ###############################
# -> Choose Optimizer and Loss Fun 
model.compile(
    optimizer="adam",           # optimizer selection
    loss="mean_squared_error")  # loss fun selection



# 3. FITTING ###############################
# -> Fit model on training data
model.fit(
    predictors,     # predictors always first,
    target)         # targets second.
# output shows "log of progress on training data as keras updates weights"



# 4. PREDICT ###############################
# -> Use fitted model on test data
predictions = model.predict(predictors_test)







################################################################################
################################################################################
# KERAS for CLASSIFICATON ######################################################
################################################################################

# Libraries
import pandas as pd
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical


# 0. DATA ###############################
# Read data
data = pd.read_csv("", delimiter=",")
# pd has great data inspection possibilities, np less so
target = to_categorical(data.shot_result)  # target transformation!!!
predictors = data.drop(["shot_result"], axis=1).as_matrix()



# 1. ARCHITECTURE ###############################
# Set up the model
model = Sequential()
# Add the first layer
model.add(Dense(32, activation="relu", input_shape=(n_cols,)))
# Output layer
model.add(Dense(2, activation="softmax"))  # softmax activation!!!



# 2. COMPILATION ###############################
# Compile the model
model.compile(
    optimizer="sgd",
    loss="categorical_crossentropy",  # categorical_crossentropy loss fun!!!
    metrics=["accuracy"])  # fraction of correct predictions!!!


# 3. FITTING ###############################
# Fit the model
model.fit(predictors, target)



# 4. PREDICT ###############################
predictions = model.predict(predictors_test)
# Calculate predicted probability of scoring: predicted_prob_true
predicted_prob_true = predictions[:, 1]
# (Note: first columns contains probs of MISSING, second column contains probs of SCORING)









################################################################################
################################################################################
# MODEL OPTIMIZATION CHALLENGES ################################################
################################################################################

# LEARNING RATE
# ACTIVATION FUN




# Testing appropriate Learning Rate with SGD 

# set up new models fun:
def get_new_model(input_shape=input_shape):
    model=Sequential()
    model.add(Dense(100, activation="relu", input_shape=input_shape))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(2, activation="softmax"))
    return(model)


# Import the SGD optimizer
from keras.optimizers import SGD

# Create list of learning rates: lr_to_test
lr_to_test = [0.000001, 0.01, 1]

# Loop over learning rates
for lr in lr_to_test:
    print('\n\nTesting model with learning rate: %f\n'%lr )
    
    # Build new model to test, unaffected by previous models
    model = get_new_model()
    
    # Create SGD optimizer with specified learning rate: my_optimizer
    my_optimizer = SGD(lr=lr)
    
    # Compile the model
    model.compile(
        optimizer=my_optimizer,
        loss="categorical_crossentropy")
    
    # Fit the model
    model.fit(
        predictors,
        target)





################################################################################
################################################################################
# MODEL VALIDATION #############################################################
################################################################################

# Model performance to be assessed on test data, not training data (-> hold-out principle). DL problems usually work with large datasets. This is why single validation splits are often satisfying (sample size) and cross val would be computationally too expensive.

model.fit(
    predictors, target,
    validation_split=0.3)


# Objective: Minimization of val_loss function (test data loss fun). EarlyStopping halts optimization when val_loss fun stops improving!

from keras.callbacks import EarlyStopping

# Save the number of columns in predictors: n_cols
n_cols = predictors.shape[1]
input_shape = (n_cols,)

# Specify the model
model = Sequential()
model.add(Dense(100, activation='relu', input_shape = input_shape))
model.add(Dense(100, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"])

# Define early_stopping_monitor
early_stopping_monitor = EarlyStopping(patience=2)  # patience: number of runs w/o improvement allowed. 2 to 3 is standard

# Fit the model
model.fit(
    predictors, target,
    validation_split=0.3,
    epochs=30,
    callbacks=[early_stopping_monitor])



################################################################################
################################################################################
# ARCHITECTURE EXPERIMENTATION #################################################
################################################################################

# -> Now that we can measure model performance, let's start to experiment with different architectures!!!


# A: Adding more nodes: ######################################

# Define early_stopping_monitor
early_stopping_monitor = EarlyStopping(patience=2)

# Create the new model: model_2
model_2 = Sequential()

# Add the first and second layers
model_2.add(Dense(100, activation="relu", input_shape=input_shape))
model_2.add(Dense(100, activation="relu"))


# Add the output layer
model_2.add(Dense(2, activation="softmax"))


# Compile model_2
model_2.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"])
    
# Fit model_1
model_1_training = model_1.fit(
    predictors, target, 
    epochs=15, validation_split=0.2, 
    callbacks=[early_stopping_monitor], verbose=False)

# Fit model_2
model_2_training = model_2.fit(
    predictors, target, 
    epochs=15, validation_split=0.2, 
    callbacks=[early_stopping_monitor], verbose=False)

# Plotting!!!
plt.plot(
    model_1_training.history['val_loss'], 'r', 
    model_2_training.history['val_loss'], 'b')
plt.xlabel('Epochs')
plt.ylabel('Validation score')
plt.show()



# B: Adding more layers: ##################################

# The input shape to use in the first hidden layer
input_shape = (n_cols,)

# Create the new model: model_2
model_2 = Sequential()

# Add the first, second, and third hidden layers
model_2.add(Dense(50, activation="relu", input_shape=input_shape))
model_2.add(Dense(50, activation="relu"))
model_2.add(Dense(50, activation="relu"))

# Add the output layer
model_2.add(Dense(2, activation="softmax"))

# Compile model_2
model_2.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"])

# Fit model 1
model_1_training = model_1.fit(
    predictors, target, 
    epochs=20, validation_split=0.4, 
    callbacks=[early_stopping_monitor], verbose=False)

# Fit model 2
model_2_training = model_2.fit(
    predictors, target, 
    epochs=20, validation_split=0.4, 
    callbacks=[early_stopping_monitor], verbose=False)

# Create the plot!!!
plt.plot(
    model_1_training.history['val_loss'], 'r', 
    model_2_training.history['val_loss'], 'b')
plt.xlabel('Epochs')
plt.ylabel('Validation score')
plt.show()





