#
# ****** Recurrent Neural Network ******

# read in the packages we'll use
# install.packages("keras")
library(keras) # for deep learning
# install.packages("tidyverse")
library(tidyverse) # general utility functions
library(caret) # machine learning utility functions

# read in our data
weather_data <- read_csv("~/Desktop/TA/seattleWeather.csv")

# check out the first few rows
head(weather_data)

# set some parameters for our model
max_len <- 6 # the number of previous examples we'll look at
batch_size <- 32 # number of sequences to look at at one time during training
total_epochs <- 15 # number of times we re-use the whole dataset to train model

# set a random seed for reproducibility
set.seed(123)

# select out the colum with info on how often it rained
rain <- weather_data$RAIN

# summerize this
table(rain)

# Cut the text in overlapping sample sequences of max_len characters

# get a list of start indexes for our (overlapping) chunks
start_indexes <- seq(1, length(rain) - (max_len + 1), by = 3)

# create an empty matrix to store our data in
weather_matrix <- matrix(nrow = length(start_indexes), ncol = max_len + 1)

# fill out matrix with the overlapping slices of our dataset
for (i in 1:length(start_indexes)){
  weather_matrix[i,] <- rain[start_indexes[i]:(start_indexes[i] + max_len)]
}

# make sure it's numeric
weather_matrix <- weather_matrix * 1

# remove NA's if any
if(anyNA(weather_matrix)){
  weather_matrix <- na.omit(weather_matrix)
}

# split our data into the day we're predict (y), and the 
# sequence of days leading up to it (X)
X <- weather_matrix[,-ncol(weather_matrix)]
y <- weather_matrix[,ncol(weather_matrix)]

# create an index to split our data into testing & training sets
training_index <- createDataPartition(y, p = .9, 
                                      list = FALSE, 
                                      times = 1)

# training data
X_train <- array(X[training_index,], dim = c(length(training_index), max_len, 1))
y_train <- y[training_index]

# testing data
X_test <- array(X[-training_index,], dim = c(length(y) - length(training_index), max_len, 1))
y_test <- y[-training_index]

# initialize our model
#A Sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor.
model <- keras_model_sequential()

# dimensions of our input data
dim(X_train)

# our input layer
model %>%
  layer_dense(input_shape = dim(X_train)[2:3], units = max_len)

model %>% 
  layer_simple_rnn(units = 6)

model %>%
  layer_dense(units = 1, activation = 'sigmoid') # output, The main reason to use sigmoid function is because it exists from 0 to 1. Generally used for models to predict probability as an output.
#look at our model architecture
summary(model)

model %>% compile(loss = 'binary_crossentropy', 
                  optimizer = 'RMSprop', 
                  metrics = c('accuracy'))

#Train our model
# Actually train our model! !!! This step will take a while !!!
trained_model <- model %>% fit(
  x = X_train, # sequence we're using for prediction 
  y = y_train, # sequence we're predicting
  batch_size = batch_size, # how many samples to pass to our model at a time
  epochs = total_epochs, # how many times we'll look @ the whole dataset
  validation_split = 0.1) # how much data to hold out for testing as we go along
#Evaluate our model
# how well did our trained model do?
trained_model


# plot how our model performance changed during training 
plot(trained_model)

# Predict the classes for the test data
classes <- model %>% predict_classes(X_test, batch_size = batch_size)

# Confusion matrix
table(y_test, classes)

model %>% evaluate(X_test, y_test, batch_size = batch_size)

# baseline: just guess the weather will be the same as yesterday
day_before <- X_test[,max_len - 1,1]

# Confusion matrix
table(y_test, day_before)

# accuracy
sum(day_before == classes)/length(classes)
