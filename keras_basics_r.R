pacman::p_load("magrittr")
pacman::p_load("tidyverse")
pacman::p_load("keras")
stopifnot( pacman::p_exists("caret") )

# download data
data = dataset_mnist()

#separating train and test file
train_x = data$train$x
train_y = data$train$y
test_x  = data$test$x
test_y  = data$test$y

rm(data)

# converting a 2D array into a 1D array for feeding into the MLP and normalising the matrix
train_x = array(train_x, dim = c(dim(train_x)[1], prod(dim(train_x)[-1]))) / 255
test_x  = array(test_x, dim = c(dim(test_x)[1], prod(dim(test_x)[-1]))) / 255

# converting the target variable to once hot encoded vectors using keras inbuilt function
train_y = to_categorical(train_y, 10)
test_y  = to_categorical(test_y, 10)

#defining a keras sequential model
model = keras_model_sequential()

# defining the model with 1 input layer[784 neurons]
# , 1 hidden layer[784 neurons] with dropout rate 0.4 and 1 output layer[10 neurons]
# i.e number of digits from 0 to 9

model %>%
  layer_dense(units = 784, input_shape = 784) %>%
  layer_dropout(rate = 0.4) %>%
  layer_activation(activation = 'relu') %>%
  layer_dense(units = 10) %>%
  layer_activation(activation = 'softmax')

#compiling the defined model with metric = accuracy and optimiser as adam.
model %>% compile(loss        = 'categorical_crossentropy'
                  , optimizer = 'adam'
                  , metrics   = c('accuracy')
                  )

# fitting the model on the training dataset
history = model %>%
  fit(train_x
      , train_y
      , epochs           = 10
      , batch_size       = 128
      , validation_split = 0.2
      )

#Evaluating model on the cross validation dataset
loss_and_metrics = model %>% evaluate(test_x, test_y, batch_size = 128)
loss_and_metrics

# plot training and validation errors
plot(history)

# generate predictions on newdata
classes <- model %>%
  predict_classes(test_x, batch_size = 128) %>%
  add(1) %>%
  as.integer()

# confusion matrix
caret::confusionMatrix(classes
                       , apply(test_y, 1, function(x) which(x == 1))
                       )
