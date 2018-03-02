#Main File to import dataset, train, test, validate and plot learning curves

#Install pre-requisite packages if not present
install.packages("readr")
install.packages("ggplot2")

#Read csv data
library(readr)
dfOwls <- readr::read_csv("owls15.csv")

#Randomly shuffle the dataset rows (repeatedly shuffled for 5 times)
rows_count <- nrow(dfOwls)
for(k in 1:5){
  dfOwls<-dfOwls[sample(rows_count),]
}

source("Perceptron.r")
source("Evaluation_Cross_Validation.r")
source("Evaluation_Validation.r")
source("Evaluation_Curves.r")


#Hold out 1/3 rd validation dataset
validation_instances <- sample(nrow(dfOwls)/3)
dfOwls_validation<-dfOwls[validation_instances,] #1/3 rd validation set
dfOwls_train <- dfOwls[-validation_instances,] #2/3 rd training set

#Build Perceptron Model
p_model <- Perceptron(0.01)

#Set number of epochs (iterations)
num_of_epochs <- 100 #Ideally, run with 1000 number of epochs but 1000 takes considerable amount (>10 min) to train

#plot Learning Curve - Accuracy vs Training Sample size
plot_learning_curve(p_model, dfOwls_train, dfOwls_validation, number_of_iterations = num_of_epochs)

#plot Learning Curve - Accuracy vs Number of Epochs (Iterations)
plot_learning_curve_epochs(p_model, dfOwls_train, dfOwls_validation)

#plot Learning Curve - Accuracy vs Learning Rate values
plot_learning_curve_learning_Rates(dfOwls_train, dfOwls_validation, num_of_epochs = num_of_epochs)

#Train - Test - Cross Validate accross 10 folds
Cross_Validate(p_model, dfOwls_train, num_of_iterations = num_of_epochs, num_of_folds = 10)
#Cross_Validate(ml_model, dataset, num_of_iterations, num_of_folds)

#Validate results with held out validation dataset

Validate(p_model, dfOwls_train, dfOwls_validation, number_of_iterations = 10)


