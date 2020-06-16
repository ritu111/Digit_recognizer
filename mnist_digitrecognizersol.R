######Digit Recogniser################
# 1. Business Understanding
# 2. Data Understanding
# 3. Data Preparation
# 4. Model Building 
#  4.1 Linear kernel
#  4.2 RBF Kernel
# 5 Hyperparameter tuning and cross validation

### 1. Business Understanding ###

#The goal is to develop a model that can correctly identify the digit (between 0-9)
#written in an image based on the pixel values as features

### 2. Data Understanding ###

# Number of Instances in train: 60000
# Number of Instances in test : 10000
# Number of Attributes        : 785

### 3. Data Preparation ###
# Installing and importing libraries
install.packages("kernlab")
install.packages("readr")
install.packages("caret")
install.packages("doParallel")
library(kernlab)
library(readr)
library(caret)
library(doParallel)

# Loading data 
Digitdata <- read.csv("mnist_train.csv",header=F,stringsAsFactors = FALSE)
test<-read.csv("mnist_test.csv",header = F,stringsAsFactors = FALSE)
View(Digitdata)
dim(Digitdata)
head(Digitdata)
str(Digitdata)
summary(Digitdata)

# Checking if data has NA values
which(is.na(Digitdata))
sapply(Digitdata, function(x) sum(is.na(x)))

# Coverting Digit column(V1) to factor 
Digitdata$V1<-factor(Digitdata$V1)
test$V1     <-factor(test$V1)

# Checking for columns with 0 values
#there are 67 columns with enitire 0 values, but we don't have to delete them as it have importance
columns<-Digitdata[,colSums(Digitdata != 0,na.rm = T)==0]
columns

set.seed(100)
train.indices = sample(1:nrow(Digitdata), 0.15*nrow(Digitdata)) #Taking 15% of data for train
train<-Digitdata[train.indices,]

test.indices = sample(1:nrow(test), 0.15*nrow(test))            #Taking 15% of data for test
test1<-test[test.indices,]

### 4. MODEL BUILDING ###
# Using linear SVM
model_linear<-ksvm(V1~.,data=train,scale=FALSE,kernel="vanilladot")
model_linear
eval_linear<-predict(model_linear,test1)
confusionMatrix(eval_linear,test1$V1)
#cost C = 1, Number of Support Vectors : 2545 


# Using Kernel RBF
model_kernel<-ksvm(V1~.,data=train,scale=FALSE,kernel="rbfdot")
model_kernel 
eval_kernel<-predict(model_kernel,test1)
confusionMatrix(eval_kernel,test1$V1)
## cost C = 1,Gaussian Radial Basis kernel function sigma = 1.63483744485576e-07, Number of Support Vectors : 3541


### 5. Cross Validation using 5 folds ###
#We will use 3 values of sigma i.e.(1.6348e-07 ,2.6348e-07 ,3.6348e-07g) because we got default sigma as 1.63483744485576e-07 from RBF
trainControl<- trainControl(method="cv", number=5)
metric<-"Accuracy"
set.seed(100)
grid<- expand.grid(.sigma=c(1.6348e-07 ,2.6348e-07 ,3.6348e-07 ), .C=c(1,5,10) )
cl<- makePSOCKcluster(4)
registerDoParallel(cl)
fit.svm<- train(V1~., data=train, method="svmRadial", metric=metric,allowParallel=T ,
                 tuneGrid=grid, trControl=trainControl)
stopCluster(cl)
plot(fit.svm)
print(fit.svm)
## The final values used for the model were sigma = 3.6348e-07 and C = 5 with accuracy of 96.87%

## Final model- using sigma = 3.6348e-07 and C = 5 creating final model
final_model_kernel<-ksvm(V1~.,data=train,scale=FALSE,kernel="rbfdot",C=5,sigma=3.6348e-07)
final_model_kernel
final_eval_kernel<-predict(final_model_kernel,test1)
confusionMatrix(final_eval_kernel,test1$V1)  # Overall accuracy: 0.966
