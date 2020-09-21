library(readr)
library(readxl)

library(ggplot2)
library(car)
library(sensitivity)
library(konfound)
library(lmtest)
library(tidyverse)
library(lubridate)
library(dplyr)
library(estimatr)
library(corrplot)
library(caret)
library(glmnet)
library(earth)
library(e1071)
library(ranger)
library(spikeslab)
library(plyr)
library(penalized)
library(mgcv)
library(beepr)
library(spls)
#library(gbm)
library(party)
library(mboost)
# library(import)
# library(RSNNS)

library(caretEnsemble)

set.seed(42)


options(na.action = "na.fail")
options(stringsAsFactors = FALSE)
#sink(stdout(), type = "message")
setwd("~/code/R/DATA_excel/kaggle_houseData/")

#rm(list = ls(all = TRUE))

require(caret)

# train_clean <- read_excel("~/code/R/DATA_excel/kaggle_houseData/train.xlsx", sheet = 3)
# test_clean <- read_excel("~/code/R/DATA_excel/kaggle_houseData/train.xlsx", sheet = 4)

train_clean <- read_excel("~/code/R/DATA_excel/kaggle_houseData/train.xlsx", sheet = 5)
test_clean <- read_excel("~/code/R/DATA_excel/kaggle_houseData/train.xlsx", sheet = 6)

# data <- as.data.frame(train_clean)
# test <- as.data.frame(test_clean)
data <- train_clean
test <- test_clean

labelName <- 'SalePrice'
predictors <- names(data)[names(data) != labelName]



#print(names(data)
#print(names(data)[nearZeroVar(data)])

#data$MSSubClass <- as.factor(data$MSSubClass)
#
str(data)
str(test)

#corr <- cor(data)
# print(corr)
#corrplot.mixed(corr, lower = "square", upper = "circle", tl.col = "black", order = "hclust")
#corrplot(corr, order = "hclust")

#print(head(data))

PP <- c('pca', 'nzv', 'center', 'scale')
#PP <- c('pca', 'nzv', 'center', 'scale')

# myGrid <- expand.grid(
#   alpha = 0:1,
#   lambda = seq(0.0001, 0.1, length = 10)
# )

myControl <- trainControl(method = "cv", # cross validation of models (cv, boot)
                          number = 10,
                          summaryFunction = defaultSummary,
                         # summaryFunction = twoClassSummary,
                          #index = createMultiFolds(data, k=75, times=1),
                          classProbs = TRUE, # IMPORTANT!
                          savePredictions = 'final',
                          verboseIter = FALSE,
                          allowParallel = TRUE
                          )

# Modeling

# Straigt up base LM
m_1 <- lm(SalePrice ~ ., data)
print(summary(m_1)) # best R2 is 0.8383

#plot(m_1)

# m_2 <- train(SalePrice ~ .,
#                data = data,
#                method = "lm",
#                #tuneGrid = myGrid,
#                preProcess = c("nzv", "center", "scale"),
#                #preProcess = "pca",
#                trControl = myControl
# )

 #print(m_2)

#
# m_3 <- caret::train(SalePrice ~ .,
#   data = data,
#   method = "glmnet",
#   #preProcess = c("medianImpute", "center", "scale"),
#   preProcess = "pca",
#   #preProcess = PP,
#   trControl = myControl
# )

#print(m_3)


#prediction <- predict(m_3, test)

# m_3 <- train(SalePrice ~ .,
#              data = data,
#              method = "glmnet",
#              preProcess = "pca",
#              #preProcess = PP,
#              trControl = myControl
# ) # LOG RMSE: 0.17196   best R2 is 0.85



#______________________________________-
## SSECTION 2

#Make a list of all the model -->  model3, model4,
print("caretList - model list")
model_list_1 <- caretEnsemble::caretList(SalePrice ~ .,
                                       data = data,
                                       trControl = myControl,
                                       preProcess = PP,
                                       continue_on_fail = FALSE,
                                       trace = FALSE,
                                       methodList = c(
                                         #"gbm",
                                                      #"knn",
                                                      #"earth",
                                            #"gbm",
                                                      #"lm",
                                                      "bagEarth",
                                                      #"gcvEarth"
                                                      "xgbLinear",
                                                      #"svmRadial",
                                                      "glmnet"
                                                      #"parRF",
                                                      #"ranger",
                                                      #"spikeslab",
                                                      #"penalized",
                                                      #"rlm",
                                                     # "ppr"
                                        )
                                       )




# Pass model_list to resamples(): resamples
print("resamples")
resamples <- resamples(model_list_1)
print(summary(resamples))


# 3
# RMSE      Rsquared   MAE
# 31695.98  0.8450359  19317.17


print("caretStack - Making A Model Porch Crawl")
stack <- caretEnsemble::caretStack(model_list_1, method = "glm")

print(stack)
#
#
preds <- predict(stack, test)
print(head(preds))
write.csv(preds, file = "submission.csv")


beep(sound = 8, expr = NULL)

## Working Best (10 repeats at the top)
# RMSE      Rsquared  MAE
# 29645.15  0.860521  17255.42


## BEST ALL TIME
# LOG RMSE: 0.14414
# Best R2 is 0.85
# 1 - 2305 / 4220

