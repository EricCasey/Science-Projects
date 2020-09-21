# ------------------------------------------------------------------------------
# --- PACKAGES -----------------------------------------------------------------
# ------------------------------------------------------------------------------
#****datasets*****
library(mlbench) # data(Sonar)
library(C50) # data(churn)
library(MASS) # MIT Datasets?

# ===== DATA COLLECTION AND CLEANING ===========================================
require("httr")           # Makes RESTful Requests
require("jsonlite")       # Converts JSON to readable format (list)?

library(readxl)           # Reads .xlsx files
# data  <- read_excel("Desktop/MMA 860/ewr.xlsx", sheet = 1)

library(dplyr)            # Data Manipulation (tbl datatypes)
# glimpse(tbl)
# TIP: Tidy Data is Variables in Columns, Oberservations in Rows

# The Five Verbs: Select, Filter, Arrange, Mutate & Summarize

# select(df_or_tbl, Group, Sum)
# filter(df_or_tbl, logical test ...)
# arrange(df_or_tbl, df_column1, df_column2, desc)
# mutate(df_or_tbl, var = df_column1 + df_column2)
# summarize(df_or_tbl, sum = asdf, avg = sadf, var=asdf)

library(broom)
# bdims_tidy <- augment(lm_1)

# >>> Helper Functions >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# group_by(___)

# starts_with("X"): every name that starts with "X",
# ends_with("X"): every name that ends with "X",
# contains("X"): every name that contains "X",
# matches("X"): every name that matches "X", where "X" can be a regular expression,
# num_range("x", 1:5): the variables named x01, x02, x03, x04 and x05,
# one_of(x): every name that appears in x, which should be a character vector.

# min(x) - minimum value of vector x.
# max(x) - maximum value of vector x.
# mean(x) - mean value of vector x.
# median(x) - median value of vector x.
# quantile(x, p) - pth quantile of vector x.
# sd(x) - standard deviation of vector x.
# var(x) - variance of vector x.
# IQR(x) - Inter Quartile Range (IQR) of vector x.
# diff(range(x)) - total range of vector x.

# first(x) - The first element of vector x.
# last(x) - The last element of vector x.
# nth(x, n) - The nth element of vector x.
# n() - The number of rows in the data.frame or group of observations that summarise() describes.
# n_distinct(x) - The number of unique values in vector x.

# %>% - "Pipe", like "THEN", let's you carry initial data param input through to inner functions

library(sqldf)


# ===== DATA VISUALIZATION  ====================================================

library(ggplot2)            # Data Visualization
# SCATTERPLOT:
# ggplot(data = , aes(y = , x = ))+ geom_point()

library(lattice)             # Data In A Grid

library(curl)
# curl_echo("http://www.lagershed.com")

library(car)
# qqPlot()

library(rvest)
library(purrr)

library(corrplot)
# corrplot(cor(mtars), method = "ellipse")

library(rpart) # Decision tree models
# treeModel <- rpart(mpg ~ ., data = mtcars)
# plot(treeModel)
# text(treeModel, cex = 1.6, col = "red", xpd = TRUE)

library(aplpack)
# bagplot()



# >>>> BASE GRAPHICS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# > HIGH Level Functions
# plot(df or x, y, type = "o", pch = 1)

# sunflowerplot(x, y, main = "title here")
# boxplot(x, y, main = "title here")
# barplot(height, width,...)
# mosaicplot()
# hist()
# bagplot()
# abline(a = NULL, b = NULL)

# > LOW Level Functions
# points()
# lines()
# text()
# text()
# abline()

# par(mfrow = c(1, 2))  # for parameter. to set global vis variables



# ------------------------------------------------------------------------------
# --- BASIC R STUFF ------------------------------------------------------------
# ------------------------------------------------------------------------------

# paste(thing, thing, thing, sep = "-") result -> thing-thing-thing

# >>> Logical Operators
# x < y, TRUE if x is less than y
# x <= y, TRUE if x is less than or equal to y
# x == y, TRUE if x equals y
# x != y, TRUE if x does not equal y
# x >= y, TRUE if x is greater than or equal to y
# x > y, TRUE if x is greater than y
# x %in% c(a, b, c), TRUE if x is in the vector c(a, b, c)
# &, and
# |, or
# !, not
# !is.na(x)

# levels() #run this on categorical data to find out how many options there are
# table() # gives you a frequency table of two categorical variables
# prop.table(data, 1 or 2) # give you it proportionally
#

# # Shuffle row indices: rows
# rows <- sample(nrow(my_data))
# # Randomly order data: Sonar
# RandomRows <- my_data[rows]



# ------------------------------------------------------------------------------
# ----- ANALYSIS TOOLS ---------------------------------------------------------
# ------------------------------------------------------------------------------

# TESTING HETEROSKEDACISITY
# GRAPH IT, WHITE IT, BREUSCH IT
# plot(lm)
# white.test(x, lag = 1, qstar = 2, q = 10, range = 4,
#            type = c("Chisq","F"), scale = TRUE, …)
# bptest(Time_on_Foodie ~ N_Pictures + N_Children + Income + M_Status + Gender, data = foodie)


# lm(Y ~ x + x + x + ..., data = blah)
library(estimatr)
# lm_robust(Y ~ X1 + X2, file, se_type=”HC3”)

library(car)
#ncvTest(reg)

library(sensitivity)
#fast() # Fourier Amplitude Sensitivity Test

library(konfound)
# konfound(model_object (lm), tested_variable, alpha = 0.05, tails = 2,
#          to_return = "print/plot", test_all = FALSE)


# lhypo <- linearHypothesis(lm_7, c("Price = 0", "Ad_Budget = 0"))
# print("--- L Hypo ---")
# print(lhypo)

library(corrplot)
# corr <- cor(data)
# #corrplot.mixed(corr, lower = "square", upper = "circle", tl.col = "black", order = "hclust")
# corrplot(corr, lower = "square", upper = "circle", tl.col = "black", order = "AOE")

# ------------------------------------------------------------------------------
# ----- PREDICTIVE MODELING ----------------------------------------------------
# ------------------------------------------------------------------------------
#TIP: START WITH GLMNET, THEN RANDOM FOREST, THEN
library(caret)
# set.seed(42)

# GRID FOR GLMNET MODELS
# myGrid <- expand.grid(
#   alpha = 0:1,
#   lambda = seq(0.0001, 0.1, length = 10)
# )

# SPLITTING DATA INTO FOLDS
# myFolds <- createFolds(train#columns, k = 5)

# REUSABLE CONTROL OBJECT TEMPLATE
# myControl <- trainControl(
#   method = "cv",
#   number = 10,
#   summaryFunction = twoClassSummary,
#   classProbs = TRUE, # IMPORTANT!
#   verboseIter = TRUE,
#   index: myFolds
# )

# TEMPLATE FOR MODEL TRAINING WITH CARET
# model <- train(formula,
  # data = data,
  # method = "glmnet",
  # method = "ranger", # for Random Forest
  # metric = "ROC", #for glmnet
  # tuneGrid = myGrid,
  # preProcess = c("medianImpute", "center", "scale", "nnImpute", "pca",
  #                "spacialSign", "zv", "nzv"),
  # trControl = myControl
# )

# CONFUSION MATRIX
#   * this is the rate at which type I & type II errors are made.
# confusionMatrix(p_class, test[["class"]])



library(caTools)
# colAUC(p, test [["Class"]], plotROC = TRUE)

# RANDOM FORESTS (SOMETIMES MORE ACCURATE THAN GLMNET)
# Random forest have hyperparameters, Require "Tuning"
# "mtry" is the number of variables at each split
# Grid Search for selecting out of sample errors



# # Print maximum ROC statistic
# print(max(model[["results"]][["ROC"]]))


# plot(lm_2$finalModel)
# print(min(model$results$RMSE))


## CHOOSING A MODEL
# look for highest average AUC
# lowest standard deviation in AUC

# model_list <- list (
#   glmnet = model_glmnet,
#   rf = model_rf
# )
# resamps <- resamples(model_list)
# summmary(resamps)

library(caretEnsenble)
#bwplot(resamples, metric = "ROC")
#dotplot(resamples, metric = "ROC")
# densityplot(resamples, metric = "ROC")
# xyplot(resamples, metricc = "ROC")
#dotplot(lots_of_samples, metric = "ROC")


# Create ensemble model: stack
stack <- caretStack(model_list, method = "glm")

# Look at summary
summary(stack)

# ------------------------------------------------------------------------------
# ----- ANALYSIS NOTES ---------------------------------------------------------
# ------------------------------------------------------------------------------

# Characterizing bivariate relationships
# > Form (linear, quadratic (shaped like a U), non-linear)
# > Direction (positive / negative)
# > Strength (scatter / noise)
# > Outliers

# Correlations (Only two variables, not multiple regression)
# > Corellation coefficient between -1 and 1
# > 0 is weak
# > Sign -> direction
# > Magnitude > strength

# Spurious Correlation (ridiculous relationships)
# >

# # Compute errors: error
# error <- p - test[["price"]]

# # Calculate RMSE (closer to 1 the better?)
# sqrt(mean(error^2))

# Helpful to remove low variance metrics

# ------------------------------------------------------------------------------
# ----- SIX SIGMA NOTES ---------------------------------------------------------
# ------------------------------------------------------------------------------

# DMAIC - Define, Measure, Analyze, Improve, Control (DMAIC Approach)

