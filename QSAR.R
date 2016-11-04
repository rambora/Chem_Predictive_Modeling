#--------------------------------------------------------------------------------                                                            #
#                        QSAR Modeling                                          #
#    (Predicting the acitivity of test compounds using supervised learning)     #
#           Source of Dataset : AquaticTox  - fromm QSARdata package            #
#--------------------------------------------------------------------------------
setwd('F:/DATASCIENCE/DS-PROJECTS/13_Chem_Modeling/')
rm(list=ls())
#-----------------------------------------------------------------

library(QSARdata)
#library(rcdk)
library(dplyr)
library(ggplot2)
library(gridExtra)
library(mlr)
library(corrplot)
library(caret)
library(pls)
library(MASS)
library(glmnet)
# ------------- Data Loading, Processing and Task Preparation ----------------
data(AquaticTox)

AT_out <- AquaticTox_Outcome
AT_QP <- AquaticTox_QuickProp
head(AT_out)
head(AT_QP)
dim(AT_out)
dim(AT_QP)
colnames(AT_QP)
names(AT_QP) <- gsub('QikProp', 'QP', names(AT_QP))
AT_full <- cbind(AT_QP, AT_out$Activity)
head(AT_full)
dim(AT_full)
colnames(AT_full)[51] <- 'Activity'
#---------------------------------------------------------------
#     processing data sets with large number of features
colSums(is.na(AT_full))
which(complete.cases(AT_full)==FALSE)
AT_full[166,]
AT_full[190,]
AT_full[243,]
# Safely we can omit these three molecules as none of the properties of these comps are reported
AT_full <- na.omit(AT_full)                  # 1. removing the missing values
AT_full <- removeConstantFeatures(AT_full)   # 2. remove features with constant variance
# identifying the numeric/integer/factor classes
sapply(AT_full, class)
which(!sapply(AT_full, is.numeric))
names(AT_full)[!sapply(AT_full,is.numeric)]
# except 'Molecule', all others are numeric/integer
# nearZeroVariance is not much useful here as all the features are having close to zero variance
#near_zero <- nearZeroVar(AT_full, freqCut = 95/5, names=TRUE, uniqueCut=10, saveMetrics=TRUE) # 3. removes the nearzero variance columns
#AT_full <- AT_full[,-which(near_zero$zeroVar)]
dim(AT_full)

AT_full_cor <- cor(AT_full[, -c(1,51)])
corrplot(AT_full_cor)
cor_cols <- findCorrelation(AT_full_cor, cutoff=0.7) # 3. remove highly correlated features
AT_full_filter <- AT_full[,-cor_cols]
# This significantly reduced highly correlated features
dim(AT_full_filter)
head(AT_full_filter)
#-----------------------------------------------------------------
summary(AT_full_filter)          
summarizeColumns(AT_full_filter)
#pairs(AT_full_filter[,c(2:10,24)])

# looks like cols-8,9,10,11,12,20,26,27- requires special attention
par(mfrow=c(3,3))
for (i in (2:10)){
  boxplot(AT_full_filter[,i], main=names(AT_full_filter)[i])
}
for (i in (11:19)){
  boxplot(AT_full_filter[,i], main=names(AT_full_filter)[i])
}
for (i in (20:32)){
  boxplot(AT_full_filter[,i], main=names(AT_full_filter)[i])
}
# many outliers and in most cases, the data is skewed

# This suggests 'normalization and BoxCox transformations' are required
#-----------------------------------------------------------
# carrying out the modeling with 'caret' package as 'mlr' doesnot
# have 'Lasso and Ridge regression' implemented
#-----------------------------------------------------------
train <- sample_frac(AT_full_filter, 0.8, replace=TRUE)
rid <- as.numeric(rownames(train))
test <- AT_full_filter[-rid,]

train <- train[,-1]
test <- test[,-1]
summarizeColumns(train)

#---------------------------------------------------------------
trainControl <- trainControl(method='CV', number=10)

# Benchmarks
set.seed(3)
fit.lm <- train(Activity~., data=train, method = 'lm', metric='RMSE', 
                preProc= c('center','scale','BoxCox'), trControl=trainControl)

set.seed(3)
fit.pls <- train(Activity~., data=train, method = 'pls', metric='RMSE', 
                preProc= c('center','scale','BoxCox'), trControl=trainControl)

# set.seed(3)
# fit.glmnet <- train(Activity~., data=train, method = 'glmnet', metric='RMSE', 
#                 preProc= c('center','scale','BoxCox'), trControl=trainControl)

set.seed(3)
fit.svm <- train(Activity~., data=train, method = 'svmRadial', metric='RMSE', 
                preProc= c('center','scale','BoxCox'), trControl=trainControl)

set.seed(3)
grid1 <- expand.grid(.cp=c(0,0.05,0.1))
fit.cart <- train(Activity~., data=train, method = 'rpart', metric='RMSE', tuneGrid=grid1, 
                preProc= c('center','scale','BoxCox'), trControl=trainControl)

set.seed(3)
mtry <- sqrt(ncol(train))
grid2 <- expand.grid(.mtry=mtry)
fit.rf <- train(Activity~., data=train, method = 'rf', metric='RMSE', tuneGrid=grid2, 
                  preProc= c('center','scale','BoxCox'), trControl=trainControl)
set.seed(3)
fit.knn <- train(Activity~., data=train, method = 'knn', metric='RMSE', 
                preProc= c('center','scale','BoxCox'), trControl=trainControl)

# Compare algorithms
trans_results <- resamples(list(LM=fit.lm, PLS=fit.pls, SVM=fit.svm, DT=fit.cart,
                               RF= fit.rf, KNN=fit.knn))
summary(trans_results)
dotplot(trans_results)

# library(glmnet)
# fit_1.glm <- glmnet(as.matrix(train[,-23]), as.vector(train[,23]), family='gaussian')
# plot(fit_1.glm)
# fit.glm <- cv.glmnet(as.matrix(train[,-23]), as.vector(train[,23]), nfolds=10, alpha=0.7, family='gaussian')
# summary(fit.glm)
# plot(fit.glm)
# fit.glm$lambda.min
# fit.glm$lambda.1se
# coef(fit.glm, s=fit.glm$lambda.min)
# par(mfrow=c(3,2))
# plot(fit.glm)
#------------------------------------------------------------
# Tuning Random Forest
rf_trainControl <- trainControl(method='repeatedcv', number=10, repeats=3, search='grid')
set.seed(3)
mtry <- sqrt(ncol(train))
grid2 <- expand.grid(.mtry=mtry)
fit.rf <- train(Activity~., data=train, method = 'rf', metric='RMSE', tuneGrid=grid2, 
                preProc= c('center','scale','BoxCox'), trControl=rf_trainControl)
print(fit.rf)

# Prediction

rf_predict <- predict(fit.rf, test) 
summary(rf_predict)
mse <- mean((test$Activity - rf_predict)^2)
rmse <- sqrt(mse)
#------------------------------------------------------------
