#--------------------------------------------------------------------------------                                                            #
#                        QSAR Modeling                                          #
#    (Predicting the acitivity of test compounds using supervised learning)     #
#           Source of Dataset : AquaticTox  - fromm QSARdata package            #
#--------------------------------------------------------------------------------
setwd('F:/DATASCIENCE/DS-PROJECTS/13_Chem_Modeling/')
rm(list=ls())
#-----------------------------------------------------------------

library(QSARdata)
library(dplyr)
library(mlr)
library(corrplot)
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
#-----------------------------------------------------------
train <- sample_frac(AT_full_filter, 0.8, replace=TRUE)
rid <- as.numeric(rownames(train))
test <- AT_full_filter[-rid,]

train <- train[,-1]
test <- test[,-1]
summarizeColumns(train)

#---------------------------------------------------------------
trainTask <- makeRegrTask(data=train, target='Activity')
testTask <- makeRegrTask(data=test, target='Activity')
trainTask <- normalizeFeatures(trainTask, method='standardize')
testTask <- normalizeFeatures(testTask, method='standardize')

lrns <- list(makeLearner(id='lm',      'regr.lm', predict.type = 'response'),
             makeLearner(id='plsr',    'regr.plsr',predict.type = 'response'),
             makeLearner(id='glmnet',  'regr.glmnet',predict.type = 'response'),
             makeLearner(id='ksvm',    'regr.ksvm', predict.type = 'response'),
             makeLearner(id='nnet',    'regr.nnet', predict.type = 'response'),
             makeLearner(id='rf',      'regr.randomForest', predict.type = 'response'),
             makeLearner(id='gbm',     'regr.gbm', predict.type = 'response'),
             makeLearner(id='xgb',     'regr.xgboost', predict.type = 'response')
)

rdesc <- makeResampleDesc(method='CV', iter=10, stratify = FALSE)
bmr <- benchmark(lrns, trainTask, rdesc)
bmr
#-----------------------------------------------------------
#---------------------- Training & Predicting --------------
listMeasures('regr')
svm_lrn <- makeLearner(id='svm', 'regr.ksvm', predict.type='response')
rf_lrn <- makeLearner(id='svm', 'regr.randomForest', predict.type='response')

svm_model <- train(svm_lrn, trainTask)
rf_model <- train(rf_lrn, trainTask)

svm_pred <- predict(svm_model, testTask)
rf_pred <- predict(rf_model, testTask)

performance(svm_pred, measures=list(mse,rmse,rsq,arsq))
performance(rf_pred, measures=list(mse,rmse,rsq,arsq))
#-------------------------------------------------------------
#------------------------ Tuning SVM -------------------------
getParamSet('regr.ksvm')
?makeParamSet
ps.svm <- makeParamSet(
  makeDiscreteParam("C", values = seq(0,5,by=0.1)), #cost parameters
  makeDiscreteParam("sigma", values = seq(0,5,by=0.1)) #RBF Kernel Parameter
)

rancontrol <- makeTuneControlRandom(maxit = 100L)
tune_svm <- tuneParams(svm_lrn, trainTask, rdesc, par.set=ps.svm, control=rancontrol)
svm_tuned <- setHyperPars(svm_lrn,par.vals=tune_svm$x)

svm_tune_model <- train(svm_tuned, trainTask)
svm_tune_pred <- predict(svm_tune_model, testTask)
performance(svm_tune_pred, measures=list(mse,rmse,rsq,arsq))
#--------------------------------------------------------------
#------------------------ Tuning RF ---------------------------
getParamSet('regr.randomForest')
ps.rf <- makeParamSet(
  makeIntegerParam('ntree', lower=50, upper=500),
  makeIntegerParam('mtry', lower=3, upper=10)
)
tune_rf <- tuneParams(rf_lrn, trainTask, rdesc, par.set=ps.rf, control=rancontrol)
rf_tuned <- setHyperPars(rf_lrn, par.vals=tune_rf$x)

rf_tuned_model <- train(rf_tuned, trainTask)
rf_tuned_pred <- predict(rf_tuned_model, testTask)
performance(rf_tuned_pred, measures=list(mse,rmse,rsq,arsq))
#------------------------------------------------------------
