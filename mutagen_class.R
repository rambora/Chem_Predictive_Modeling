#---------------------------------------------------------------------------                                                            #
#          Mutagen vs Nonmutagen classification                            #
# (predicting the mutagenicity of a given smile (correspond to a compound) #
#   Source of Dataset : https://dsdht.wikispaces.com/                      #
#---------------------------------------------------------------------------
setwd('F:/DATASCIENCE/DS-PROJECTS/13_Chem_Modeling/')
#-----------------------------------------------------------------

library(rcdk)
library(dplyr)
library(ggplot2)
library(gridExtra)
library(mlr)

# ------------- Data Loading, Processing and Task Preparation ----------------
data <- read.csv('mutagendata', sep='\t', header = F)
head(data)
colnames(data) <- c('Smile.st', 'ID', 'Class')
sapply(data, class)

data <- mutate(data, Smile.st = as.character(Smile.st))
data <- mutate(data, ID = as.character(ID))

smi_parsed <- lapply(data$Smile.st, parse.smiles)
comp_fp <- vector('list',nrow(data))

# generating finger prints
for (i in 1:nrow(data)){
  comp_fp[i] <- lapply(smi_parsed[[i]], get.fingerprint, type='maccs')
}

##Convert fingerprints to matrix form
fp_matx <- fp.to.matrix(comp_fp)
comp_finger <- as.data.frame(fp_matx)

data_fp <- as.data.frame(cbind(comp_finger, data$Class))
#colnames(data_fp)[1] <- 'ID'
colnames(data_fp)[167] <- 'Class'
head(data_fp)

train <- sample_frac(data_fp, 0.8, replace=T)
rid <- as.numeric(rownames(train))
test <- data_fp[-rid,]

trainTask <- makeClassifTask(data=train, target = 'Class')
testTask <- makeClassifTask(data=test, target = 'Class')
#--------------------------------------------------------------------
#                          Feature Importance 
# much room for dimension reduction (as data is parse ). 
# But lets use all the features in the following modeling
#
# imp_feat <- generateFilterValuesData(trainTask, 
#                           method =c("information.gain","chi.squared"))
imp_feat <- generateFilterValuesData(trainTask, method =c("information.gain"))
plotFilterValues(imp_feat,n.show=50)
#--------------------------------------------------------------------
#                         Benchmarking the models
listLearners()

lrns1 <- list(
             makeLearner('classif.logreg',       predict.type = "prob", id='logreg'),
             makeLearner('classif.rpart',        predict.type = "prob", id='dt'),
             makeLearner('classif.randomForest', predict.type = "prob", id='rf'),
             makeLearner('classif.ksvm',         predict.type = "prob", id='svm'),
             makeLearner('classif.naiveBayes',   predict.type = "prob", id='nb'),
             makeLearner('classif.gbm',          predict.type = "prob", id='gbm'),
             makeLearner('classif.xgboost',      predict.type = "prob", id='xgb')
             )

lrns2 <- makeLearner('classif.knn', predict.type = "response", id='knn')

set.seed(3)
rdesc <- makeResampleDesc(method='CV', iter=5, stratify = TRUE)
bmr <- benchmark(lrns1, trainTask, rdesc, measures = acc)
bmr2 <- benchmark(lrns2, trainTask, rdesc, measures = acc)

tot_bmr <- mergeBenchmarkResultLearner(bmr,bmr2)
tot_bmr

bmr_roc = generateThreshVsPerfData(bmr, measures = list(fpr, tpr,acc), aggregate = FALSE)
plotROCCurves(bmr_roc)
#-------------------------------------------------------
#---------------------Training -------------------------
set.seed(3)
knn_learn <- makeLearner('classif.knn', predict.type = "response")
svm_learn <- makeLearner('classif.svm', predict.type = "prob")
rf_learn <- makeLearner('classif.randomForest', predict.type = "prob")

knn_model <- train(learner=knn_learn, task=trainTask)
svm_model <- train(learner=svm_learn, task=trainTask)
rf_model <- train(learner=rf_learn, task=trainTask)

# ----------------- Prediction -----------------------
set.seed(3)
knn_pred <- predict(knn_model, testTask)
svm_pred <- predict(svm_model, testTask)
rf_pred <- predict(rf_model, testTask)

# performance Evaluations

getConfMatrix(knn_pred)
getConfMatrix(rf_pred)
getConfMatrix(svm_pred)

performance(knn_pred, measures=list(acc,mmce), task=knn_model)
performance(svm_pred, measures=list(acc,mmce,auc), task=rf_model)
performance(rf_pred, measures=list(acc,mmce,auc), task=rf_model)

roc = generateThreshVsPerfData(list(rf = rf_pred, svm = svm_pred),measures = list(fpr, tpr))
plotROCCurves(roc)

#------------------------------------------------------
#---------------- RF parameter tuning --------------
getParamSet('classif.randomForest')

rf_param <- makeParamSet(
  makeIntegerParam("ntree",lower = 50, upper = 200),
  makeIntegerParam("mtry", lower = 3, upper = 10)
  )
rancontrol <- makeTuneControlRandom(maxit = 10L) # 10 -- too small but its taking too long 
rf_tune <- tuneParams(learner = 'classif.randomForest', resampling = rdesc, task = trainTask, par.set =
                        rf_param, control = rancontrol, measures = acc)
rf_tune$y
rf_tune$x

#using hyperparameters for modeling

#rf.tree <- setHyperPars(learner = 'lrn_rf', par.vals = rf_tune$x)
rf_lrn_tuned <- makeLearner('classif.randomForest', predict.type = "prob",
           par.vals = list(ntree =89, mtry = 9))
#------------------------------------------------------
#--------------------- Training w/ tuned parameters-------------
set.seed(3)

rf_mod_tuned <- train(rf_lrn_tuned, task=trainTask)
rf_pred2 <- predict(rf_mod_tuned, testTask)

getConfMatrix(rf_pred2)
performance(rf_pred2, measures=list(acc,mmce, auc), task=rf_model)   # after tuning
performance(rf_pred, measures=list(acc,mmce,auc), task=rf_model)     # before tuning

rf_roc <- generateThreshVsPerfData(list(w_tune=rf_pred2, wo_tune= rf_pred),measures=list(fpr,tpr)) 
plotROCCurves(rf_roc)
#------------------------------------------------------
