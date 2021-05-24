
names(HR_Analytics)
summary(HR_Analytics)
str(HR_Analytics)
sum(is.na(HR_Analytics))

library(dplyr)
library(DataExplorer)
library(caTools)
library(car)
library(scales)
library(DMwR)
library(corrplot)
library(caret)
library(rpart)
library(rpart.plot)
library(e1071)
library(pROC)
library(ROCR)
library(ipred)
library(gbm)
library(xgboost)
library(randomForest)
library(ranger)
library(mlr)
library(vip)
attach(HR_Analytics)

#Variable Transformation
HR_Analytics$Attrition = as.factor(HR_Analytics$Attrition)
HR_Analytics$Involvement = factor(HR_Analytics$Involvement, ordered = TRUE, levels = c("1","2","3","4","5"))
HR_Analytics$TravelProfile = as.factor(HR_Analytics$TravelProfile)
HR_Analytics$Designation = as.factor(HR_Analytics$Designation)
HR_Analytics$JobSatisfaction = factor(HR_Analytics$JobSatisfaction, ordered = TRUE, levels = c("1","2","3","4","5"))
HR_Analytics$Department = as.factor(HR_Analytics$Department)
HR_Analytics$MaritalStatus = as.factor(HR_Analytics$MaritalStatus)
HR_Analytics$MaritalStatus = recode_factor(HR_Analytics$MaritalStatus, 'M' = "Married")
HR_Analytics$EducationField = as.factor(HR_Analytics$EducationField)
HR_Analytics$Gender = recode_factor(HR_Analytics$Gender, 'F' = 'Female')
HR_Analytics$HourlnWeek = as.factor(HR_Analytics$HourlnWeek)
HR_Analytics$Designation = as.factor(HR_Analytics$Designation)
HR_Analytics$OverTime = as.factor(HR_Analytics$OverTime)

#Univariate Analysis
plot(HR_Analytics$Attrition)
plot_histogram(HR_Analytics$Age)
plot(HR_Analytics$TravelProfile)
plot(HR_Analytics$Department)
plot_histogram(HomeToWork)
plot(HR_Analytics$EducationField)
plot(HR_Analytics$Gender)
plot(HR_Analytics$MaritalStatus)
plot(HR_Analytics$HourlnWeek)
plot(HR_Analytics$Involvement)
plot(HR_Analytics$Designation)
plot(HR_Analytics$JobSatisfaction)
plot_histogram(HR_Analytics$MonthlyIncome)
plot_histogram(HR_Analytics$NumCompaniesWorked)
plot(HR_Analytics$OverTime)
plot_histogram(HR_Analytics$SalaryHikelastYear)
plot_histogram(HR_Analytics$WorkExperience)
plot_histogram(HR_Analytics$LastPromotion)
plot_histogram(HR_Analytics$CurrentProfile)

#Bi-variate analysis
plot(HR_Analytics$Attrition, HR_Analytics$Age, xlab = "Attrition", ylab = "Age")
plot(HR_Analytics$TravelProfile,HR_Analytics$Attrition, xlab = "Travel Profile", ylab = "Attrition")
plot(HR_Analytics$Department, HR_Analytics$Attrition, xlab = "Department", ylab = "Attrition")
plot(HR_Analytics$Attrition, HR_Analytics$HomeToWork,xlab = "Attrition", ylab = "Home to Work")
plot(HR_Analytics$EducationField, HR_Analytics$Attrition, xlab = "Education Field", ylab = "Attrition")
plot(HR_Analytics$Gender,HR_Analytics$Attrition, xlab = "Gender", ylab = "Attrition")
plot(HR_Analytics$Involvement, HR_Analytics$Attrition, xlab = "Involvement", ylab = "Attrition")
plot(HR_Analytics$Designation, HR_Analytics$Attrition, xlab = "Designation", ylab = "Attrition")
plot(HR_Analytics$JobSatisfaction, HR_Analytics$Attrition, xlab = "Job Satisfaction", ylab = "Attrition")
plot(HR_Analytics$MaritalStatus, HR_Analytics$Attrition, xlab = "Marital Status", ylab = "Attrition")
plot(HR_Analytics$Attrition, HR_Analytics$MonthlyIncome, xlab = "Attrition", ylab = "Monthly Income")
plot(HR_Analytics$Attrition, HR_Analytics$NumCompaniesWorked, xlab = "Attrition", ylab = "Number of companies worked")
plot(HR_Analytics$Attrition, HR_Analytics$OverTime, xlab = "Attrition", ylab = "Overtime")
plot(HR_Analytics$Attrition, HR_Analytics$SalaryHikelastYear, xlab = "Attrition", ylab = "Salary Hike Last Year")
plot(HR_Analytics$Attrition, HR_Analytics$WorkExperience, xlab = "Attrition", ylab = "Work Experience")
plot(HR_Analytics$Attrition, HR_Analytics$LastPromotion, xlab = "Attrition", ylab = "Last Promotion")
plot(HR_Analytics$Attrition, HR_Analytics$CurrentProfile, xlab = "Attrition", ylab = "Current Profile")

prop.table(table(HR_Analytics$Gender))
table(HR_Analytics$MaritalStatus)
prop.table(table(HR_Analytics$OverTime))

#removing unwanted variables
HR_Analytics = (HR_Analytics[,-1])
View(HR_Analytics)
#Missing value treatment
plot_intro(HR_Analytics)

HR_Analytics = as.data.frame(HR_Analytics)

for(a in 1:19)
 {
  HR_Analytics[,a]= as.numeric(HR_Analytics[,a])
  HR_Analytics[is.na(HR_Analytics[,a]),a] = median(HR_Analytics[,a],na.rm = TRUE)
}

plot_intro(HR_Analytics)

#Outlier Treatment
Outval = boxplot(HR_Analytics)$out

for(b in 2:19)
{
  q = quantile(HR_Analytics[,b], c(0.1,0.99))
  HR_Analytics[,b] = squish(HR_Analytics[,b],q)
}  

Outval = boxplot(HR_Analytics)$out

#Correlation plot
HR_Analytics.new = data.frame(lapply(HR_Analytics, as.numeric)) 
corrplot(cor(HR_Analytics.new, use = "pairwise.complete.obs"))

#Multicollinearity
vifmatrix = vif(lm(HR_Analytics.new$Attrition~., data = HR_Analytics.new))
vifmatrix

#Checking for Imbalance
HR_Analytics$Attrition = as.factor(HR_Analytics$Attrition)
HR_Analytics$Attrition = ifelse(HR_Analytics$Attrition=="1", 0,1)
table(HR_Analytics$Attrition)

prop.table(table(HR_Analytics$Attrition))

#SMOTE Data Preperation
set.seed(500)
split = sample.split(HR_Analytics$Attrition, SplitRatio = 0.70)
HR_Analytics.Train = subset(HR_Analytics, split == TRUE)
HR_Analytics.Test = subset(HR_Analytics, split == FALSE)
table(HR_Analytics$Attrition)

#SMOTE-Balancing the data
attach(HR_Analytics.Train)
str(HR_Analytics.Train)
HR_Analytics.Train$Attrition = as.factor(HR_Analytics.Train$Attrition)
balanced.data = SMOTE(Attrition ~., HR_Analytics.Train, perc.over = 250, perc.under = 150)
prop.table(table(balanced.data$Attrition))

##Model Building
#Logistic Regression 
set.seed(300)
Model1 = glm(Attrition~., family = binomial, data = balanced.data)
Model1
summary(Model1)
Predict.Model1= predict(Model1,newdata = HR_Analytics.Test, type = "response")
str(Predict.Model1)
table.Model1= table(HR_Analytics.Test$Attrition, Predict.Model1 >0.7)


table.Model1
sum(diag(table.Model1))/sum(table.Model1)
roc(balanced.data$Attrition,Model1$fitted.values)
plot(roc(balanced.data$Attrition,Model1$fitted.values))

set.seed(123)
Model1.final = glm(Attrition~.-HourlnWeek -Designation -WorkExperience, data = balanced.data,
                     family = binomial)
summary(Model1.final)


Predict.Model1.final = predict(Model1.final,newdata = HR_Analytics.Test, type = "response")
str(Predict.Model1.final)
table.Model1.final = table(Predict.Model1.final >0.7, HR_Analytics.Test$Attrition)
table.Model1.final
str(HR_Analytics.Test$Attrition)
str(balanced.data$Attrition)
str(Predict.Model1.final)
sum(diag(table.Model1.final))/sum(table.Model1.final)
roc(balanced.data$Attrition,Model1.final$fitted.values)
plot(roc(balanced.data$Attrition,Model1.final$fitted.values))

#CART
set.seed(200)
cart.model = rpart(Attrition~., data = balanced.data, method = "class", parms = list(split = "gini"))
plotcp(cart.model)
rpart.plot(cart.model, cex = 0.6)
cart.model$cptable
cart.model$variable.importance
Predict.cart = predict(cart.model, HR_Analytics.Test, type = "prob")
Predict.cart_Prob1 = Predict.cart[,1]

Predict.cart$predict_class = predict(cart.model, HR_Analytics.Test, type = "class")
x= HR_Analytics.Test$Attrition
Predict.cart$predict_score = predict(cart.model, HR_Analytics.Test, type = "prob")
confusionMatrix(Predict.cart$predict_class, as.factor(x))

##ROC AND AUC
Predict.cart$predict_class= as.numeric(Predict.cart$predict_class)
roc(Predict.cart$predict_class, HR_Analytics.Test$Attrition)
plot(roc(Predict.cart$predict_class, HR_Analytics.Test$Attrition))

#Pruning
pruned.model= prune(cart.model, cp= 0.017)
printcp(pruned.model)
rpart.plot(pruned.model, cex = 0.65)

CartPred = predict(pruned.model, HR_Analytics.Test, type = "prob")
CartPred_Prob1 = CartPred[,1]
head(CartPred_Prob1,10)
CartTest = HR_Analytics.Test
CartTest$predict_class = predict(pruned.model, CartTest, type = "class")
x= CartTest$Attrition
CartTest$predict_score = predict(pruned.model, CartTest, type = "prob")
confusionMatrix(CartTest$predict_class, as.factor(x))
pruned.model$variable.importance
##ROC AND AUC
CartTest$predict_class = as.numeric(CartTest$predict_class)
roc(CartTest$predict_class, CartTest$Attrition)
str(CartTest$Attrition)
CartTest$Attrition = as.numeric(CartTest$Attrition)
CartTest$Attrition[CartTest$Attrition==1]=0
CartTest$Attrition[CartTest$Attrition==2]=1
CartTest$predict_class[CartTest$predict_class==1]=0
CartTest$predict_class[CartTest$predict_class==2]=1
plot(roc(CartTest$predict_class, CartTest$Attrition))


##KNN
set.seed(700)

knn_model2 <- knn3(Attrition~.,data = balanced.data,k=6)

knn.results<- predict(knn_model2,newdata=HR_Analytics.Test[2:19],type = "prob")

knn_results <- as.data.frame(knn.results)
knn_results['result'] <- ifelse(knn_results$`1`>0.5,1,0)
str(knn_results$result)
confusionMatrix(as.factor(knn_results$result), as.factor(HR_Analytics.Test$Attrition))
roc(HR_Analytics.Test$Attrition,knn_results$result)
plot(roc(HR_Analytics.Test$Attrition,knn_results$result))


##Naive Bayes
set.seed(500)
NBmodel.bal = naiveBayes(Attrition ~., data = balanced.data)
NBmodel.bal
NBpredTest.bal = predict(NBmodel.bal, newdata = HR_Analytics.Test)
tabNB.bal = table(NBpredTest.bal, HR_Analytics.Test$Attrition)
confusionMatrix(tabNB.bal) 
str(NBpredTest.bal)
NBpredTest.bal = as.numeric(NBpredTest.bal)
roc(HR_Analytics.Test$Attrition, NBpredTest.bal)
plot(roc(HR_Analytics.Test$Attrition, NBpredTest.bal))



#Bagging

HR_Analytics.Test$Attrition = as.factor(HR_Analytics.Test$Attrition)
set.seed(1000)
mod.bagging = bagging(Attrition~., data = balanced.data,
                      control = rpart.control(maxdepth = 3, minsplit = 4))
bag.pred = predict(mod.bagging, HR_Analytics.Test)
confusionMatrix(bag.pred, HR_Analytics.Test$Attrition)
str(bag.pred)
str(HR_Analytics.Test$Attrition)
bag.pred = as.numeric(bag.pred)
HR_Analytics.Test$Attrition = as.numeric(HR_Analytics.Test$Attrition)
roc(bag.pred, HR_Analytics.Test$Attrition)
plot(roc(bag.pred, HR_Analytics.Test$Attrition))

#Boosting-gbm
str(balanced.data$Attrition)

View(balanced.data$Attrition)
str(HR_Analytics.Test$Attrition)
balanced.data$Attrition = as.numeric(balanced.data$Attrition)
balanced.data$Attrition[balanced.data$Attrition==1]=0
balanced.data$Attrition[balanced.data$Attrition==2]=1
set.seed(2000)
mod.boost = gbm(Attrition~., data = balanced.data, distribution = "bernoulli", n.trees = 5000,
                interaction.depth = 6, shrinkage = 0.01, cv.folds = 5, n.cores = NULL, verbose = FALSE)
summary(mod.boost)

boost.pred = predict(mod.boost, HR_Analytics.Test, n.trees= 5000, type  = "response")
y_pred_num = ifelse(boost.pred >0.5, 1,0)
y_pred = factor(y_pred_num, levels = c(0,1))
table(y_pred, HR_Analytics.Test$Attrition)
str(HR_Analytics.Test$Attrition)
HR_Analytics.Test$Attrition[HR_Analytics.Test$Attrition==1]=0
HR_Analytics.Test$Attrition[HR_Analytics.Test$Attrition==2]=1
HR_Analytics.Test$Attrition = as.factor(HR_Analytics.Test$Attrition)
str(y_pred)
y_pred = as.numeric(y_pred)
y_pred[y_pred==1]=0
y_pred[y_pred==2]=1
y_pred = as.factor(y_pred)

confusionMatrix(y_pred, HR_Analytics.Test$Attrition)

y_pred = as.numeric(y_pred)
HR_Analytics.Test$Attrition = as.factor(HR_Analytics.Test$Attrition)
roc(HR_Analytics.Test$Attrition, y_pred)
plot(roc(HR_Analytics.Test$Attrition, y_pred))

gbm.perf(mod.boost, method = "cv")



#Boosting-xgboost

feature_train = as.matrix(balanced.data[,2:19])
label_train = as.matrix(balanced.data$Attrition)
label_train
set.seed(5000)
xgb.mod = xgboost(
  data = feature_train,
  label = label_train,
  eta = 0.03,
  max_depth = 5,
  min_child_weight = 3,
  nrounds = 5000,
  objective = "binary:logistic",
  verbose = 0,
  early_stopping_rounds = 10
)

HR_Analytics.Test$Attrition = as.numeric(HR_Analytics.Test$Attrition)
str(HR_Analytics.Test$Attrition)
HR_Analytics.Test$Attrition[HR_Analytics.Test$Attrition==1]=0
HR_Analytics.Test$Attrition[HR_Analytics.Test$Attrition==2]=1
feature_test = as.matrix(HR_Analytics.Test[,c(2:19)])
xgb.pred = predict(xgb.mod, feature_test)
tabxgb = table(xgb.pred>0.5, HR_Analytics.Test$Attrition)
tabxgb
HR_Analytics.Test$Attrition

xgb.pred = predict(xgb.mod, feature_test)
tabxgb = table(xgb.pred>0.5, HR_Analytics.Test$Attrition)

sum(diag(tabxgb))/sum(tabxgb)
roc(HR_Analytics.Test$Attrition, xgb.pred)
plot(roc(HR_Analytics.Test$Attrition, xgb.pred))
VarImp_xgb = xgb.importance(feature_names = colnames(balanced.data), model = xgb.mod)
xgb.plot.importance(VarImp_xgb)

#Tuning XGBoost
fact_col = colnames(balanced.data)[sapply(balanced.data,is.character)]
for(i in fact_col) set(balanced.data,j=i,value = factor(balanced.data[[i]]))
for (i in fact_col) set(HR_Analytics.Test,j=i,value = factor(HR_Analytics.Test[[i]]))

#create tasks
balanced.data$Attrition = as.factor(balanced.data$Attrition)
str(HR_Analytics.Test$Attrition)
HR_Analytics.Test$Attrition=as.factor(HR_Analytics.Test$Attrition)

traintask = makeClassifTask (data = balanced.data,target = "Attrition")
testtask = makeClassifTask (data = HR_Analytics.Test,target = "Attrition")

#one hot encoding`<br/> 
traintask = createDummyFeatures (obj = traintask) 
testtask = createDummyFeatures (obj = testtask)
lrn = makeLearner("classif.xgboost",predict.type = "response")
lrn$par.vals = list( objective="binary:logistic", eval_metric="error", nrounds=100L, eta=0.1)

#set parameter space
params = makeParamSet( makeDiscreteParam("booster",values = c("gbtree","gblinear")), 
                        makeIntegerParam("max_depth",lower = 3L,upper = 10L), 
                        makeNumericParam("min_child_weight",lower = 1L,upper = 10L), 
                        makeNumericParam("subsample",lower = 0.5,upper = 1), 
                        makeNumericParam("colsample_bytree",lower = 0.5,upper = 1))

#set resampling strategy
rdesc = makeResampleDesc("CV",stratify = T,iters=5L)
ctrl = makeTuneControlRandom(maxit = 10L)

library(parallel)
library(parallelMap) 
parallelStartSocket(cpus = detectCores())

#parameter tuning
mytune = tuneParams(learner = lrn, task = traintask, resampling = rdesc, measures = acc, 
                     par.set = params, control = ctrl, show.info = T)
mytune$y 
#set hyperparameters
lrn_tune = setHyperPars(lrn,par.vals = mytune$x)

#train model
xgmodel = train(learner = lrn_tune,task = traintask)

#predict model
xgpred = predict(xgmodel,testtask)
confusionMatrix(xgpred$data$response,xgpred$data$truth)
xgpred$data$response = as.numeric(xgpred$data$response)
xgpred$data$truth = as.numeric(xgpred$data$truth)
roc(xgpred$data$response,xgpred$data$truth)
plot(roc(xgpred$data$response,xgpred$data$truth))

vip(xgmodel)

#Random Forest

balanced.data$Attrition = as.factor(balanced.data$Attrition)
HR_Analytics.Test$Attrition = as.factor(HR_Analytics.Test$Attrition)
set.seed(1200)

RandomForest_model = randomForest(Attrition~., data = balanced.data)
print(RandomForest_model)

err = RandomForest_model$err.rate
head(err)

oob_err = err[nrow(err), "OOB"]
print(oob_err)

plot(RandomForest_model)
legend(x="topright", legend = colnames(err), fill = 1:ncol(err))

ranfost_pred = predict(RandomForest_model, HR_Analytics.Test, type = "prob")[,1]

HR_Analytics.Test$RFpred = ifelse(ranfost_pred>=0.8,"1","0")

HR_Analytics.Test$RFpred = as.factor(HR_Analytics.Test$RFpred)

levels(HR_Analytics.Test$RFpred)
str(HR_Analytics.Test$Attrition)

RFConfusion_Matx = confusionMatrix(HR_Analytics.Test$RFpred, HR_Analytics.Test$Attrition)
RFConfusion_Matx

str(HR_Analytics.Test$RFpred)
HR_Analytics.Test$RFpred = as.numeric(HR_Analytics.Test$RFpred)
str(HR_Analytics.Test$Attrition)
HR_Analytics.Test$Attrition = as.numeric(HR_Analytics.Test$Attrition)
roc(HR_Analytics.Test$RFpred, HR_Analytics.Test$Attrition)
plot(roc(HR_Analytics.Test$RFpred, HR_Analytics.Test$Attrition))

#tuning
set.seed(300)

tuned_RandFors = tuneRF(x = subset(balanced.data, select = -Attrition),
                        y= balanced.data$Attrition, 
                        ntreeTry = 501, doBest = T)

print(tuned_RandFors)


#Tuning RF with ranger
set.seed(808)
#re-invoking the caret library may be required

RGModel = train(Attrition~., data = balanced.data, tuneLength = 3, method = "ranger", 
                trControl= trainControl(method = 'cv',number = 5,verboseIter = FALSE))
RGModel

plot(RGModel)

#Tuning Ranger Grid

tuneGrid = data.frame(.mtry = c(2,10,18), .splitrule = "gini", .min.node.size=1)

set.seed(10000)
RFgrid.model = train(Attrition~., data = balanced.data, tuneGrid = tuneGrid,
                     method = "ranger", 
                     trControl= trainControl(method = 'cv',
                                             number = 5, verboseIter = FALSE))
RFgrid.model

plot(RFgrid.model)

#Refined Ranger Model

set.seed(100)

RangeModel = ranger(Attrition~., data = balanced.data, num.trees = 511,
                    mtry = 2, min.node.size = 1, verbose = FALSE, importance = "impurity")

RangeModel

range_pred = predict(RangeModel, HR_Analytics.Test)



table(HR_Analytics.Test$Attrition, range_pred$predictions)


str(range_pred$predictions)
range_pred$predictions = as.numeric(range_pred$predictions)
range_pred$predictions[range_pred$predictions==1]=0
range_pred$predictions[range_pred$predictions==2]=1
range_pred$predictions = as.factor(range_pred$predictions)
str(HR_Analytics.Test$Attrition)
HR_Analytics.Test$Attrition[HR_Analytics.Test$Attrition==1]=0
HR_Analytics.Test$Attrition[HR_Analytics.Test$Attrition==2]=1
HR_Analytics.Test$Attrition = as.factor(HR_Analytics.Test$Attrition)

RangeConMatx = confusionMatrix(range_pred$predictions, 
                               HR_Analytics.Test$Attrition, positive = "1")
RangeConMatx
RangeModel$variable.importance
vip(RangeModel)

#AUC-ROC Curve

PredictionLabels = as.numeric(range_pred$predictions)

ActualLabels = as.numeric(HR_Analytics.Test$Attrition)
require(pROC)

ROC.curve = roc(PredictionLabels, ActualLabels)

plot(ROC.curve)
auc(ROC.curve)
