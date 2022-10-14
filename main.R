# install.packages("reticulate") # Use Python Modules in R
# library("reticulate") #Load "reticulate" package
# py_install("pyod",pip=T) # Install the PyOD package in python
# py_install("joblib",pip=T)
# install.packages("foreach") #Execute different detector with a looping construct
library(reticulate)
library(foreach)
######## Functions #######################################################################
GetJKScore<-function(traindata,clf,K=m){
  # detector is used in "PyOD" python package (https://github.com/yzhao062/pyod)
  if (!inherits(clf, "pyod.models.base.BaseDetector")) 
    warning("detector is not available in 'PyOD' python package")
  
  # Split $m$ datapoint into $K$ folds.
  m <- nrow(traindata)
  if ((K > m) || (K <= 1)) 
    stop("'K' outside allowable range")
  K.o <- K
  K <- round(K)
  kvals <- unique(round(m/(1L:floor(m/2))))
  temp <- abs(kvals - K)
  if (!any(temp == 0)) 
    K <- kvals[temp == min(temp)][1L]
  if (K != K.o) 
    warning(gettextf("'K' has been set to %f", K), 
            domain = NA)
  f <- ceiling(m/K)
  v <- sample(rep(1L:K, f), m,replace = FALSE)
  mv <- max(v)
  
  # compute score on the whole traindata
  s.train <- list()
  for (i in seq_len(mv)) {
    j.out <- seq_len(m)[(v == i)]
    j.in <- seq_len(m)[(v != i)]
    # use subset of traindata D[-j] learn score function S[-j]
    model <- clf$fit(traindata[j.in, , drop = FALSE])
    # compute score at D[j]
    s.train[[i]] <- model$decision_function(traindata[j.out, , drop = FALSE])
  }
  s.train=unlist(s.train)
  return(s.train)
}
GetScore<-function(traindata,clf,newdata){
  # detector is used in "PyOD" python package (https://github.com/yzhao062/pyod)
  if (!inherits(clf, "pyod.models.base.BaseDetector")) 
    warning("detector is not available in 'PyOD' python package")
  if (missing(newdata) || is.null(newdata)) 
    stop("Can not find newdata")
  if (is.vector(newdata)) {
    newdata <- array_reshape(newdata,dim=c(-1,length(newdata)))
  }
  if (ncol(newdata) != ncol(traindata) && !(missing(newdata) || is.null(newdata)))
    warning("detection may be misleading due to different dimensions of traindata and newdata")
  
  # learn score function based on the traindata 
  model=clf$fit(traindata)
  # evaluate scores for newdata
  s.test=c(model$decision_function(newdata)) 
  return(s.test)
}
GetThreshold<-function(s.train,s.test,alpha){
  m <- length(s.train)
  n <- length(s.test)
  # Given a threshold $t$, get a corresponding FDP
  EstimateFDP<-function(t,s.train,s.test){
    # estimate P value
    Pvalue <- sum(s.train>=t)/m 
    # estimate FDP
    FDP <- (n*Pvalue)/max(1,sum(s.test>=t)) 
    return(FDP)
  }
  s.temp <- unique(c(s.train,s.test))
  FDP <- sapply(s.temp, EstimateFDP, s.train=s.train,s.test=s.test)
  # find the minimum of $t$, s.t. estimated FDP<=alpha
  ok=which(FDP<=alpha)
  thre=ifelse(length(ok)>0,min(s.temp[ok]),Inf)
  return(thre)
}

# For a given detector "clf", return the detection result
Detection<-function(traindata,newdata,alpha,clf,method,K,calprop){
  if (! (method == "JK"|| method == "SRS") ){
    stop(gettextf(" method = '%s' is not supported. Using 'JK' or 'SRS'", 
                   method), domain = NA)
  } else{
    if (method == "JK"){
      s.train <- GetJKScore(traindata = traindata,clf = clf,K = K)
      s.test <- GetScore(traindata = traindata,clf = clf,newdata = newdata)
    }
    if (method == "SRS"){
      m <- nrow(traindata);mcal=floor(m*calprop)
      i.cal <- sample(seq_len(m),mcal,replace = FALSE)
      # use m-mcal traindata to train
      trndata <- traindata[-i.cal, , drop = FALSE]
      #use mcal traindata to calibration
      caldata <- traindata[i.cal, , drop = FALSE]
      s.train <- GetScore(traindata = trndata,clf = clf,newdata = caldata)
      s.test <- GetScore(traindata = trndata,clf = clf,newdata = newdata)
    }
    L <- GetThreshold(s.train = s.train,s.test = s.test,alpha = alpha)
    pred.label <- ifelse(s.test<L,0,1)
    num.out <- sum(pred.label)
    return(list(pred.label=pred.label,num.out=num.out))
  }
}
# The best model is the one with the largest number of detected outliers
GetBestModel<-function(record,model.lists){
  o=sapply(record, function(a) a$num.out)
  opt.id <- which.max(o)
  opt.model.preLabel <- record[[opt.id]]$pred.label
  opt.model <- names(model.lists)[opt.id]
  print(paste0("best model ",opt.model))
  return(list(model=opt.model,pred.label=opt.model.preLabel))
}
# Give the top "n_recommend" models as recommendations
OrderModel<-function(record,model.lists,n_recommend){
  w=sapply(record, function(a) a$num.out)
  temp=data.frame(names(model.lists),num.out=w)
  temp=temp[order(temp$num.out,decreasing = T),]
  return(temp[1:n_recommend,])
}
# Select the best model for "newdata" with the target FDR level "alpha"
SelectModel<-function(traindata,newdata,alpha,method,K,calprop){
  model.lists<-joblib$load("saved_model_list.R")
  record=foreach(clf=model.lists,.packages = c("reticulate")) %do%
    Detection(traindata=traindata,newdata=newdata,alpha=alpha,clf,method=method,K=K,calprop=calprop)
  best.model <- GetBestModel(record = record,model.lists = model.lists)
  recom.model <- OrderModel(record = record,model.lists = model.lists,n_recommend = 5)
  return(list(best.model.pred.label=best.model$pred.label,recom.model=recom.model))
}
# Known the true label of "newdata", evaluate the prediction result
getFDR<-function(pred.label,label){
  trueFDP=sum(pred.label[label==0]==1)/max(1,sum(pred.label==1))
  trueTDP=sum(pred.label[label==1]==1)/max(1,sum(label==1))
  return(c(FDP=trueFDP,TDP=trueTDP))
}

######## Start AutoMS #############################################################################
#traindata is used to train the score function. m*d
#newdata is new and needs to detect outliers in it. n*d
#Each row represents a data point. A matrix or data frame containing the data. The rows should be cases and the columns correspond to variables
#clf is one of the candidate models
#alpha is the target FDR level
# calprop is the proportion of traindata used for calibration. 
#K is the number of groups into which the traindata should be split to estimate the cross-validation score. 

####load data
load("./dataset/opt.digitstrain.Rdata") # traindata
load("./dataset/opt.digitstest.Rdata") # newdata
traindata=trainset[[1]]
newdata=testset[[1]]

joblib<-import("joblib")
# produce the selected detector and result
res1=SelectModel(traindata=traindata,newdata=newdata,alpha=0.1,method="SRS",calprop=0.5)
res2=SelectModel(traindata=traindata,newdata=newdata,alpha=0.1,method="JK",K=100)

######## Evaluate ####################################################################
load("./dataset/opt.digitslabel.Rdata") #true label of "newdata"
# evaluate via the FDP and TDP
pred.label1<-res1$best.model.pred.label
getFDR(pred.label1,label) 
pred.label2<-res2$best.model.pred.label
getFDR(pred.label2,label) 

############# Predict with a given detector #########################################################################

# given a detector "OCSVM(0.5,poly)", produce detection result
od<-import("pyod")
ocsvm<-od$models$ocsvm
clf <- ocsvm$OCSVM(nu=as.numeric(0.5),kernel = "poly")
m<-dim(traindata)[1]
res3=Detection(traindata=traindata,newdata=newdata,alpha=0.1,clf=clf,method="SRS",calprop=0.5)
res4=Detection(traindata=traindata,newdata=newdata,alpha=0.1,clf=clf,method="JK",K=m)

######## Evaluate 
load("./dataset/opt.digitslabel.Rdata") #true label of "newdata"
# evaluate via the FDP and TDP
pred.label1<-res3$pred.label
getFDR(pred.label1,label) #one time:0.032 0.652 another time:0.063 0.747
#### different results are caused by random splitting

pred.label2<-res4$pred.label
getFDR(pred.label2,label) #0.038 0.734
#### Jackknife technique only has one result
