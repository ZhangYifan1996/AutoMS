# AutoMS
Code for `AutoMS` proposed in the NeurIPS 2022 paper [AutoMS: Automatic Model Selection for Novelty Detection with Error Rate Control](https://openreview.net/forum?id=HIslGib8XD&noteId=w4bHLYK4ZA9&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DNeurIPS.cc%2F2022%2FConference%2FAuthors%23your-submissions))
## Requirment
To install requirements:
```
install.packages("foreach") # execute different detector with a looping construct 
install.packages("reticulate") # Use Python Modules in R
py_install("pyod",pip=T) # Install the [PyOD](https://github.com/yzhao062/pyod) package in python
```
## Construct the Set of detectors
We have already given a example of candidate model set as in [saved_model_list.R](./saved_model_list.R)
```
joblib<-import("joblib")
model.lists<-joblib$load("saved_model_list.R")
```
Or the R code for the construction of the example model list is [train_model_list.R](./train_model_list.R)
## AutoMS
Start Automatic Model Selection for novelty detection using `AutoMS` method in [main.R](./main.R).
Main Function is `SelectModel()`.

`traindata` is used to train the score function which sample size is $m$ and dimension is $d$.
`newdata` is new and needs to detect outliers in it which sample size is $n$ and dimension is $d$.
Each row represents a data point. A matrix or data frame containing the data. The rows should be cases and the columns correspond to variables
`method` is the version of AutoMS method to be used. Use `method="JK"` for AutoMS-JK. Use `method="SRS"` for AutoMS-SRS.  
`alpha` is the target FDR level.
`calprop` is the proportion of `traindata` used for calibration.
`K` is the number of groups into which the `traindata` should be split to estimate the cross-validation score. 
```
res1=SelectModel(traindata,newdata,method="SRS",alpha,calprop=0.5)
res2=SelectModel(traindata,newdata,method="JK",alpha,K=m)
```
If we only want to use one detector during detection, `Detection()` can be used with the given detector.
`clf` is the given detector, either one of the candidate models in `model.lists`, or a new algorithm or new hyperparameters constructed.
```
res3=Detection(traindata,newdata,alpha,clf,method="SRS",calprop=0.5)
res4=Detection(traindata,newdata,alpha,clf,method="JK",K=m)
```
Note that res3 produce the same result as SRS-based-method in [Bates et al.(2021)](https://arxiv.org/abs/2104.08279).
## Evaluation
We use FDP=1-Precision and TDP=Recall as performance measures. We say a detection procedure is superior to its competitor if it has a larger TDR for the same target FDR level $\alpha$, that is, a superior detector has a larger empirical TDR whose empirical FDR is below or around the target FDR level $\alpha$. The preset target FDR level $\alpha$ should be an acceptable level, which means all detectors that control the FDR at this level α should be accepted. As long as the FDR of the selected detector is below $\alpha$, we have achieved our goal on FDR control and we should focus on improving TDR. 
```
trueFDP=sum(pred.label[label==0]==1)/max(1,sum(pred.label==1))
trueTDP=sum(pred.label[label==1]==1)/max(1,sum(label==1))
```
`pred.label` is the predicted labels for each sample. `label` is the true labels for each sample. `0` represents a normal sample (inlier), while `1` is the label of the abnormal sample (outlier).
