# AutoMS
Code for `AutoMS` proposed in the NeurIPS 2022 paper [AutoMS: Automatic Model Selection for Novelty Detection with Error Rate Control](https://openreview.net/forum?id=HIslGib8XD&noteId=w4bHLYK4ZA9&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DNeurIPS.cc%2F2022%2FConference%2FAuthors%23your-submissions))

Given an unsupervised novelty detection task on a new dataset, how can we automatically select a "best" detection model while simultaneously controlling the error rate of the best model? For novelty detection analysis, numerous detectors have been proposed to detect outliers on a new unseen dataset based on a score function trained on available clean data. However, **due to the absence of labeled data for model evaluation and comparison, there is a lack of systematic approaches that are able to select the "best" model/detector (i.e., the algorithm as well as its hyperparameters) and achieve certain error rate control simultaneously.** In this paper, we introduce a unified data-driven procedure to address this issue. The key idea is to maximize the number of detected outliers while controlling the false discovery rate (FDR) with the help of Jackknife prediction. We establish non-asymptotic bounds for the false discovery proportions and show that the proposed procedure yields valid FDR control under some mild conditions. 

* The proposed AutoMS can select the best model and simultaneously control the error rate of the best model.
Notably, AutoMS is a unified data-driven framework, and it does not rely on any labeled anomalous data for model selection. 
- To our best knowledge, it is the first effort to select the "best" model/detector with theoretical guarantees in the view of FDR control. We establish non-asymptotic bounds for the FDP and show that the proposed AutoMS yields valid FDR control. 
* The AutoMS can be easily coupled with commonly used novelty detection algorithms. Extensive numerical experiments indicate that AutoMS outperforms other methods significantly, with respect to both error rate control and detection power. 

## Requirment
We use the models/detectors that are available in the Python [PyOD](https://github.com/yzhao062/pyod).
To install requirements:
```
install.packages("reticulate") # Use Python Modules in R
library("reticulate") #Load "reticulate" package
py_install("pyod",pip=T) # Install the PyOD package in python
```
Execute different detector with a looping construct 
```
install.packages("foreach") 
```
## Construct the Set of detectors
We have already given a example of candidate model set as in [saved_model_list.R](./saved_model_list.R)
```
joblib<-import("joblib")
model.lists<-joblib$load("saved_model_list.R")
```
Or the R code for the construction of the example model list is [train_model_list.R](./train_model_list.R)
## AutoMS
Start Automatic Model Selection for novelty detection using `AutoMS` method in [main.R](./main.R). We use opt.digits dataset as an example.
Main Function is `SelectModel()`.
```
res1=SelectModel(traindata,newdata,method="SRS",alpha,calprop=0.5)
res2=SelectModel(traindata,newdata,method="JK",alpha,K=m)
```
`traindata` is used to train the score function which sample size is $m$ and dimension is $d$.
`newdata` is new and needs to detect outliers in it which sample size is $n$ and dimension is $d$.
Each row represents a data point. A matrix or data frame containing the data. The rows should be cases and the columns correspond to variables
`method` is the version of AutoMS method to be used. Use `method="JK"` for AutoMS-JK. Use `method="SRS"` for AutoMS-SRS.
`alpha` is the target FDR level $\alpha$.
`calprop` is the proportion of `traindata` used for calibration in AutoMS-SRS.
`K` is the number of groups into which the `traindata` should be split to estimate the cross-validation score in AutoMS-JK. 

If we only want to use one detector during detection, `Detection()` can be used with the given detector.
Note that res3 produce the same result as SRS-based-method in [Bates et al.(2021)](https://arxiv.org/abs/2104.08279).
```
res3=Detection(traindata,newdata,alpha,clf,method="SRS",calprop=0.5)
res4=Detection(traindata,newdata,alpha,clf,method="JK",K=m)
```
`clf` is the given detector, either one of the candidate models in `model.lists`, or a new algorithm or new hyperparameters constructed.

## Evaluation
We use FDP=1-Precision and TDP=Recall as performance measures. We say a detection procedure is superior to its competitor if it has a larger TDR for the same target FDR level $\alpha$, that is, a superior detector has a larger empirical TDR whose empirical FDR is below or around the target FDR level $\alpha$. The preset target FDR level $\alpha$ should be an acceptable level, which means all detectors that control the FDR at this level $\alpha$ should be accepted. As long as the FDR of the selected detector is below $\alpha$, we have achieved our goal on FDR control and we should focus on improving TDR. 
```
trueFDP=sum(pred.label[label==0]==1)/max(1,sum(pred.label==1))
trueTDP=sum(pred.label[label==1]==1)/max(1,sum(label==1))
```
`pred.label` is the predicted labels for each sample. `label` is the true labels for each sample. `0` represents a normal sample (inlier), while `1` is the label of the abnormal sample (outlier).
