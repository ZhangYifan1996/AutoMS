# AutoMS
Code for `AutoMS` proposed in the NeurIPS 2022 paper [AutoMS: Automatic Model Selection for Novelty Detection with Error Rate Control](https://openreview.net/forum?id=HIslGib8XD&noteId=w4bHLYK4ZA9&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DNeurIPS.cc%2F2022%2FConference%2FAuthors%23your-submissions))
## Requirment
To install requirements:
```
install.packages("foreach") # execute different detector with a looping construct 
install.packages("reticulate") # Use Python Modules in R
py_install("pyod",pip=T) # Install the PyOD package in python
```
## Construct the Set of detectors
We have already given a example of candidate model set as in [saved_model_list.R](./saved_model_list.R)
```
joblib<-import("joblib")
model.lists<-joblib$load("saved_model_list.R")
```
Or the R code for the construction of the example model list is [train_model_list.R](./train_model_list.R)
