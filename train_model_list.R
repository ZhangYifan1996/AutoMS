library(reticulate)
model_lists=list()
clfname=c()
for (method in c("HBOS","iForest","KNN","LODA","LOF","OCSVM")){
  od<-import("pyod")
  if( method =="HBOS"){
    para1<-c(5,10,20,30,40,50,75,100);para2<-c(0.5)
  }
  if (method=="iForest"){
    para1<-c(50,100,150,200);para2<-c(0.2*(1:5))
  }
  if (method=="KNN"){
    para1<-c(1,5,10,15,20,50,75,100);para2<-c("largest","mean")#,"median"
  }
  if(method=="LODA"){
    para1<-c(10,20,30,40,50,75,100,150,200);para2<-c(100)
  }
  if (method=="LOF"){
    para1<-c(1:10)*10;para2<-c("minkowski")#,"manhattan","euclidean")
  }
  
  if (method=="OCSVM"){
    para1<-c(0.2*(1:5)-0.1);para2<-c("rbf","linear","poly","sigmoid")
  }
  for (i in seq_len(length(para1))){
    for (j in seq_len(length(para2))){
      if( method =="HBOS"){
        hbos<-od$models$hbos
        clf=hbos$HBOS(n_bins=as.integer(para1[i]),tol= para2[j])
      }
      if (method=="iForest"){
        iforest<-od$models$iforest
        clf=iforest$IForest(n_estimators = as.integer(para1[i]),
                            max_features =  para2[j])
      }
      if (method=="KNN"){
        knn<-od$models$knn
        clf=knn$KNN(n_neighbors=as.integer(para1[i]),method = para2[j])
      }
      if(method=="LODA"){
        loda<-od$models$loda
        clf=loda$LODA(n_bins = as.integer(para1[i]),n_random_cuts = as.integer( para2[j]))
      }
      if (method=="LOF"){
        lof<-od$models$lof
        clf=lof$LOF(n_neighbors=as.integer(para1[i]),metric=para2[j])
      }
      if (method=="OCSVM"){
        ocsvm<-od$models$ocsvm
        clf=ocsvm$OCSVM(nu=para1[i],kernel = para2[j])
      }
      clfname=c(clfname,paste0(method," (",para1[i],", ",para2[j],")"))
      model_lists=c(model_lists,clf)
    }
  }
}

names(model_lists)=clfname
joblib<-import("joblib")
joblib$dump(model_lists,filename = "saved_model_list.R")

