import numpy as np
import matplotlib.pyplot as plt
import math
# import json
# from sklearn.grid_search import GridSearchCV
from sklearn.externals import joblib
from sklearn import ensemble
from sklearn.svm import SVR
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error

algorithms = {
	'GRADIENT_BOOSTING' : 0,
	'DEEP_LEARNING' : 1,
	'LINEAR_REGRESSION' : 2,
	'RANDOM_FOREST' : 3,
	'SVM' : 4
}

MODEL_STORAGE_EXTENSION = ".pkl"

def parseTrainingSet(training):

	a=0
	nIgnored=len(training["ignoredFeatures"])
	training["xTrain"] = []
	training["yTrain"] = []
	for row in open(training["file"]["path"],"rb"):	
		row = row.replace("\n","")
		col=row.split(",")
		nCols=len(col)
		if a==0:
			training["essentialFeatures"] = []
			efCount = 0
			for i in range(nCols):
				if col[i]==training["output"]["name"]:
					training["output"]["index"]=i
					continue
				f1 = 0
				for j in range(nIgnored):
					if col[i]==training["ignoredFeatures"][j]["name"]:	
						training["ignoredFeatures"][j]["index"]=i
						f1 = 1
						break
				if f1==0:
					if col[i]==training["weights"]["name"]:
						training["weights"]["index"]=i
						f1 = 1
				if f1==0:
					training["essentialFeatures"].append({"name":col[i],"index":i})	

		else:
			temp = []
			for i in range(nCols) :
				flag = 1
				for j in training["ignoredFeatures"] :
					
					if i==j["index"]:
						flag = 0
						break
				if flag==1 and i==training["weights"]["index"]:
					training["weights"]["values"].append(float(col[i]))
				elif flag==1 and i!=training["output"]["index"]:
					temp.append((float(col[i])))
			training["xTrain"].append(temp)
			
			training["yTrain"].append(float(col[training["output"]["index"]]))
		a+=1
	return training

def buildModel (training, model):
	if model["algorithm"] == algorithms["GRADIENT_BOOSTING"] :
		return buildGBMModel(training,model)
	elif model["algorithm"] == algorithms["LINEAR_REGRESSION"] :
		return buildLRModel(training,model)
	elif model["algorithm"] == algorithms["DEEP_LEARNING"] :
		return buildDLModel(training,model)
	elif model["algorithm"] == algorithms["RANDOM_FOREST"] :
		return buildRFModel(training,model)
	elif model["algorithm"] == algorithms["SVM"] :
		return buildSVMModel(training,model)

def buildGBMModel (training, model) :
	if "parameters" not in model:
		model["parameters"]={'subsample':0.5,'n_estimators':100,'learning_rate':0.1,'min_samples_split':2,'max_depth':8,'min_samples_leaf':15,'max_features':0.1,'loss':'ls'} 

	clf=ensemble.GradientBoostingRegressor(**model["parameters"])
	if len(training["weights"]["values"])==0:
		clf.fit(training["xTrain"],training["yTrain"])
	else:
		clf.fit(training["xTrain"],training["yTrain"],sample_weight=training["weights"]["values"])
	joblib.dump(clf,model["file"]["path"])
	return clf
			
def buildRFModel (training, model) :
	print("Random Forest Model is not supported currently.")

def buildDLModel (training, model) :
	print("Deep Learning Model is not supported currently.")

def buildLRModel (training, model) :
	print("Linear Regression Model is not supported currently.")

def buildSVMModel (training, model) :
	clf = SVR(C=1.0,epsilon=0.2)
	clf.fit(training["xTrain"],training["yTrain"])
	joblib.dump(clf,model["file"]["path"])
	return clf

def parsePredictionSet(prediction):
	a=0
	nIgnored=len(prediction["ignoredFeatures"])
	prediction["xTest"] = []
	prediction["yTest"] = []
	for row in open(prediction["file"]["path"],"rb"):	
		row = row.replace("\n","")
		col=row.split(",")
		nCols=len(col)
		if a==0:
			prediction["essentialFeatures"] = []
			efCount = 0
			for i in range(nCols):
				if col[i]==prediction["output"]["name"]:
					prediction["output"]["index"]=i
					continue
				f1 = 0
				for j in range(nIgnored):
					if col[i]==prediction["ignoredFeatures"][j]["name"]:	
						prediction["ignoredFeatures"][j]["index"]=i
						f1 = 1
						break
				if f1==0:
					prediction["essentialFeatures"].append({"name":col[i],"index":i})	
			print prediction["output"]["index"]
		else:
			temp = []
			for i in range(nCols) :
				flag = 1
				for j in prediction["ignoredFeatures"] :
					if i==j["index"]:
						flag = 0
						break
				if flag==1 and i!=prediction["output"]["index"]:
					#print(col)
					temp.append(col[i])
		
			prediction["xTest"].append(temp)
			# prediction["yTest"].append(float(col[prediction["output"]["index"]]))
		a+=1
	return prediction

def makePrediction(prediction,model) :
		clf=joblib.load(model["file"]["path"])
		# print clf.score(prediction["xTest"],prediction["yTest"])
  		return clf.predict(prediction["xTest"])

# def gridSearch(training,param_grid):
# 	# //if !param_grid:
#   	param_grid = {'n_estimators':[p for p in range(20,3001,10)],'subsample':[p/10.0 for p in range(1,11)],'learning_rate':[0.01,0.05,0.1,0.2],'min_samples_split':[p for p in range(1,101)],'max_depth':[p for p in range(1,100)],'min_samples_leaf':[p for p in range(1,50)],'max_features':[p for p in range(1,6)]}
#   	est = ensemble.GradientBoostingRegressor()
#   	gs_cv = GridSearchCV(est,param_grid,cv=10).fit(training["xTrain"],training["yTrain"])
#   	print gs_cv.best_score_
#   	return gs_cv.best_params_


training ={}				
training["id"]=1			#tbe
training["file"]={}			
training["file"]["basePath"]="/home/dev/FoodHubML"	#tbe 
training["file"]["name"] = "trainingVirtual.csv"		#tbe
training["file"]["path"]=training["file"]["basePath"]+"/"+training["file"]["name"]
training["file"]["type"]="csv"
training["weights"]={}
training["weights"]["name"]="weights"
training["weights"]["index"]=-1 #tbe
training["weights"]["values"] = []
training["output"]={}
training["output"]["name"]='Rating'	#tbe
training["output"]["index"]=6	#tbe
training["essentialFeatures"]=[]	#tbe
training["ignoredFeatures"]=[]	#tbe
training["ignoredFeatures"].append({"name":"noOfClicks"})
training["ignoredFeatures"].append({"name":"noOfOrders"})

model={}
model["id"]=3	#tbe
model["algorithm"]=algorithms["GRADIENT_BOOSTING"]
model["file"]={}
model["file"]["basePath"]="/home/dev/FoodHubML/Model"	  #tbe    +str(model["id"])
model["file"]["path"]=model["file"]["basePath"]+"/"+str(model["id"])+".pkl"

prediction={}
prediction["id"]=1			#tbe
prediction["file"]={}			
prediction["file"]["basePath"]="/home/dev/FoodHubML"	#tbe 
prediction["file"]["name"] = "testVirtual.csv"		#tbe
prediction["file"]["path"]=prediction["file"]["basePath"]+"/"+prediction["file"]["name"]
prediction["file"]["type"]="csv"
prediction["output"]={}
prediction["output"]["name"]="Rating"	#tbe
prediction["output"]["index"]=6	#tbe
prediction["essentialFeatures"]=[]	#tbe
prediction["ignoredFeatures"]=[]	#tbe
prediction["ignoredFeatures"].append({"name":"noOfClicks"})
prediction["ignoredFeatures"].append({"name":"noOfOrders"})

prediction["model"] = {}
prediction["model"]["id"] = 1 #tbe
prediction["model"]["basePath"] = "/home/dev/FoodHubML/Model" #tbe
prediction["model"]["path"] = prediction["model"]["basePath"] + "/" + str(prediction["model"]["id"])
training = parseTrainingSet(training)
prediction = parsePredictionSet(prediction)
clf=buildModel(training,model)
yTest = makePrediction(prediction,model)

with open (prediction["file"]["basePath"]+"/"+prediction["output"]["name"]+".csv","ab") as f:
	nLine = len(yTest)
	for i in range(nLine):
		f.write(str(yTest[i])+"\n")