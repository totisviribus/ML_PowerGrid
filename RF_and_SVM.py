import sys
import numpy as np
import random as rd
from sklearn import svm
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os.path


what=str(sys.argv[1])	# Homo or Hetero
cnt=int(sys.argv[2])	# 1~10
alg=str(sys.argv[3])	# RF or SVM
if alg == "RF" or alg == "SVM":
	print(alg)
else:
	print("3rd argument -- RF or SVM")
	quit()
#rd.seed(cnt)


##	Data
## / 0. Node index
## / 1. k / 2. k-core / 3. Eigenvalue Centrality / 4. Clustering Coefficient / 5. Betweenness Centrality / 6. CoI / 
## / 7. P / 8. K / 9. alpha / 10. (th_pert - th) / 11. w_pert /
## / 12. sync /
## / 13. PR / 14. C /
##

# Data Load
data_path='./Synthetic_Data/Data%d/%s/' % (cnt,what)
Train = data_path + 'Train.txt'
Val = data_path + 'Validation.txt'
Test  = data_path + 'Test.txt'

# Training Data
Train_data=np.loadtxt(Train)
rd.shuffle(Train_data)
X1 = Train_data[:, 1:8]
X2 = Train_data[:, 10:12]
X_train = np.append(X1,X2,axis=1)
y_train = Train_data[:, 12]

Val_data=np.loadtxt(Val)
rd.shuffle(Val_data)
X1 = Val_data[:, 1:8]
X2 = Val_data[:, 10:12]
X_val = np.append(X1,X2,axis=1)
y_val = Val_data[:,12]

X_train=np.concatenate((X_train,X_val),axis=0)
y_train=np.concatenate((y_train,y_val),axis=None)

# Test Data
Test_data=np.loadtxt(Test)
X1 = Test_data[:, 1:8]
X2 = Test_data[:, 10:12]
X_test = np.append(X1,X2,axis=1)
y_test = Test_data[:, 12]

#
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# Training
if alg=="SVM":
	model = svm.SVC(gamma='scale')
	model.fit(X_train,y_train)

	# Save Model
	svmf='./SVM_Model_%s%d.sav' %(what,cnt)
	joblib.dump(model, svmf)
	# Load Model
	#model = joblib.load(svmf)
elif alg=="RF":
	model = RandomForestClassifier(n_estimators=500, bootstrap= True, max_features = 'sqrt')
	model.fit(X_train,y_train)

	# Save Model
	rff='RF_Model_%s%d.sav' %(what,cnt)
	joblib.dump(model, rff)
	# Load Model
	#model = joblib.load(rff)
else:
	quit()

# Test
y_pred= model.predict(X_test)

# Calc.
Ndata=len(y_test)
TP,TN,FP,FN = 0.,0.,0.,0.
for i in range(Ndata):
	if y_pred[i] == 1: # Positive
		if y_test[i]==1:	# True Positive
			TP+=1.
		else:		# False Positive
			FP+=1.
	else:		  # Netative
		if y_test[i]==1: # False Negative
			FN+=1.
		else:		# True Negative
			TN+=1.
acc=(TP+TN)/(TP+TN+FP+FN)
precision=TP/(TP+FP)
sensitivity=TP/(TP+FN)
npv=TN/(TN+FN)
specificity=TN/(TN+FP)
	 
wfname_rslt="./%s_Synthetic_%s%d.txt" % (alg,what,cnt)
wfile_rslt=open(wfname_rslt,'w')
wfile_rslt.write( "%.8le\t%.8le\t%.8le\t%.8le\t%.8le\n"%(acc,precision,sensitivity,npv,specificity) )
wfile_rslt.close()
