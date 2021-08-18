import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import pickle
import sys

##########################################################
class SyncDataSet(Dataset):
##		Data
## / 0. node_index /
## / 1. k / 2. core / 3. EC / 4. CC / 5. BC / 6. CoI /
## / 7. P /
## / 8. K / 9. alpha /
## / 10. (th_pert - th) / 11. w_pert /
## / 12. label /
## / 13. PR / 14. C /
	def __init__(self, path,fname):
		self.fname=fname
		data=np.loadtxt(path + fname, dtype=np.float32)

		self.len=data.shape[0]
		topology=data[:, 1:8] # Data 1--7
		perturbation=data[:, 10:12] # Data 10, 11
		target=data[:, 12] # Data 12

		input_data=np.append(topology,perturbation,axis=1)
		self.x_data = torch.from_numpy(input_data)

		y_data=[]
		for t in target:
			if t == 1.:
				y_data.append([1.0, 0.0])
			else:
				y_data.append([0.0, 1.0])
		self.y_data = torch.FloatTensor(y_data)

		if torch.cuda.is_available():
			self.x_data=self.x_data.cuda()
			self.y_data=self.y_data.cuda()
	def __getitem__(self,index):
		return self.x_data[index], self.y_data[index]
	def __len__(self):
		return self.len

##########################################################
class Net(nn.Module):
	def __init__(self,input_size,hidden1_size,hidden2_size,hidden3_size,hidden4_size,hidden5_size):
		super(Net, self).__init__()
		self.relu = nn.ReLU()
		self.sf = nn.Softmax(dim=1)
		self.drop = nn.Dropout(p=0.1)

		self.input_size = input_size
		self.hidden1_size = hidden1_size
		self.hidden2_size = hidden2_size
		self.hidden3_size = hidden3_size
		self.hidden4_size = hidden4_size
		self.hidden5_size = hidden5_size

		self.ln1 = nn.Linear(self.input_size,self.hidden1_size,bias=True)
		self.ln2 = nn.Linear(self.hidden1_size,self.hidden2_size,bias=True)
		self.ln3 = nn.Linear(self.hidden2_size,self.hidden3_size,bias=True)
		self.ln4 = nn.Linear(self.hidden3_size,self.hidden4_size,bias=True)
		self.ln5 = nn.Linear(self.hidden4_size,self.hidden5_size,bias=True)

		nn.init.xavier_uniform_(self.ln1.weight)
		nn.init.xavier_uniform_(self.ln2.weight)
		nn.init.xavier_uniform_(self.ln3.weight)
		nn.init.xavier_uniform_(self.ln4.weight)
		nn.init.xavier_uniform_(self.ln5.weight)

#		nn.init.xavier_normal_(self.ln1.weight)
#		nn.init.xavier_normal_(self.ln2.weight)
#		nn.init.xavier_normal_(self.ln3.weight)
#		nn.init.xavier_normal_(self.ln4.weight)
#		nn.init.xavier_normal_(self.ln5.weight)

		self.fc1_bn = nn.BatchNorm1d(self.hidden1_size)
		self.fc2_bn = nn.BatchNorm1d(self.hidden2_size)
		self.fc3_bn = nn.BatchNorm1d(self.hidden3_size)
		self.fc4_bn = nn.BatchNorm1d(self.hidden4_size)
		self.fc5_bn = nn.BatchNorm1d(self.hidden5_size)
		
	def forward(self, x):
		hidden1 = self.relu(self.fc1_bn(self.ln1(x)))
		hidden2 = self.relu(self.fc2_bn(self.ln2(self.drop(hidden1))))
		hidden3 = self.relu(self.fc3_bn(self.ln3(self.drop(hidden2))))
		hidden4 = self.relu(self.fc4_bn(self.ln4(self.drop(hidden3))))
		hidden5 = self.relu(self.fc5_bn(self.ln5(self.drop(hidden4))))
		output  = self.sf(hidden5)
		return output

##########################################################
def Run_NN(device,model,optimizer,criterion,epoch,ftn,data_loader,fname):
	N_dataset=len(data_loader.dataset)
	ave_loss, acc= 0., 0.
	TP, FP, TN, FN = 0., 0., 0., 0.

	if ftn=="Train":
		model.train(True)
	else:
		model.eval()

	for batch, data in enumerate(data_loader):
		inputs, labels = data
		inputs, labels = Variable(inputs), Variable(labels)
		N_MB=labels.size(0)

		inputs.to(device)
		labels.to(device)
		
		optimizer.zero_grad()
		y_pred=model(inputs)
	
		loss = criterion(y_pred.squeeze(), labels)

		if ftn!="Test":
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		ave_loss += loss.item()
		checker=[ int( int(y_pred[i][0] > 0.50) == labels[i][0] )  for i in range(N_MB) ]
		acc+=sum(checker)

		for i in range(N_MB):
			PN=labels[i][0]
			if(checker[i]==1):		# True
				if(PN==1):			#TP
					TP+=1
				else:				#TN
					TN+=1
			else:					# False
				if(PN==1):			#FP
					FP+=1
				else:				#FN
					FN+=1
	ave_loss/=N_dataset
	acc/=N_dataset

	precision=TP/(TP+FP)
	sensitivity=TP/(TP+FN)
	npv=TN/(TN+FN)
	specificity=TN/(TN+FP)
	#
	wfile=open(fname,'a')
	wfile.write( "%d\t%.8le\t%.8le\t%.8le\t%.8le\t%.8le\t%.8le\n"%(epoch,ave_loss,acc,precision,sensitivity,npv,specificity) )
	wfile.close()
##########################################################

#def main():
device = torch.device('cuda:0')

MaxEpoch=int(sys.argv[1])	# 150
what=str(sys.argv[2])		# Homo or Hetero
cnt=int(sys.argv[3])		# 1~10

#random.seed(cnt)
#torch.manual_seed(cnt)
#if device =='cuda:0':
#	torch.cuda.manual_seed_all(cnt)

BS=64
hyper_param=1e-4	# 1e-6
params={'batch_size': BS, 'shuffle' : True, 'drop_last' : True}


# Model
model=Net(9,10,20,20,10,2)
model=model.to(device)
# Binary Cross Entorpy Loss
criterion=nn.BCELoss(reduction='mean')
# Adam optimizer
optimizer = optim.Adam(model.parameters(), lr =hyper_param)

# Data
data_path='./Synthetic_Data/Data%d/%s/' % (cnt,what)
# Training
train_dataset = SyncDataSet(data_path,"Train.txt")
train_loader = DataLoader(dataset=train_dataset, **params)
wfname_train="./NN_Training_%s%d.txt" % (what,cnt)
# Test
test_dataset = SyncDataSet(data_path,"Test.txt")
test_loader = DataLoader(dataset=test_dataset, **params)
wfname_test="./NN_Synthetic_%s%d.txt" % (what,cnt)

# Running
for epoch in range(MaxEpoch+1):
# Training
	Run_NN(device,model,optimizer,criterion,epoch,"Train",train_loader,wfname_train)

model_path='./NN_Model_%s%d.pt' %(what,cnt)
# Save model
torch.save(model,model_path)
# Load model
#model=torch.load(model_path)
# Test
with torch.no_grad():
	Run_NN(device,model,optimizer,criterion,MaxEpoch,"Test",test_loader,wfname_test)

	
##########################################################
	
#if __name__ == "__mani__":
#	main()
