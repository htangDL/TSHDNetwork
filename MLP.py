import scipy.io as sio 
import numpy as np     
from keras.models import Sequential  
from keras.layers import Dense  
import random
from keras import optimizers
import matplotlib.pyplot as plt
from keras import losses
from keras.layers import Activation
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.models import model_from_json
import scipy.io as scio

data = sio.loadmat('inputData_norm4.mat')
datas= data['inputData']
X=datas[:,(0,1,2,3)] # operating variables
Y = datas[:,(4,5)] # performance index

# data = sio.loadmat('data7.mat')
# datas= data['data7']
# X=datas[:,(0,1,2,3,4,5,6)]
# Y = datas[:,(7)]
r1=random.sample(range(0,600000),360000)
r2=random.sample(range(0,600000),120000)
r3=random.sample(range(0,600000),120000)
X_train, Y_train = X[r1], Y[r1]     # training data
X_test, Y_test = X[r2], Y[r2]      #test data  v
X_cv, Y_cv = X[r3], Y[r3]      #cross validation data  v

# MLP model
model = Sequential()  
model.add(Dense(input_dim=4, units=8))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(input_dim=8, units=8))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(input_dim=8, units=8))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(input_dim=8, units=8))
model.add(BatchNormalization())
model.add(Activation('relu'))
  

model.add(Dense(input_dim=8, units=4))
model.add(BatchNormalization())
model.add(Activation('relu'))
  

model.add(Dense(input_dim=4, units=2))


# =============================================================================
# # # load json and create model
# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# model = model_from_json(loaded_model_json)
# #load weights into new model
# model.load_weights("model.h5")
# print("Loaded model from disk")
# #
# =============================================================================


# Adam optimizer
ada=optimizers.Adam(lr=0.003, beta_1=0.9, beta_2=0.999, epsilon=1e-08)  #密度和流速参数
model.compile(loss='mean_squared_error', optimizer=ada)

score = model.evaluate(X, Y, verbose=0)

trnum=10000
a=np.arange(trnum)
c=np.sin(a)
# 训练过程  
print('Training -----------')  
for step in range(trnum):
    cost = model.train_on_batch(X_train, Y_train)
    c[step]= cost
    if step % 10 == 0:  
        print("After %d trainings, the cost: %f" % (step, cost))  
  
# 测试过程  
print('\nTesting ------------')  
cost = model.evaluate(X_test, Y_test, batch_size=1024)
print('test cost:', cost)

cost = model.evaluate(X_train, Y_train, batch_size=1024)
print('train cost:', cost)

cost = model.evaluate(X_cv, Y_cv, batch_size=1024)
print('cv cost:', cost)

W, b = model.layers[0].get_weights()  
print('Weights=', W, '\nbiases=', b)  
  
# 将训练结果绘出  
Y_pretest = model.predict(X_test)
Y_pretrain = model.predict(X_train)
Y_precv=model.predict(X_cv)
scio.savemat('save_var42', {'cost': c, 'Y_pretest': Y_pretest, 'Y_test': Y_test, 'Y_pretrain': Y_pretrain,'Y_train': Y_train,'Y_precv': Y_precv, 'Y_cv': Y_cv})
#plt.plot(a, c)
#plt.show()  
# # serialize model to JSON
model_json = model.to_json()
#
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
 
# later...
 

