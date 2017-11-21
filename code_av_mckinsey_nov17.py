import pandas as pd
import numpy as np
import xgboost as xgb
import datetime as dt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor

train=pd.read_csv("/home/sohom/Desktop/AV_MC-Kinsey_Nov17/train.csv",low_memory=False,header=0)
test=pd.read_csv("/home/sohom/Desktop/AV_MC-Kinsey_Nov17/test.csv",low_memory=False,header=0)

train['Junction'].value_counts()
# 3    14592
# 2    14592
# 1    14592
# 4     4344

test['Junction'].value_counts()
# 4    2952
# 3    2952
# 2    2952
# 1    2952

train_test=train.append(test)

#2017-10-31 23:00:00
train_test['DateTime'] =  pd.to_datetime(train_test['DateTime'], format='%Y-%m-%d %H:%M:%S',errors='coerce')

train_test['weekday'] = train_test['DateTime'].dt.weekday
train_test['day_of_month'] = train_test['DateTime'].dt.day
train_test['month'] = train_test['DateTime'].dt.month
train_test['hour'] = train_test['DateTime'].dt.hour
train_test['year'] = train_test['DateTime'].dt.year
train_test['year']=train_test['year'].replace(to_replace={2015:1,2016:2,2017:3})
train_test['minutes'] = train_test['DateTime'].dt.minute
train_test['seconds'] = train_test['DateTime'].dt.second

features = ['Junction', 'weekday', 'day_of_month','month', 'hour', 'year', 'minutes', 'seconds']

train.groupby('Junction').agg({'Vehicles' : np.mean})


X_train_all = train_test[0:len(train.index)]

X_train = X_train_all.sample(frac=0.80, replace=False)
X_valid = pd.concat([X_train_all, X_train]).drop_duplicates(keep=False)
X_test = train_test[len(train.index):len(train_test.index)]
'''

#Better Approach
np.random.seed(40)
msk = np.random.rand(len(X_train_all)) < 0.8
trn = X_train[msk]
val = X_train[~msk]

'''


dtrain = xgb.DMatrix(X_train[features], X_train['Vehicles'], missing=np.nan)
dvalid = xgb.DMatrix(X_valid[features], X_valid['Vehicles'], missing=np.nan)
dtest = xgb.DMatrix(X_test[features], missing=np.nan)

nrounds = 800
watchlist = [(dtrain, 'train')]

params = {"objective": "reg:linear","booster": "gbtree", "nthread": 16, "silent": 1,
                "eta": 0.01, "max_depth": 5, "subsample": 0.9, "colsample_bytree": 0.7,
                "min_child_weight": 1,
                "seed": 2016, "tree_method": "exact"}

bst = xgb.train(params, dtrain, num_boost_round=nrounds, evals=watchlist, verbose_eval=20)

valid_preds = bst.predict(dvalid)
mean_squared_error(X_valid['Vehicles'], [int(i) for i in valid_preds]) 
test_preds = bst.predict(dtest)
submit=pd.DataFrame({'ID':test['ID'],'Vehicles':[int(i) for i in test_preds]})
submit[['ID','Vehicles']].to_csv('xgb3.csv',index=False)


'''
#catboost: Junction, year,month... as categorical
cat_cols = ['Junction', 'weekday', 'day_of_month','month', 'hour', 'year', 'minutes', 'seconds']

for col in cat_cols:
	X_train[col+"cat"] = pd.to_numeric(pd.Series(X_train[col]),errors='coerce')

cat_cols_new=['Junctioncat', 'weekdaycat', 'day_of_monthcat', 'monthcat', 'hourcat', 'yearcat', 'minutescat', 'secondscat']

model = CatBoostRegressor(depth=10, iterations=10, learning_rate=0.1, l2_leaf_reg=3, rsm=1, random_seed=1,loss_function='RMSE',use_best_model=True)
model.fit(X_train[features],[int(i) for i in X_train['Vehicles']],cat_features=cat_cols,eval_set = (X_valid[features], [int(i) for i in X_valid['Vehicles']]),use_best_model = True)
valid_preds = model.predict(X_valid[features])
valid_preds =[int(i) for i in valid_preds]
pred = model.predict(test[features])
pred_ans = [int(i) for i in list(pred[:,0])]
submit = pd.DataFrame({'connection_id': test['connection_id'],'target':pred_ans})
submit.to_csv('catboost1.csv',index=False)

mean_squared_error(X_valid['Vehicles'], valid_preds) 
'''
