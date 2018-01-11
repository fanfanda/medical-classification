import xgboost as xgb
import sys
from sklearn.metrics import roc_auc_score,accuracy_score
import random
import types
from sklearn.model_selection import StratifiedKFold

def get_feature(num):
    ff=open("/Users/steven/Desktop/sample_data_1208_1_1v/fe.txt",'r')
    all_feature=eval(ff.read())
    temp=[0]
    for i in range(len(all_feature)):
        for j in range(num):
            temp.append(int(all_feature[i][j].split(',')[0][1:]))
    temp=list(set(temp))
    return temp

#read data
f = open("/Users/steven/Desktop/sample_data_1208_1_1v/data.txt","r")
data_x = eval(f.read())
f = open("/Users/steven/Desktop/sample_data_1208_1_1v/lable.txt","r")
data_y = eval(f.read())
data_y = list(map(lambda x:x[0],data_y))
##data_all = list(map(lambda x,y:x+y,data_x,data_y))
feature_index=get_feature(30)
filter_data_x=[]
for i in range(len(data_x)):
    temp=[]
    for j in feature_index:
        temp.append(data_x[i][j])
    filter_data_x.append(temp)
data_all=list(map(lambda x,y:x+[y,],data_x,data_y))
user_id=list(map(lambda x:int(x[0]),data_all))
dumpli_user_id=list(set(user_id))
#train

params={'booster':'gbtree',
	    'objective': 'binary:logistic',
	    'eval_metric':'auc',
	    'gamma':10,
##	    'min_child_weight':1.1,
	    'max_depth':5,
##	    'lambda':10,
##            'alpha':1,
	    'subsample':0.8,
	    'colsample_bytree':0.8,
	    'colsample_bylevel':0.8,
	    'eta': 0.01,
	    'tree_method':'exact',
	    'seed':0,
	    'nthread':12
	    }
##watchlist = [(train_data_DMatrix,'train'),(test_data_DMatrix,'val')]

def get_dataset_array_by_id(X,y,k,s_flag):

    data=[]
    if type(X)==list:
        data=X
    else:
        data=X.tolist()

    label=[]
    if type(y)==list:
        label=y
    else:
        label=y.tolist()

    user_set_1=set()
    user_set_0=set()

    for i in range(0,len(data)):
        user_id=int(data[i][0])
        user_flag=int(label[i])
        if user_flag==1:
            user_set_1.add(user_id)
        elif user_flag==0:
            user_set_0.add(user_id)

    user_list_1=list(user_set_1)
    user_list_0=list(user_set_0)

    train_index_array=[]
    test_index_array=[]
    for i in range(0,k):
        slice_user_id_1 = user_list_1[i::k]
        slice_user_id_0 = user_list_0[i::k]

        train_data_index_1=  []
        train_data_index_0 = []
        test_data_index=[]

        for j in range(0,len(data)):
            if (int(data[j][0]) not in slice_user_id_1) and (int(data[j][0]) not in slice_user_id_0):
                if int(label[j])==1:
                    train_data_index_1.append(j)
                elif int (label[j])==0:
                    train_data_index_0.append(j)
            else:
                test_data_index.append(j)

        s_index=[]
        if s_flag==1:
            l_1 = len(train_data_index_1)
            l_0 = len(train_data_index_0)
            if l_1>l_0:
                for j in range(0,l_1-l_0):
                    s_index.append(train_data_index_0[random.randint(0,l_0-1)])
            elif l_0>l_1:
                for j in range(0,l_0-l_1):
                    s_index.append(train_data_index_1[random.randint(0,l_1-1)])
        train_index=[]
        train_index.extend(train_data_index_1)
        train_index.extend(train_data_index_0)
        train_index.extend(s_index)
        train_index_array.append(train_index)
        test_index_array.append(test_data_index)
    return train_index_array,test_index_array


def single_train(num):
    train_data=[]
    test_data=[]

    for i in train_index[num]:
        train_data.append(data_all[i])
    for i in test_index[num]:
        test_data.append(data_all[i])
        
    train_data_x=list(map(lambda x:x[1:len(x)-1],train_data))
    train_data_y=list(map(lambda x:x[len(x)-1],train_data))

    test_data_x=list(map(lambda x:x[1:len(x)-1],test_data))
    test_data_y=list(map(lambda x:x[len(x)-1],test_data))
    
    train_data_DMatrix = xgb.DMatrix(train_data_x,label=train_data_y)
    test_data_DMatrix = xgb.DMatrix(test_data_x,label=test_data_y)
    #5 folds to assure the best iteration
    res = xgb.cv(params,dtrain=train_data_DMatrix,num_boost_round=1400, nfold=5,
                        seed=0,callbacks=[xgb.callback.print_evaluation(show_stdv=False),
                        xgb.callback.early_stop(100)])
    
    
    best_iteration=res.shape[0]

##    watchlist = [(train_data_DMatrix,'train'),(test_data_DMatrix,'val')]
##    model = xgb.train(params,train_data_DMatrix,num_boost_round=3000,evals=watchlist,early_stopping_rounds=100)
    model = xgb.train(params,train_data_DMatrix,num_boost_round=best_iteration)
    #auc and accuracy
    result=model.predict(test_data_DMatrix)

    feature_score = model.get_fscore()
    feature_score = sorted(feature_score.items(), key=lambda x:x[1],reverse=True)
    fs = []
    for (key,value) in feature_score:
        fs.append("{0},{1}".format(key,value))
    feature.append(fs)
    
    print('auc:',roc_auc_score(test_data_y,result))
    print('accuracy:',accuracy_score(list(map(lambda x:1 if x>=0.5 else 0,result)),test_data_y))
    return roc_auc_score(test_data_y,result),accuracy_score(list(map(lambda x:1 if x>=0.5 else 0,result)),test_data_y)


temp_result=[]
train_index,test_index=get_dataset_array_by_id(data_x,data_y,5,0)
##sys.exit()
feature=[]
for i in range(0,1):
    print(str(i)+" th training.......")                                                           
    temp_result.append(single_train(i))
    


auc=list(map(lambda x:x[0],temp_result))
accuracy=list(map(lambda x:x[1],temp_result))

auc=sum(auc)/len(auc)
accuracy=sum(accuracy)/len(accuracy)

print('auc:',auc,'accuracy:',accuracy)


        
