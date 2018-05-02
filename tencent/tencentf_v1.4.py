import pandas as pd
import lightgbm as lgb
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
import os
import gc
import math
import numpy as np
import MeanEncoder
# =============================================================================
# from sklearn import preprocessing
# =============================================================================


names = ['LBS', 'age', 'appIdAction', 'appIdInstall', 'carrier','consumptionAbility', 'ct', 'education', 'gender', 'house', 'interest1','interest2', 'interest3', 'interest4', 'interest5', 'kw1', 'kw2', 'kw3','marriageStatus', 'os', 'topic1', 'topic2', 'topic3', 'uid']

# =============================================================================
# def get_user_feature():
#     if os.path.exists('C:/Users/1707500/Desktop/preliminary_contest_data/userFeature.csv'):
#         user_feature=pd.read_csv('C:/Users/1707500/Desktop/preliminary_contest_data/userFeature.csv')
#     else:
#         userFeature_data = []
#         with open('C:/Users/1707500/Desktop/preliminary_contest_data/userFeature.data', 'r') as f:
#             for i, line in enumerate(f):
#                 line = line.strip().split('|')
#                 userFeature_dict = {}
#                 for each in line:
#                     each_list = each.split(' ')
#                     userFeature_dict[each_list[0]] = ' '.join(each_list[1:])
#                 userFeature_data.append(userFeature_dict)
#                 if i % 100000 == 0:
#                     print(i)
#             user_feature = pd.DataFrame(userFeature_data)
#             user_feature.to_csv('C:/Users/1707500/Desktop/preliminary_contest_data/userFeature.csv', index=False)
#         gc.collect()
#     return user_feature
# =============================================================================

def get_data():
    if os.path.exists('C:/Users/1707500/Desktop/preliminary_contest_data/data.csv'):
        return pd.read_csv('C:/Users/1707500/Desktop/preliminary_contest_data/data.csv')
    else:
        ad_feature = pd.read_csv('C:/Users/1707500/Desktop/preliminary_contest_data/adFeature.csv', sep = ',')
        train=pd.read_csv('C:/Users/1707500/Desktop/preliminary_contest_data/train.csv', sep = ',')
        predict=pd.read_csv('C:/Users/1707500/Desktop/preliminary_contest_data/test1.csv', sep = ',')
        train.loc[train['label']==-1,'label']=0
        predict['label']=-1
        user_feature = pd.read_csv('C:/Users/1707500/Desktop/preliminary_contest_data/userFeature.csv', sep = ',',names = names)
        data=pd.concat([train,predict])
        data=pd.merge(data,ad_feature,on='aid',how='left')
        data=pd.merge(data,user_feature,on='uid',how='left')
        data=data.fillna('-1')
        del user_feature
        return data

def batch_predict(data,index):
    one_hot_feature=['age','carrier','consumptionAbility','education','gender','house','os','marriageStatus','productType']
    total = ['age','carrier','consumptionAbility','education','gender','house','os','ct','marriageStatus','productType','LBS','advertiserId','campaignId', 'creativeId','adCategoryId', 'productId']
    vector_feature=['appIdAction','appIdInstall','interest1','interest2','interest3','interest4','interest5','kw1','kw2','kw3','topic1','topic2','topic3']
    for feature in total:
        try:
            data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))
        except:
            data[feature] = LabelEncoder().fit_transform(data[feature])
    print('1')
    train=data[data.label!=-1]
    train_y=train.pop('label')
    test=data[data.label==-1]
    res=test[['aid','uid']]
    test=test.drop('label',axis=1)
    enc = OneHotEncoder()
    train_x=train[['creativeSize','LBS','advertiserId','campaignId', 'creativeId','adCategoryId', 'productId']]
    test_x=test[['creativeSize','LBS','advertiserId','campaignId', 'creativeId','adCategoryId', 'productId']]
    
# =============================================================================
#     min_max_scaler = preprocessing.MinMaxScaler()
#     features_new = min_max_scaler.fit_transform(train_x)       
#     train_x = pd.DataFrame(features_new, columns=train_x.columns)
#     
#     features_new_1 = min_max_scaler.fit_transform(test_x)       
#     test_x = pd.DataFrame(features_new_1, columns=test_x.columns)
# =============================================================================
    
    print('2')
    for feature in one_hot_feature:
        enc.fit(data[feature].values.reshape(-1, 1))
        train_a=enc.transform(train[feature].values.reshape(-1, 1))
        test_a = enc.transform(test[feature].values.reshape(-1, 1))
        train_x= sparse.hstack((train_x, train_a))
        test_x = sparse.hstack((test_x, test_a))
        print(feature+' finish')
    print('one-hot prepared !')

    cv=CountVectorizer()
    for feature in vector_feature:
        cv.fit(data[feature])
        train_a = cv.transform(train[feature])
        test_a = cv.transform(test[feature])
        train_x = sparse.hstack((train_x, train_a))
        test_x = sparse.hstack((test_x, test_a))
        print(feature + ' finish')
    print('cv prepared !')
    del data
    return LGB_predict(train_x, train_y, test_x, res,index)

def LGB_predict(train_x,train_y,test_x,res,index):
    print("LGB test")
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=40, reg_alpha=0.2, reg_lambda=2,
        max_depth=15, n_estimators=1000, objective='binary',
        subsample=0.7, colsample_bytree=0.7, subsample_freq=2,
        learning_rate=0.1, min_child_weight=30, random_state=2018, n_jobs=-1
    )
    clf.fit(train_x, train_y, eval_set=[(train_x, train_y)], eval_metric='auc',early_stopping_rounds=100)
    res['score'+str(index)] = clf.predict_proba(test_x)[:,1]
    res['score'+str(index)] = res['score'+str(index)].apply(lambda x: float('%.6f' % x))
    print(str(index)+' predict finish!')
    gc.collect()
    res=res.reset_index(drop=True)
    return res['score'+str(index)]

#数据分片处理，对每片分别训练预测，然后求平均
data=get_data()
train_old=data[data['label']!=-1]
test_old=data[data['label']==-1]
MeanEncoder_features = ['LBS','advertiserId','campaignId', 'creativeId','adCategoryId', 'productId','ct']
MeanEncoder = MeanEncoder.MeanEncoder(MeanEncoder_features, n_splits=5, target_type='classification', prior_weight_func=None)
train = MeanEncoder.fit_transform(train_old,train_old['label'])
test = MeanEncoder.fit_transform(test_old,test_old['label'])

train = train.drop(['LBS','advertiserId','campaignId', 'creativeId','adCategoryId', 'productId','ct','LBS_pred_0','advertiserId_pred_0','campaignId_pred_0', 'creativeId_pred_0','adCategoryId_pred_0', 'productId_pred_0','ct_pred_0'],axis=1)
test = test.drop(['LBS','advertiserId','campaignId', 'creativeId','adCategoryId', 'productId','ct','LBS_pred_0','advertiserId_pred_0','campaignId_pred_0', 'creativeId_pred_0','adCategoryId_pred_0', 'productId_pred_0','ct_pred_0'],axis=1)



del data
predict=pd.read_csv('C:/Users/1707500/Desktop/preliminary_contest_data/test1.csv')
cnt=20
size = math.ceil(len(train) / cnt)
result=[]
for i in range(cnt):
    start = size * i
    end = (i + 1) * size if (i + 1) * size < len(train) else len(train)
    slice = train[start:end]
    result.append(batch_predict(pd.concat([slice,test]),i))
    gc.collect()

result=pd.concat(result,axis=1)
result['score']=np.mean(result,axis=1)
result=result.reset_index(drop=True)
result=pd.concat([predict[['aid','uid']].reset_index(drop=True),result['score']],axis=1)
result[['aid','uid','score']].to_csv('C:/Users/1707500/Desktop/preliminary_contest_data/submission.csv', index=False)