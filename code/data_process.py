#!/usr/bin/env python
# coding: utf-8

# In[6]:

import tensorflow as tf
import copy
import pandas as pd
from joblib import Parallel, delayed
import multiprocessing, time

data = pd.read_csv('I:/Code/IPRec/IPRec-main-pre/IPRec-main/code/pre_produce/shapy_parallel_data_2.csv',
                   keep_default_na=False, header=None, error_bad_lines=False)

f_data = pd.DataFrame()
tra_data = pd.DataFrame()
val_data = pd.DataFrame()
tes_data = pd.DataFrame()
u_list, s_u_list, pos_list, neg_list = [],[],[],[]
k=0
print("begin!")

def fun_avg(user,d):

    neg_ratio = 5          #负比例
    u_list.append(user)      #用户数组添加用户
    pos = d[d[3]==1]  #   pos=d[1],d[3]=1||pos=d[0],d[3]!=1
    neg = d[d[3]==0]  #   neg=d[1],d[3]=0|| neg=d[0],d[3]!=0
    l1 = pos.shape[0]
    l2 = neg.shape[0]
    assert l1+l2 == d.shape[0]
    pos=pos.reset_index(drop=True)
    neg=neg.reset_index(drop=True)
    if l1>=1 and not neg.empty:

        s_u_list.append(user)
        pos_list.append(l1)
        neg_list.append(l2)
        if l2 >= neg_ratio*l1:
            s_neg = neg[:neg_ratio*l1]
        else:
            s_neg = neg

        s_data = pd.concat([pos,s_neg],ignore_index=True).sample(frac=1).reset_index(drop=True)
        tra_data = s_data[:int(s_data.shape[0]*0.6)]
        val_data = s_data[int(s_data.shape[0]*0.7):int(s_data.shape[0]*0.8)]
        tes_data = s_data[int(s_data.shape[0]*0.8):]


        return s_data, tra_data, val_data, tes_data


def applyParallel(dfGrouped, func):

    res = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(func)(name,group) for name, group in dfGrouped)

    res = list(filter(None, res))

    d = [item[0] for item in res]   #s_data
    t = [item[1] for item in res]    #tra_data
    v = [item[2] for item in res]    #val_data
    s = [item[3] for item in res]    #test_data
    return pd.concat(d, ignore_index=True),pd.concat(t, ignore_index=True),pd.concat(v, ignore_index=True),pd.concat(s, ignore_index=True)
t1=time.time()
f_data, tra_data, val_data, tes_data = applyParallel(data.groupby(0),fun_avg)
print(time.time()-t1)
f_data
print(f_data.head())



f_max_len=20
user_set = set(tra_data[0])
item_set = set(tra_data[2])
biz_set = set(tra_data[4])
user_set.update([item for sublist in tra_data[6] for item in list(map(lambda x: int(x),sublist.split(',')[:f_max_len]))])

user_map = dict(zip(user_set,range(1,len(user_set)+1)))
item_map = dict(zip(item_set,range(1,len(item_set)+1)))
biz_map = dict(zip(biz_set,range(1,len(biz_set)+1)))

tes_data = tes_data[(tes_data[2].isin(item_set))&(tes_data[4].isin(biz_set))]

tes_data[6] = tes_data[6].map(lambda x: ','.join([y for y in x.split(',')[:f_max_len] if int(y) in user_set]))
#
tes_data = tes_data[tes_data[6]!='']
val_data = val_data[(val_data[2].isin(item_set))&(val_data[4].isin(biz_set))]
val_data[6] = val_data[6].map(lambda x: ','.join([y for y in x.split(',')[:f_max_len] if int(y) in user_set]))
val_data = val_data[val_data[6]!='']




tra_data[0] = tra_data[0].map(lambda x: user_map[x])#用户列表
tra_data[2] = tra_data[2].map(lambda x: item_map[x])#文章列表
tra_data[4] = tra_data[4].map(lambda x: biz_map[x])#媒体列表
tra_data[6] = tra_data[6].map(lambda x: [user_map[int(y)] for y in x.split(',')[:f_max_len]])#朋友列表

tra_data=tra_data.drop([1], axis=1)
#tra_data.drop([1],axis=1)删除列[1]
val_data[0] = val_data[0].map(lambda x: user_map[x])
val_data[2] = val_data[2].map(lambda x: item_map[x])
val_data[4] = val_data[4].map(lambda x: biz_map[x])
val_data[6] = val_data[6].map(lambda x: [user_map[int(y)] for y in x.split(',')[:f_max_len]])
val_data=val_data.drop([1], axis=1)
tes_data[0] = tes_data[0].map(lambda x: user_map[x])
tes_data[2] = tes_data[2].map(lambda x: item_map[x])
tes_data[4] = tes_data[4].map(lambda x: biz_map[x])
tes_data[6] = tes_data[6].map(lambda x: [user_map[int(y)] for y in x.split(',')[:f_max_len]])
tes_data=tes_data.drop([1], axis=1)
f_data = pd.concat([tra_data, val_data, tes_data],ignore_index=True)
tra_data


# In[11]:

from collections import defaultdict
user_packages=dict()
user_items=dict()
user_bizs=dict()
user_friends=dict()
pack_neighbors_b=defaultdict(list)
pack_neighbors_f=defaultdict(list)


t1=time.time()
for u,d in f_data.groupby([0]):
# def fun_avg1(u,d):
#     global pack_neighbors_f
    i = list(d[2]) #item
    b = list(d[4]) #biz
    f = list(map(lambda x: x+[0]*(f_max_len-len(x)),d[6])) #friend
    packages = list(zip(i,b,f))
    for index in range(len(packages)):
        for j in range(index+1,len(packages)):
            a = set(packages[index][2]) & set(packages[j][2])-{0}
            if a and packages[index][0] != packages[j][0]:
                pack_neighbors_f[str(u)+'_'+str(packages[index][0])].append((len(a),packages[j]))
                pack_neighbors_f[str(u)+'_'+str(packages[j][0])].append((len(a),packages[index]))
            #邻居包判断
    for item in set(i):
        try:
            pack_neighbors_f[str(u)+'_'+str(item)].sort(key=lambda x:x[0], reverse = True)
            pack_neighbors_f[str(u)+'_'+str(item)] = list(map(lambda x: x[1],pack_neighbors_f[str(u)+'_'+str(item)]))
        except:
            print(pack_neighbors_f[str(u)+'_'+str(packages[index][0])])
print(time.time()-t1)
                
t1=time.time()
for name,d in f_data.groupby([0,4]):
# def fun_avg2(name,d):
#     global pack_neighbors_b
    if d.shape[0]>1:
        i = list(d[2])
        b = list(d[4])
        f = list(map(lambda x: x+[0]*(f_max_len-len(x)),d[6]))
        packages = list(zip(i,b,f))
        for index in range(len(packages)):
            tmp = packages.copy()
            del tmp[index]
#             print(name[0],packages[index][0])
            pack_neighbors_b[str(name[0])+'_'+str(packages[index][0])] = tmp
print(time.time()-t1)

t1=time.time()
for user,d in tra_data.groupby(0):

    pos = d[(d[3]==1)]
    b_pos = d[(d[5] == 1)]
    i = list(pos[2])
    b = list(pos[4])
    f = list(map(lambda x: x+[0]*(f_max_len-len(x)),pos[6]))
    f_ = list(pos[6])
    b_ = list(b_pos[4])
    user_packages[user]=list(zip(i,b,f))
    user_items[user]=i
    user_bizs[user]=b_
    user_friends[user]=[item for sublist in f_ for item in sublist]
print(time.time()-t1)
pack=copy.deepcopy(user_packages)
print("package:",list(map(lambda i:i, pack.values())))


tra_data=tra_data.sample(frac=1).reset_index(drop=True)
tra_data




def to_list(l):
    res = []
    for i in l:
        res.append(i[0])
        res.append(i[1])
        res.extend(i[2])
    return res

def to_tfrecords1(features,file):
    writer=tf.python_io.TFRecordWriter(path=file)
    size = features.shape[0]
    print(features.head(5))
    for i in range(size):
        pack = str(features.iloc[i][0])+'_'+str(features.iloc[i][2])
        u_i = copy.deepcopy(user_items[features.iloc[i][0]])
        u_b = copy.deepcopy(user_bizs[features.iloc[i][0]])
        u_f = copy.deepcopy(user_friends[features.iloc[i][0]])
        print(len(features.iloc[i][6]))

        u_p = copy.deepcopy(user_packages[features.iloc[i][0]])
        if features.iloc[i][3] == 1:
            u_i.remove(features.iloc[i][2])
            for x in features.iloc[i][6]:
                u_f.remove(x)
            for j,x in enumerate(u_p):
                if x[0] == features.iloc[i][2]:
                    del u_p[j]
                    break
        example=tf.train.Example(
            features=tf.train.Features(
                feature={
                    "user":tf.train.Feature(int64_list=tf.train.Int64List(value=[features.iloc[i][0]])),
                    "item":tf.train.Feature(int64_list=tf.train.Int64List(value=[features.iloc[i][2]])),
                    "biz":tf.train.Feature(int64_list=tf.train.Int64List(value=[features.iloc[i][4]])),
                    "friends":tf.train.Feature(int64_list=tf.train.Int64List(value=features.iloc[i][6])),
                    "user_items":tf.train.Feature(int64_list=tf.train.Int64List(value=u_i)),
                    "user_bizs":tf.train.Feature(int64_list=tf.train.Int64List(value=u_b)),
                    "user_friends":tf.train.Feature(int64_list=tf.train.Int64List(value=u_f)),
                    "user_packages":tf.train.Feature(int64_list=tf.train.Int64List(value=to_list(u_p[:50]))),
                    "pack_neighbors_b":tf.train.Feature(int64_list=tf.train.Int64List(value=to_list(pack_neighbors_b[pack][:20]))),
                    "pack_neighbors_f":tf.train.Feature(int64_list=tf.train.Int64List(value=to_list(pack_neighbors_f[pack][:20]))),
                    "f_max_len":tf.train.Feature(int64_list=tf.train.Int64List(value=[f_max_len])),
                    "n_users":tf.train.Feature(int64_list=tf.train.Int64List(value=[len(user_map)])),
                    "n_items":tf.train.Feature(int64_list=tf.train.Int64List(value=[len(item_map)])),
                    "n_bizs":tf.train.Feature(int64_list=tf.train.Int64List(value=[len(biz_map)])),
                    "label":tf.train.Feature(int64_list=tf.train.Int64List(value=[features.iloc[i][3]])),
                    "label2":tf.train.Feature(int64_list=tf.train.Int64List(value=[features.iloc[i][5]]))
                }
            )
        )
        writer.write(record=example.SerializeToString())

    writer.close()


def to_tfrecords(features,file):
    writer=tf.python_io.TFRecordWriter(path=file)
    size = features.shape[0]
    for i in range(size):
        pack = str(features.iloc[i][0])+'_'+str(features.iloc[i][2])
#         print(features.iloc[i][5])
        example=tf.train.Example(
            features=tf.train.Features(
                feature={
                    "user":tf.train.Feature(int64_list=tf.train.Int64List(value=[features.iloc[i][0]])),
                    "item":tf.train.Feature(int64_list=tf.train.Int64List(value=[features.iloc[i][2]])),
                    "biz":tf.train.Feature(int64_list=tf.train.Int64List(value=[features.iloc[i][4]])),
                    "friends":tf.train.Feature(int64_list=tf.train.Int64List(value=features.iloc[i][6])),
                    "user_items":tf.train.Feature(int64_list=tf.train.Int64List(value=user_items[features.iloc[i][0]][:50])),
                    "user_bizs":tf.train.Feature(int64_list=tf.train.Int64List(value=user_bizs[features.iloc[i][0]][:50])),
                    "user_friends":tf.train.Feature(int64_list=tf.train.Int64List(value=user_friends[features.iloc[i][0]][:50*f_max_len])),
                    "user_packages":tf.train.Feature(int64_list=tf.train.Int64List(value=to_list(user_packages[features.iloc[i][0]][:50]))),
                    "pack_neighbors_b":tf.train.Feature(int64_list=tf.train.Int64List(value=to_list(pack_neighbors_b[pack][:20]))),
                    "pack_neighbors_f":tf.train.Feature(int64_list=tf.train.Int64List(value=to_list(pack_neighbors_f[pack][:20]))),
                    "f_max_len":tf.train.Feature(int64_list=tf.train.Int64List(value=[f_max_len])),
                    "n_users":tf.train.Feature(int64_list=tf.train.Int64List(value=[len(user_map)])),
                    "n_items":tf.train.Feature(int64_list=tf.train.Int64List(value=[len(item_map)])),
                    "n_bizs":tf.train.Feature(int64_list=tf.train.Int64List(value=[len(biz_map)])),
                    "label":tf.train.Feature(int64_list=tf.train.Int64List(value=[features.iloc[i][3]])),
                    "label2":tf.train.Feature(int64_list=tf.train.Int64List(value=[features.iloc[i][5]]))

                }

            )
        )

        writer.write(record=example.SerializeToString())

    writer.close()

to_tfrecords1(tra_data, '../train_shapi_2.tfrecords')
to_tfrecords(val_data, '../validation_shapi_2.tfrecords')
to_tfrecords(tes_data, '../test_shapi_2.tfrecords')

print("finish")