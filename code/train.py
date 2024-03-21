# -*- coding: utf-8 -*-
# @Create Time : 2020/7/13 15:11
# @Author : lee
# @FileName : train.py

import argparse
import os, time
import random
import pandas as pd
import tensorflow as tf
from LoadData import parse_function_
from Logging import Logging
from Pack import Model
from sklearn.metrics import roc_auc_score, f1_score, recall_score, accuracy_score, precision_recall_fscore_support,precision_recall_curve,hamming_loss
import numpy as np
from multiprocessing import Process
from multiprocessing import Pool
import os
import sys
import time

#启用动态图机制
#tf.enable_eager_execution()




def eval_epoch(args, sess, test_score, test_loss, test_data,a1_,a2_,a3_,test=False):
    # a = 1
    # global a
    loss = []
    score = []
    pred = []
    label = []
    a1,a2,a3,ta = [],[],[],[]
    user = []
    while True:
        try:
            score_, loss_, label_,label2_,a1__,a2__,a3__,u= sess.run([test_score, test_loss, test_data['label'],test_data['label2'],a1_,a2_,a3_,test_data['user']])
            loss.append(loss_)
            score.extend(score_)
            label.extend(label_)
            a1.extend(a1__)
            a2.extend(a2__)
            a3.extend(a3__)
            user.extend(u)
        except tf.errors.OutOfRangeError:
            break

    auc = roc_auc_score(label, score)
    pred = list(map(lambda x: 1 if x>=0.5 else 0, score))
    acc = accuracy_score(label, pred)

    error =hamming_loss(label,pred)
    prec, rec, f1, _ = precision_recall_fscore_support(label, pred, average="binary")

    return np.mean(np.array(loss)), auc, f1, acc, prec, rec,error




def train_process(theta,d):
    acc_param = []
    auc_param = []
    train_auc = []
    epoch_param = []
    f1_param = []
    recall_param = []
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../Data/', help='Input data path.')
    parser.add_argument('--dataset', type=str, default='3day', help='Dataset.')
    parser.add_argument('--epoch', type=int, default=300, help='Number of epoch.')#20
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size.')#256
    parser.add_argument('--K', type=int, default=4, help='Disentangle components.')#解耦成k个embeding
    parser.add_argument('--reg', type=float, default=1e-6, help='Regularization.') #归一化 1e-2
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate.')  #学习率0.00001
    parser.add_argument('--drop_out', type=float, default=0.4, help='dropout rate.')#特征丢掉率，按照一定概率丢掉一部分神经网络单元，训练时每个神经单元以概率p去除，测试时每个神经单元存在，权重乘以p

    parser.add_argument('--dimension', type=int, default=64, help='Dimension of embedding size.') #嵌入纬度
    parser.add_argument('--gpu_id', type=int, default=0, help='Gpu id.') #GPU序号
    parser.add_argument('--buffer_size', type=int, default=2000, help='Buffer size.') #100000
    parser.add_argument('--ratio', type=float, default=1, help='train ratio.') #训练比例
    args = parser.parse_args()
    # 隐私预算
    # theta = 10
    
    # train_f1
    if args.dataset == '3day':
        print('3day')
        f_max_len = 20  # 'friends': [f_max_len]， 朋友最多数量
        u_max_pack = 50  # 'user_packages': [u_max_pack, 2+f_max_len]， 用户-包
        pack_max_nei_b = 20 # pack_neighbors_b': [pack_max_nei_b, 2+f_max_len], 包-邻居-media
        pack_max_nei_f = 20 #'pack_neighbors_f': [pack_max_nei_f, 2+f_max_len], 包-邻居-朋友
        n_users = 34388#554237   #用户数据条数#554399
        n_items = 2147 #344230#344087    Article 数量#344363
        n_bizs =  100  #166465       #media 数量#166311
        u_max_i = 400# 71
        u_max_f = 300 #220

        n_train =18883 #1990000   #训练集数量
        # test_filenames = '../data/test_f1.tfrecords'
        # train_filenames = '../data/train_f1.tfrecords'
        # valid_filenames = '../data/validation_f1.tfrecords'
        test_filenames = '../code/test_shapi_2.tfrecords'
        train_filenames = '../code/train_shapi_2.tfrecords'
        valid_filenames = '../code/validation_shapi_2.tfrecords'
        # test_filenames = '../code/test_shapi.tfrecords'
        # train_filenames = '../code/train_shapi.tfrecords'
        # valid_filenames = '../code/validation_shapi.tfrecords'

        # test_filenames = '../code/test_f1_min.tfrecords'
        # train_filenames = '../code/train_f1_min.tfrecords'
        # valid_filenames = '../code/validation_f1_min.tfrecords'



#构建字典结构，定义batch形状
    padded_shape = {'user': [],   #用户
                'item': [],     #项
                'biz': [],   #媒体
                'friends': [f_max_len],  #朋友[f1,...,f20]
                'user_items': [u_max_i],   #[a1,a2,...,a99]
                'user_bizs': [u_max_i],    #[m1,m2,...,m99]
                'user_friends': [u_max_f],  #
                'user_packages': [u_max_pack, 2+f_max_len],
                'pack_neighbors_b': [pack_max_nei_b, 2+f_max_len],
                'pack_neighbors_f': [pack_max_nei_f, 2+f_max_len],
                'label': [],'label2': []}

    # --------------------------read data from files---------------------------------读取数据，从文件中读取数据
    train_dataset = tf.data.TFRecordDataset(train_filenames)
    test_dataset = tf.data.TFRecordDataset(test_filenames)
    valid_dataset = tf.data.TFRecordDataset(valid_filenames)
    # -----------shuffle and padding of datasets-----------------------数据集填充，shuffle打乱数据，buffersize越大，打乱程度越大，buffer_size=1不打乱顺序，保持原序，
    # *********map是把DateSet里面每个元素都map一遍，返回值作为新的dataset，主要是对每个元素进行操作
    #------------------------------------------flat_map,既执行了map还进行降维，filter，过滤
    train_dataset = train_dataset.map(parse_function_(f_max_len)).shuffle(buffer_size=args.buffer_size)#自定义parse_function_：将数据转换成numpy类型
    #**************映射特征键至张量值Tesor构建dataset 并打乱数据顺序
    test_dataset = test_dataset.map(parse_function_(f_max_len))
    valid_dataset = valid_dataset.map(parse_function_(f_max_len))

    #批量读取，shapes=[-1]或者不设置，按照每个batch中最大的size进行padding，填充形状为padded_shape
    #*****************按照254位大小，批量读取数据，padded_shapes指定各个成员要pad成的形状，drop_remainder表示将最后未处理的数据丢掉
    train_batch_padding_dataset = train_dataset.padded_batch(args.batch_size, padded_shapes=padded_shape,
                                                             drop_remainder=True)#false

    # make_initializable_iterator()迭代器，构建train_batch_padding_dataset迭代器
    train_iterator = train_batch_padding_dataset.make_initializable_iterator()


    #对测试数据进行相同的数据处理，固定格式、构建迭代器
    test_batch_padding_dataset = test_dataset.padded_batch(args.batch_size, padded_shapes=padded_shape,drop_remainder=True)
    test_iterator = test_batch_padding_dataset.make_initializable_iterator()

    valid_batch_padding_dataset = valid_dataset.padded_batch(args.batch_size, padded_shapes=padded_shape,
                                                             drop_remainder=True)
    valid_iterator = valid_batch_padding_dataset.make_initializable_iterator()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    time0 = time.time()
    log_path = '../Log/%s_%s.log' % (args.dataset, str(int(time0)))
    log = Logging(log_path)
   #存储日志，日志路径
    log.print('Initializing model...')
    #**************************运行model函数，初始化模型，初始化权重矩阵，初始化学习参数
    model = Model(args, n_users, n_items, n_bizs, f_max_len, u_max_pack, pack_max_nei_b, pack_max_nei_f, u_max_i, u_max_f)
    #通过迭代器获取元素
    train_data = train_iterator.get_next()
    test_data = test_iterator.get_next()
    valid_data = valid_iterator.get_next()

    #variable_scope()变量作用域, reuse=true变量可分享，重用
    with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
        print("User items: {}".format(train_data['pack_neighbors_f']))
        train_loss, train_score, train_opt, att, u_e, u, p= model.forward(train_data['user'], train_data['item'], train_data['biz'],
                                              train_data['friends'], train_data['user_items'],
                                              train_data['user_bizs'],train_data['user_friends'],
                                              train_data['user_packages'], train_data['pack_neighbors_b'],
                                              train_data['pack_neighbors_f'], train_data['label'], train_data['label2'], theta,d,
                                                                          train=True)
        # print("*****************************")
        # print("label2:",u)
        # print("User items: {}".format(train_data['user_items']))


    with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
        test_loss, test_score, a1,a2,a3 = model.forward(test_data['user'], test_data['item'], test_data['biz'],
                                              test_data['friends'], test_data['user_items'],
                                              test_data['user_bizs'],test_data['user_friends'],
                                              test_data['user_packages'], test_data['pack_neighbors_b'],
                                              test_data['pack_neighbors_f'], test_data['label'], test_data['label2'],  theta,d,train=False)
    with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
        valid_loss, valid_score, _a1,_a2,_a3 = model.forward(valid_data['user'], valid_data['item'], valid_data['biz'],
                                              valid_data['friends'], valid_data['user_items'],
                                              valid_data['user_bizs'],valid_data['user_friends'],
                                              valid_data['user_packages'], valid_data['pack_neighbors_b'],
                                              valid_data['pack_neighbors_f'], valid_data['label'], valid_data['label2'], theta,d, train=False)

    # with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
    #     index=random.randrange(100)
    #     test1_data=test_data['user'].pop(index)
    #     test1_loss, test1_score, a11,a12,a13 = model.forward(test1_data, test_data['item'], test_data['biz'],
    #                                           test_data['friends'], test_data['user_items'],
    #                                           test_data['user_bizs'],test_data['user_friends'],
    #                                           test_data['user_packages'], test_data['pack_neighbors_b'],
    #                                           test_data['pack_neighbors_f'], test_data['label'], test_data['label2'], train=False)

    config = tf.ConfigProto()
    # #配置tf.Session的运算方式, GPU运算或者CPU运算
    config.gpu_options.allow_growth = True

    with tf.Session() as sess:    #config=config
        #初始化模型的参数
        tf.local_variables_initializer().run()
        sess.run(tf.global_variables_initializer())
        step = 0

        for epoch in range(args.epoch):
            sess.run([train_iterator.initializer, test_iterator.initializer])
            #初始化迭代器
            # sess.run(train_iterator.initializer)
            t0 = time.time()
            step = 0
            loss = []
            log.print('start training: ')
            score = []
            label = []
            while True:
                try:
                    if step > (n_train*args.ratio)/args.batch_size:#步数*学习率, 学习率设置为1
                        break
                    # rand=np.random.random_sample([100,64])
                    # u_e.load(rand)
                    # u_e.load(np.random.laplace(0.0,5.0,(100,64)))
                    loss_, _, sco, lab,att1,b,u1,p1 = sess.run([train_loss, train_opt, train_score,train_data['label'],att,u_e,u,p])
                    # print("原始user_嵌入：",u1)
                    # print("加噪后用户嵌入：",p1)
                    # noisydata=p1-u1

                    score.extend(sco)
                    label.extend(lab)
                    loss.append(loss_)
                    step += 1
                    # print("step:",step)
                    if step % 1000 == 0:
                        print(step)
                        tr_auc = roc_auc_score(label, score)
                        print('pre_train auc:%.4f\t' % tr_auc)
                        print("lable:", label[1:10])
                        print("score:", score[1:10])

                        sess.run(valid_iterator.initializer)

                        _val_loss, auc, f1, acc, prec, rec,err = eval_epoch(args,sess, valid_score, valid_loss, valid_data,_a1,_a2,_a3)
                        print('---After %d steps' % (step),
                                  'train_loss:%.4f\tvalid_loss:%.4f\tauc:%.4f' % (loss_, _val_loss, auc))
                except tf.errors.OutOfRangeError:
                    break
            t1 = time.time()
            # print("b值：", b)
            # print("user_emb:",u1)
            # print('pre_user_emb:',p1)
            print('finish training: %.4fs'%(t1-t0))
            log.print('start predicting: ')
            _test_loss, auc, f1, acc, prec, rec,err = eval_epoch(args,sess, test_score, test_loss, test_data,a1,a2,a3,test=True)
            _train_loss = np.mean(np.array(loss))
            t2 = time.time()
            tr_auc = roc_auc_score(label, score)
            print("error:",err)
            # print("label:",label,'score:',score)
            # print(test_score,'VS',test1_score)
            print("acc:%.4f\t" %acc)
            print('train auc:%.4f\t' % tr_auc)
            print('Epoch:%d\ttime: %.4fs\ttrain loss:%.4f\ttest loss:%.4f\tauc:%.4f' %
                       (epoch, (t2 - t1), _train_loss, _test_loss, auc))
            epoch_param.append(epoch)
            train_auc.append(tr_auc)
            acc_param.append(acc)
            auc_param.append(auc)
            f1_param.append(f1)
            recall_param.append(rec)
            # file = open('../result/' +'shapi_'+ args.dataset + '.txt', 'a')
            # file.write(str(epoch) + '\n')
            # file.write('auc:'+str(auc) +',train_loss:'+str(_train_loss)+',test_loss:'+str(_test_loss)+ '\n')
        recall_param=pd.Series(recall_param)
        epoch_param=pd.Series(epoch_param)
        train_auc=pd.Series(train_auc)
        acc_param=pd.Series(acc_param)
        auc_param=pd.Series(auc_param)
        f1_param=pd.Series(f1_param)
        graph_param=pd.concat([epoch_param,train_auc,acc_param,auc_param,f1_param,recall_param],axis=1)

        graph_param.to_csv('8.10-version-param_efficient-lr='+str(args.lr)+'d='+str(d)+'drop-out='+str(args.drop_out)+'-theta='+str(theta)+'.csv',header=['epoch','train_auc','acc','test_auc','f1','recall'],index=False)
        # graph_param.to_csv('param_efficient-initial.csv',header=['epoch','train_auc','acc','test_auc','f1','recall'],index=False)


if __name__ == "__main__":
    for d in{1}:
        for theta in {0.01}:
            print("************************************************")
            print("隐私预算:",theta)
            print("************************************************")
            p = Pool(1)
            p.apply(train_process,(theta,d,))
            print('进程结束..')
            p.close()
            p.join()



