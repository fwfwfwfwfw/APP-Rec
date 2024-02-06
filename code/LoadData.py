# -*- coding: utf-8 -*-
# @Create Time : 2020/7/13 15:54
# @Author : lee
# @FileName : LoadData.py
import  tensorflow as tf

def parse_function_(f_max_len):
    def parse_function(example_proto):
        dics = {'user': tf.FixedLenFeature(shape=(), dtype=tf.int64),  #FixedLenFeature:定长特征, VarLenFeature:变长特征, FixedLenSequenceFeature: 定长序列特征
                'item': tf.FixedLenFeature(shape=(), dtype=tf.int64),
                'biz': tf.FixedLenFeature(shape=(), dtype=tf.int64),
                'friends': tf.VarLenFeature(dtype=tf.int64),
                'user_items': tf.VarLenFeature(dtype=tf.int64),
                'user_bizs': tf.VarLenFeature(dtype=tf.int64),
                'user_friends': tf.VarLenFeature(dtype=tf.int64),
                'user_packages': tf.VarLenFeature(dtype=tf.int64),
                'pack_neighbors_b': tf.VarLenFeature(dtype=tf.int64),
                'pack_neighbors_f': tf.VarLenFeature(dtype=tf.int64),
                # 'f_max_len':tf.FixedLenFeature(dtype=tf.int64),
                'label': tf.FixedLenFeature(shape=(), dtype=tf.int64),
                'label2': tf.FixedLenFeature(shape=(), dtype=tf.int64)
                }
        parsed_example = tf.parse_single_example(example_proto, dics)  #parse_single_example：解析Example原型, serialized:标量字符串Tensor, 单个序列化Example,
        #定义空集合，规定参数类型
        print("parsed_example:{}".format(parsed_example['user_items']))
        print("dict:{}".format(dics))
        print("example_proto:{}".format(example_proto))

        #parse_single_example：features: 字典映射到FixedLenFeature或者VarLenFeature
        parsed_example['friends'] = tf.sparse_tensor_to_dense(parsed_example['friends'])  #tf.sparse_tensor_to_dense()得到numpy类型的数值
        parsed_example['user_items'] = tf.sparse_tensor_to_dense(parsed_example['user_items'])[:273]
        # parsed_example['user_items'],_ = parsed_example['user_items'],)
        parsed_example['user_bizs'] = tf.sparse_tensor_to_dense(parsed_example['user_bizs'])[:273]
        # parsed_example['user_bizs'],_ = parsed_example['user_bizs'],[50,825-50])
        parsed_example['user_friends'] = tf.sparse_tensor_to_dense(parsed_example['user_friends'])[:289]
        # parsed_example['user_friends'],_ = tf.split(parsed_example['user_friends'],[50,825-50],axis=0)
        parsed_example['user_packages'] = tf.sparse_tensor_to_dense(parsed_example['user_packages'])


        parsed_example['user_packages'] = tf.reshape(parsed_example['user_packages'], [-1, f_max_len + 2])[:50,:]
        # parsed_example['user_packages'],_ = tf.split(parsed_example['user_packages'],[50,825-50],axis=0)
        parsed_example['pack_neighbors_b'] = tf.sparse_tensor_to_dense(parsed_example['pack_neighbors_b'])
        parsed_example['pack_neighbors_b'] = tf.reshape(parsed_example['pack_neighbors_b'], [-1, f_max_len + 2])

        parsed_example['pack_neighbors_f'] = tf.sparse_tensor_to_dense(parsed_example['pack_neighbors_f'])
        parsed_example['pack_neighbors_f'] = tf.reshape(parsed_example['pack_neighbors_f'], [-1, f_max_len + 2])
        return parsed_example
    return parse_function