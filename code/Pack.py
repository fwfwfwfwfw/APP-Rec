# -*- coding: utf-8 -*-
# @Create Time : 2020/7/13 15:10
# @Author : lee
# @FileName : Pack.py
import copy

import tensorflow as tf
import math
import numpy as np

class Model():
    def __init__(self, args, n_users, n_items, n_bizs, f_max_len, u_max_pack, pack_max_nei_b, pack_max_nei_f, u_max_i, u_max_f):

        self.batch_size = args.batch_size  #导入args参数
        self.K = args.K              #
        self.lr = args.lr
        self.reg = args.reg
        self.n_users = n_users
        self.n_items = n_items
        self.n_bizs = n_bizs
        self.emb_dim = args.dimension
        self.f_max_len, self.u_max_pack, self.pack_max_nei_b, self.pack_max_nei_f = f_max_len, u_max_pack, pack_max_nei_b, pack_max_nei_f
        self.u_max_i, self.u_max_f = u_max_i, u_max_f
        self.params = [] #参数，包含 user嵌入、文章嵌入、和媒体嵌入
        self.stdv = 1.0 / math.sqrt(self.emb_dim) #嵌入纬度的开方倒数
        self.keep_prob = 1 - args.drop_out  #神经元激活概率
        #************************生成用户-文章-媒体三个矩阵的初始化权重矩阵
        self.user_embedding = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(), shape=[self.n_users, self.emb_dim],
                                              dtype=tf.float32, name='user_embedding')
        #tf.contrib.layers.xavier_initializer()输出初始化权重矩阵，初始化具有明确值变量，给出大小，数据类型和目的
        self.user_embedding = tf.concat([tf.zeros(shape=[1, self.emb_dim]), self.user_embedding], 0)
        #将零矩阵和用户嵌入矩阵连接起来，[0,0,...,0(emb_dim)]，[n_users,emdim],生成[n_users+1,emdim]
        self.item_embedding = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(), shape=[self.n_items, self.emb_dim],
                                              dtype=tf.float32, name='item_embedding')
        self.item_embedding = tf.concat([tf.zeros(shape=[1, self.emb_dim]), self.item_embedding], 0)
        self.biz_embedding = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(), shape=[self.n_bizs, self.emb_dim],
                                              dtype=tf.float32, name='biz_embedding')
        self.biz_embedding = tf.concat([tf.zeros(shape=[1, self.emb_dim]), self.biz_embedding], 0)
        # print("user_embedding:{0}".format(self.user_embedding.shape))
        # print("biz_embedding{0}".format(self.biz_embedding.shape))
        # print("item_embedding:{0}".format(self.item_embedding.shape))


        self.params.append(self.user_embedding)
        self.params.append(self.item_embedding)
        self.params.append(self.biz_embedding)

        #**************初始化学习参数，包括权重参数w，偏移参数b和高度h
        self.weight_size = [64,32] #权重的列表定义
        self.n_layers = len(self.weight_size)
        print("item_size:",self.n_layers)
        self.weight_size_list = [3 * self.emb_dim] + self.weight_size

        #列表拼接[3*64,64,32]
        self.weights = {}
        initializer = tf.contrib.layers.xavier_initializer()
        #初始化权重矩阵
        for i in range(self.n_layers):
            self.weights['W_%d' %i] = tf.Variable(
                #学习权重参数w_1,w_2
                initializer([self.weight_size_list[i], self.weight_size_list[i+1]]), name='W_%d' %i)
            self.weights['b_%d' %i] = tf.Variable(
                #学习偏移参数b_1,b_2
                initializer([1, self.weight_size_list[i+1]]), name='b_%d' %i)

        self.weights['h'] = tf.Variable(initializer([self.weight_size_list[-1], 1]), name='h')


    def laplace(slef, mu,b,l_shape):
        # alpha=tf.random_uniform(l_shape,minval=-0.5,maxval=0.5,dtype='float32')
        #
        # noise=mu-b*tf.sign(alpha)*tf.log(tf.subtract(tf.ones(shape=l_shape),2*tf.abs(alpha)))
        # noise=np.random.laplace(mu,b,l_shape)
        #拉普拉斯噪声
        noise=tf.distributions.Laplace(mu,b).sample(l_shape)
        # print(noise)
        return noise

    def forward(self, user, item, biz, friends, user_items, user_bizs, user_friends, user_packages,
                pack_neighbors_b, pack_neighbors_f, label, label2, theta,d,train):
        #非训练集的时候，所有神经单元激活
        if not train:
            self.keep_prob = 1

        #tf.nn.embedding_lookup(params,ids), params：表示完整的嵌入张量 或者除了第一维度之外的具有相同形状的P个张量列表,
        # ids表示类型为int32或者int64的Tensor，包含要在params中查找的id
        #如user=[1,2],即在user_embedding中找，第二个和第三个元素
        #将id转换为向量
        user_emb = tf.nn.embedding_lookup(self.user_embedding, user) # B*D
        #从user_embedding中查询user序号的元素，user的batch_size=100, user_embeding为[n_user, emdim],user_emb[100,emdim]
        # print("label2:",label2)
        #从低层增加一个维度，expand_dims
        #user_packages用户和包，tf.sign激活函数，输出0,1，tf.cast()强制转换格式
        up_mask = tf.expand_dims(tf.nn.softmax(tf.cast(tf.abs(tf.reduce_sum(user_packages, axis=-1)), dtype=tf.float32)),-1) # B*N*1
        #pack_neighbors_b,包-邻居-media
        pb_mask = tf.expand_dims(tf.nn.softmax(tf.cast(tf.abs(tf.reduce_sum(pack_neighbors_b, axis=-1)), dtype=tf.float32)),-1)
        # pb_mask = tf.expand_dims(tf.cast(tf.sign(tf.abs(tf.reduce_sum(pack_neighbors_b, axis=-1))), dtype=tf.float32),-1)

        #pack_neighbors_f 包-邻居-friends, 数据格式(?,20,1)
        pf_mask = tf.expand_dims(tf.cast(tf.sign(tf.abs(tf.reduce_sum(pack_neighbors_f, axis=-1))), dtype=tf.float32),-1)

        #split,进行划分,axis=-1即按照列划分，分为列分别为[1,1,f_max_len]的三个数组
        [up_items, up_bizs, up_friends] = tf.split(user_packages, [1, 1, self.f_max_len], axis=-1)  # B * u_max_p * 1, B * u_max_p * 1, B * u_max_p * max_f
        [pb_items, pb_bizs, pb_friends] = tf.split(pack_neighbors_b, [1, 1, self.f_max_len], axis=-1) # B * p_max_nei(biz) * 1
        [pf_items, pf_bizs, pf_friends] = tf.split(pack_neighbors_f, [1, 1, self.f_max_len], axis=-1) # B * p_max_nei(fri) * 1
        #将item重建为一列矩阵，将item,up_items,pd_items,pf_items合并，axis=1,即按照列拼接
        _items = tf.concat([tf.reshape(item, [-1, 1, 1]), up_items, pb_items, pf_items], axis=1)
        _bizs = tf.concat([tf.reshape(biz, [-1, 1, 1]), up_bizs, pb_bizs, pf_bizs], axis=1)
        _friends = tf.concat([tf.expand_dims(friends, axis=1), up_friends, pb_friends, pf_friends], axis=1)

        
        #包与包之间的函数
        user_emb,a2,a3,ta = self.dual_aggregate(user_emb, user_items, user_bizs, user_friends,train)
        #user_emb = user_emb + tf.reduce_mean(u_packs*up_mask, axis=1)
        
        # print("intra_item,{}".format(tf.reshape(_bizs,[-1])))
        # print("self_embediing,{}".format(self.item_embedding))
        intra_packages,att = self.intra(user_emb, tf.reshape(_friends,[-1, self.f_max_len]), tf.reshape(_items,[-1]), tf.reshape(_bizs,[-1]), train)
        print(intra_packages,att)
        intra_packages = tf.reshape(intra_packages, [self.batch_size, -1, self.emb_dim])
        att = tf.reshape(att, [self.batch_size, -1, 7])
        [tar_pack, u_packs, pb_packs, pf_packs] = tf.split(intra_packages, [1, self.u_max_pack, self.pack_max_nei_b, self.pack_max_nei_f], axis=1)
        [tar_att, u_att, pb_att, pf_att] = tf.split(att, [1, self.u_max_pack, self.pack_max_nei_b, self.pack_max_nei_f], axis=1)
        tar_pack = tf.reshape(tar_pack, [self.batch_size, self.emb_dim])
        tar_att = tf.reshape(tar_att, [self.batch_size, 7])
        u_packs = tf.reshape(u_packs, [self.batch_size, -1, self.emb_dim])
        pb_packs = tf.reshape(pb_packs, [self.batch_size, -1, self.emb_dim])
        pf_packs = tf.reshape(pf_packs, [self.batch_size, -1, self.emb_dim])


        u_packs=tf.clip_by_value(u_packs,tf.zeros_like(u_packs),tf.ones_like(u_packs))
        # tar_pack = tf.reshape(intra_packages, [self.batch_size, self.emb_dim])
        
        # a = tf.reduce_mean(pb_packs*pb_mask, axis=1)
        # pack_emb = tar_pack + tf.reduce_mean(pb_packs*pb_mask, axis=1) + tf.reduce_mean(pf_packs*pf_mask, axis=1)
        
        # gate_attention
        #tf.reduce_mean(), tensor值沿着轴方向移动，取平均值
        #求出包嵌入
        pack_emb = tar_pack + tf.reduce_mean(self.gate_attention(tar_pack,pb_packs,pb_mask,self.emb_dim,'biz'), axis=1) \
                   + tf.reduce_mean(self.gate_attention(tar_pack,pf_packs,pf_mask,self.emb_dim,'friend'), axis=1)
        
        # gate_attention
        user_emb = user_emb + tf.reduce_mean(self.gate_attention(user_emb,u_packs,up_mask,self.emb_dim,'user'), axis=1)

        pre_user_emb=user_emb
        # print("用户嵌入：",user_emb)
        # print("用户[1]嵌入：", user_emb.shape[0].value)


        i_mu=0.0
        # b=tf.get_variable(initializer=tf.random_uniform(user_emb.shape,minval=10,maxval=15),name='privacy_b',trainable=True)
        # b=tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(),shape=user_emb.shape,name='privacy_b')
        # delta_f=1+2*tf.reduce_max(u_packs)

        delta_f=3*d
        print("deta_type:",u_packs.shape)


        b=delta_f/theta
        # laplacian_noise= tf.Variable(self.laplace(i_mu,b,user_emb.shape),dtype='float32',trainable=False)

        laplacian_noise= self.laplace(i_mu,b,user_emb.shape)

        # l=tf.Variable(tf.zeros(user_emb.shape))
        # laplacian_noise=tf.assign(l,np.random.laplace(i_mu,b,user_emb.shape))
        # laplacian_noise=tf.Variable(np.random.random_sample([100,64]),dtype='float32')

        # print("噪音：",laplacian_noise)
        # laplacian_noise=np.random.laplace(i_mu,b,user_emb.shape)

        user_emb=tf.add(user_emb,laplacian_noise)


        # print("加噪用户嵌入",user_emb)
        # w_user = tf.get_variable('w_user', initializer=tf.random_normal([self.batch_size,self.batch_size], stddev=0.1),trainable=True)
        # w_pack = tf.get_variable('w_pack', initializer=tf.random_normal([self.batch_size,self.batch_size], stddev=0.1),trainable=True)
        # self.params.extend([w_user,w_pack])
        # user_emb=tf.tensordot(w_user,user_emb,axes=1)
        # pack_emb=tf.tensordot(w_pack,pack_emb,axes=1)
        # pack_emb = tar_pack
        item_emb = tf.nn.embedding_lookup(self.item_embedding, item)
        
        #MLP多层感知机
        z = []
        z.append(tf.concat([user_emb, pack_emb, user_emb * pack_emb], 1))
        
        for i in range(self.n_layers):
            temp = tf.nn.relu(tf.matmul(z[i], self.weights['W_%d' % i]) + self.weights['b_%d' % i])
            temp = tf.nn.dropout(temp, self.keep_prob)
            print('tem',i,":",temp)
            z.append(temp)
        print("Z[-1]:",z[-1],'len:',len(z))
        agg_out = tf.matmul(z[-1], self.weights['h'])
        # tf.squeeze(), 删除所有大小=1的维度
        self.scores = tf.squeeze(agg_out)
        # 激活函数tf.sigmoid(), 就是y=1/(1+exp(-x))
        self.scores_normalized = tf.sigmoid(self.scores)
        #预测标签
        # laplacian_score_noise = self.laplace(i_mu, 1.0, self.scores.shape)
        # self.scores = tf.add(self.scores, laplacian_score_noise)

        self.predict_label = tf.cast(self.scores > 0.5, tf.int32)




        with tf.variable_scope('train'):

            #交叉损失
            base_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(label,tf.float32), logits=self.scores))
                
            #调整损失
            l2_loss = tf.Variable(tf.constant(0., dtype=tf.float32), trainable=False)
            for param in self.params:
                l2_loss = tf.add(l2_loss, self.reg * tf.nn.l2_loss(param))

            #差分隐私损失

            l3_loss = tf.keras.losses.KLD(pre_user_emb, user_emb)
            #l3_loss=tf.reduce_mean(tf.square(pre_user_emb-user_emb,name='l3'))
            # beta=tf.get_variable(initializer=tf.contrib.layers.xavier_initializer())
            # loss = base_loss + l2_loss

            loss = base_loss + l2_loss +l3_loss
            print('步骤一')
            optimizer = tf.train.AdamOptimizer(self.lr).minimize(loss)#以lr的学习率，优化损失函数

            print('步骤二')
            # laplace=tf.convert_to_tensor(laplacian_noise)
            # optimizer = tf.train.RMSPropOptimizer(self.lr).minimize(loss)
            # optimizer = tf.train.AdagradOptimizer(self.lr).minimize(loss)
            # max_pack=tf.reduce_max(u_packs)
        #回传用户嵌入
        if train:
            return loss, self.scores_normalized, optimizer, tar_att, laplacian_noise, user_emb, pre_user_emb
        else:
            return loss, self.scores_normalized, tar_att, user_emb, self.predict_label#a2,a3,ta
    


    def intra(self, user_emb, friends, item, biz, train):
        #tf.tile()为了匹配矩阵维度？
        #tf.tile(x,[2,3])即x的x=0维复制数值*2,x=1维复制数值*3
        #tf.expand_dims(x,axis=1)即在x的基础上增加一个维度
        _user_emb = tf.reshape(tf.tile(tf.expand_dims(user_emb, axis=1),
                                      [1, 1 + self.u_max_pack + self.pack_max_nei_b + self.pack_max_nei_f, 1]),
                              [-1, self.emb_dim, 1])  # BN*D*1
        friend_emb = tf.nn.embedding_lookup(self.user_embedding, friends)  # BN*F*D, N max neighbor size, M max friend size.
        masks = tf.sign(tf.abs(tf.reduce_sum(friend_emb, axis=-1)))  # BN*F
        #tf.nn.embedding_lookup函数选取张量里面索引对应的元素
        item_emb = tf.nn.embedding_lookup(self.item_embedding, item)  # BN*D
        biz_emb = tf.nn.embedding_lookup(self.biz_embedding, biz)  # BN*D
        

        # social influence
        f_list = []
        for i in range(self.K):
            w_k = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(), shape=[
                self.emb_dim, self.emb_dim], dtype=tf.float32, name='wk_%d' % i)
            w_i = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(), shape=[
                self.emb_dim , self.emb_dim], dtype=tf.float32, name='wi_%d' % i)

            # BN*F*D dot D*D -> BN*F*D
            f_k_emb = tf.tensordot(friend_emb, w_k, axes=1)  # BN*F*D
            _item = tf.expand_dims(tf.matmul(item_emb, w_i),1) # BN*1*D
            
            inputs = tf.concat([tf.tile(_item,[1,self.f_max_len,1]),f_k_emb],-1)
            w_omega = tf.get_variable('w_omega_%d'%i, initializer=tf.random_normal([2*self.emb_dim, 1], stddev=0.1))
            b_omega = tf.get_variable('b_omega_%d'%i, initializer=tf.random_normal([1], stddev=0.1))
            u_omega = tf.get_variable('o_omega_%d'%i, initializer=tf.random_normal([1], stddev=0.1))
            self.params.extend([w_k,w_i,w_omega])
            
            with tf.name_scope('v'):
                # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
                #  the shape of `v` is (BN,F,D)*(D,A)=(BN,F,A), where A=attention_size
                v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

            
            vu = tf.tensordot(v, u_omega, axes=1, name='vu') # BN*F
            paddings = tf.ones_like(vu) * (-2 ** 32 + 1)
            x = tf.where(tf.equal(masks, 0), paddings, vu)
            att = tf.nn.softmax(x, axis=-1)
            #keep_dims表示保持维度不变
            output = tf.reduce_sum(f_k_emb * tf.expand_dims(att, -1), 1, keep_dims=True) # BN*1*D
            #append()将列表整个打包加进来，即[a,b,c,[d,e]], extend()将内容加进来，即[a,b,c,d,e]
            f_list.append(output)  # K*BN*D
        
        f_K_emb = tf.concat(f_list,1)

        t_user = tf.reshape(tf.tile(tf.expand_dims(user_emb, axis=1),
                                      [1, 1 + self.u_max_pack + self.pack_max_nei_b + self.pack_max_nei_f, 1]),
                              [-1, 1, self.emb_dim])
        inputs = tf.concat([tf.tile(t_user,[1,self.K,1]),f_K_emb],-1)
        #tf.contrib.layers.xavier_initializer 初始化权重矩阵
        w_a = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(), shape=[
                2*self.emb_dim , self.emb_dim], dtype=tf.float32, name='w_a_d')

        inputs = tf.nn.relu(tf.tensordot(inputs,w_a, axes=1))

        inputs = tf.nn.dropout(inputs, self.keep_prob)
        w_omega = tf.get_variable('w_omega_d', initializer=tf.random_normal([self.emb_dim, 1], stddev=0.1))
        b_omega = tf.get_variable('b_omega_d', initializer=tf.random_normal([1], stddev=0.1))
        u_omega = tf.get_variable('o_omega_d', initializer=tf.random_normal([1], stddev=0.1))
        self.params.extend([w_omega])
        
        with tf.name_scope('v'):
            # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
            #  the shape of `v` is (BN,F,D)*(D,A)=(BN,F,A), where A=attention_size
            v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

        
        vu = tf.tensordot(v, u_omega, axes=1, name='vu') # BN*F
        
        att = tf.nn.softmax(vu, axis=-1)
        f_emb = tf.reduce_sum(f_K_emb * tf.expand_dims(att, -1), 1) # BN*D
        

        
        # interaction
        pack = [f_emb, item_emb, biz_emb, f_emb*item_emb, f_emb*biz_emb, item_emb*biz_emb, f_emb*item_emb*biz_emb]
        # pack = [item_emb, biz_emb, item_emb*biz_emb]
        pack = tf.transpose(pack, perm=[1, 0, 2])
        # user_emb B*D  pack BN*7*D
        # pack_emb, att,_,__ = self.attention(tf.transpose(_user_emb, perm=[0,2,1]), pack, 7, self.batch_size * (
                    # 1 + self.u_max_pack + self.pack_max_nei_b + self.pack_max_nei_f))
        # pack_emb = tf.reduce_mean(pack,1)
        masks = tf.sign(tf.abs(tf.reduce_sum(pack, axis=-1)))
        #tf.transpose(), 转置perm=[0,2,1]交换内层的两个维度，perm=[1,0,2]将最外两层进行转置
        _user_emb = tf.tile(tf.transpose(_user_emb, perm=[0,2,1]),[1,7,1])

        pack_emb, att = self._attention(_user_emb, pack, 2*self.emb_dim, self.emb_dim,'pack_attention',train, masks)
        
        return pack_emb,att
    #
    #
    # def inter(self, c_pack, n_packs):

    def dual_aggregate(self, user_emb, items, bizs, friends,train):
        friends_emb = tf.nn.embedding_lookup(self.user_embedding, friends) #B*M*D
        # print("Forward User items: {}".format(self.item_embedding))

        items_emb = tf.nn.embedding_lookup(self.item_embedding, items)  # B*N*D
        bizs_emb = tf.nn.embedding_lookup(self.biz_embedding, bizs)
        # self.params.append(user_emb)
        user_emb_ = tf.expand_dims(user_emb, axis=1)

        with tf.variable_scope("friends", reuse=tf.AUTO_REUSE):
            # friend_type, att1,m1,x1 = self.attention(user_emb_, friends_emb, self.u_max_pack*self.f_max_len, self.batch_size)
            f_masks = tf.sign(tf.abs(tf.reduce_sum(friends_emb, axis=-1)))
            _user_emb = tf.tile(user_emb_,[1,self.u_max_f,1])
            friend_type, att1 = self._attention(_user_emb,friends_emb, 2*self.emb_dim, self.emb_dim, 'friends', train,f_masks)
            
            ### self connection
            # w = tf.get_variable('wf', initializer=tf.contrib.layers.xavier_initializer(), shape=[2*self.emb_dim, self.emb_dim])
            # friend_type = tf.nn.relu(tf.matmul(tf.concat([friend_type, user_emb],-1), w))

        with tf.variable_scope("items", reuse=tf.AUTO_REUSE):
            # item_type, att2,m2,x2 = self.attention(user_emb_, items_emb, self.u_max_pack, self.batch_size)
            i_masks = tf.sign(tf.abs(tf.reduce_sum(items_emb, axis=-1)))
            #tf.tile()复制user_emb_中元素，按照所在维度，分别复制1，self.u_max_i,1份，组成大矩阵
            _user_emb = tf.tile(user_emb_,[1,self.u_max_i,1])
            ### inputs = tf.concat([_user_emb, items_emb], -1)
            #*************************************执行自注意力机制
            item_type, att2 = self._attention(_user_emb,items_emb, 2*self.emb_dim, self.emb_dim,'items', train,i_masks)
            
            # w = tf.get_variable('wi', initializer=tf.contrib.layers.xavier_initializer(), shape=[2*self.emb_dim, self.emb_dim])
            # item_type = tf.nn.relu(tf.matmul(tf.concat([item_type, user_emb],-1), w))
            
        with tf.variable_scope("bizs", reuse=tf.AUTO_REUSE):
            # biz_type, att3,m,x = self.attention(user_emb_, bizs_emb, self.u_max_pack, self.batch_size)  # B*D
            b_masks = tf.sign(tf.abs(tf.reduce_sum(bizs_emb, axis=-1)))
            _user_emb = tf.tile(user_emb_,[1,self.u_max_i,1])
            ### inputs = tf.concat([_user_emb, bizs_emb], -1)
            biz_type, att2 = self._attention(_user_emb, bizs_emb, 2*self.emb_dim, self.emb_dim,'bizs', train, b_masks)
            
            # w = tf.get_variable('wbi', initializer=tf.contrib.layers.xavier_initializer(), shape=[2*self.emb_dim, self.emb_dim])
            # biz_type = tf.nn.relu(tf.matmul(tf.concat([biz_type, user_emb],-1), w))
            
        with tf.variable_scope("type_attention", reuse=tf.AUTO_REUSE):
            # n_emb = tf.concat([tf.expand_dims(friend_type, axis=1), tf.expand_dims(item_type, axis=1),
                               # tf.expand_dims(biz_type, axis=1)], axis=1)
            # _user_emb, t_att,mm,xx = self.attention(user_emb_, n_emb, 3, self.batch_size)
            
            inputs = tf.concat([tf.expand_dims(friend_type, axis=1), tf.expand_dims(item_type, axis=1),
                               tf.expand_dims(biz_type, axis=1)], axis=1)
            masks = tf.sign(tf.abs(tf.reduce_sum(inputs, axis=-1)))
            _user_emb = tf.tile(user_emb_,[1,3,1])
            #### inputs = tf.concat([_user_emb, inputs], -1)
            _user_emb, t_att = self._attention(_user_emb, inputs, 2*self.emb_dim, self.emb_dim,'type_attention',train, masks)
            
            #### self connection
            w = tf.get_variable('w_self', initializer=tf.contrib.layers.xavier_initializer(), shape=[2*self.emb_dim, self.emb_dim])
            user_emb = tf.nn.relu(tf.matmul(tf.concat([_user_emb, user_emb],-1), w))
        
        
        # user_emb = tf.reduce_mean(tf.concat([tf.reduce_mean(friends_emb,axis=1,keep_dims=True),tf.reduce_mean(items_emb,axis=1,keep_dims=True),tf.reduce_mean(bizs_emb,axis=1,keep_dims=True)],axis=1),axis=1)
        return user_emb, att1,att1,att1


    def attention(self, user_emb, node_emb, n_nodes, batch_size):
        #tile 复制
        user_emb = tf.tile(user_emb, [1, n_nodes, 1])
        #sign=x/|x|, abs=|x|
        masks = tf.sign(tf.abs(tf.reduce_sum(node_emb, axis=-1)))  # B*N
        # masks = tf.expand_dims(masks, axis=-1)
        w1 = tf.get_variable('w1', initializer=tf.contrib.layers.xavier_initializer(), shape=[2*self.emb_dim, self.emb_dim])
        b1 = tf.get_variable('b1', initializer=tf.contrib.layers.xavier_initializer(), shape=[self.emb_dim])
        w2 = tf.get_variable('w2', initializer=tf.contrib.layers.xavier_initializer(), shape=[self.emb_dim, 1])
        b2 = tf.get_variable('b2', initializer=tf.contrib.layers.xavier_initializer(), shape=[1])
        self.params.extend([w1,w2])
        # print("-----\n",w1)
        x = tf.reshape(tf.concat([user_emb, node_emb], axis=-1), [-1, 2*self.emb_dim])
        #tf.nn.rulu()线性整流函数，定义横坐标左边（小于0）为零，右边（大于0）等于自身,即为激活函数
        #tf.matmul()即矩阵乘法
        x = tf.nn.relu(tf.matmul(x, w1)+b1)
        #tf.nn.dropout()是防止或减轻过拟合使用的函数，一般用于全连接层
        x = tf.nn.dropout(x, self.keep_prob)
        x = tf.matmul(x, w2)  # BN*1
        # x = tf.nn.dropout(x, self.keep_prob)
        x = tf.reshape(x, [-1,n_nodes])
        #ones_like()构建与x矩阵格式相同的矩阵，并全部用1填充，*2表示所有元素乘以2，**2表示所有元素平方，+1表示所有元素加一
        paddings = tf.ones_like(x) * (-2 ** 32 + 1)
        #tf.where(tensor,a,b)，若tensor[x]==true，a[x]=a,否则 a[x]=b[x]
        x = tf.where(tf.equal(masks, 0), paddings, x)
        #softmax回归
        att = tf.nn.softmax(x, axis=-1)
        return tf.reduce_sum(tf.expand_dims(att,-1) * node_emb, axis=1), att,masks,x
    
    def _attention(self, inputs1,inputs2, emb_dim1, emb_dim2, name_scope, train, masks = None):
        with tf.variable_scope(name_scope,reuse=tf.AUTO_REUSE):
            #tf.get_variable()创建tensorflow变量，initializer是初始化方式，tf.random_normal_initializer()表示正太分布初始化器,
            # stddev表示正态分布的标准差，默认为1
            w_omega = tf.get_variable('w_omega', initializer=tf.random_normal([emb_dim2, 1], stddev=0.1))
            w = tf.get_variable('w_t', initializer=tf.random_normal([emb_dim1, emb_dim2], stddev=0.1))
            b_omega = tf.get_variable('b_omega', initializer=tf.random_normal([1], stddev=0.1))
            u_omega = tf.get_variable('o_omega', initializer=tf.random_normal([1], stddev=0.1))
            #params后面追加列表[w_omega,w]
            self.params.extend([w_omega,w])
            #从列增加嵌入值（如用户嵌入+朋友嵌入）
            inputs = tf.concat([inputs1,inputs2],-1)
            # inputs = tf.nn.relu(tf.tensordot(inputs,w, axes=1))
            #tf.tensordot（）即tensor矩阵乘法，axes=1表示，将前一个矩阵的最后一个维度与后一个矩阵的第一个维度矩阵相乘
            #tf.layers.batch_normalization()用来构建待训练的神经网络模型
            inputs = tf.nn.relu(tf.layers.batch_normalization(tf.tensordot(inputs,w, axes=1),training=train))
            #tf.nn.dropout()中的keep_prob表示每一个元素被保存下来的概率
            #tf.layer.dropout()中的rate表示每个元素丢弃的概率，keep_prob=1-rate
            inputs = tf.nn.dropout(inputs, self.keep_prob)
            with tf.name_scope('v'):
                # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
                #  the shape of `v` is (BN,F,D)*(D,A)=(BN,F,A), where A=attention_size
                v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

            
            vu = tf.tensordot(v, u_omega, axes=1, name='vu') # BN*F
            
            paddings = tf.ones_like(vu) * (-2 ** 32 + 1)
            vu = tf.where(tf.equal(masks, 0), paddings, vu)
            att = tf.nn.softmax(vu, axis=-1)
            f_emb = tf.reduce_sum(inputs2 * tf.expand_dims(att, -1), 1) # BN*D
            return f_emb, att
            
    def gate_attention(self, input1, input2, mask, emb_dim, name_scope):
        with tf.variable_scope(name_scope,reuse=tf.AUTO_REUSE):


            w_g1 = tf.get_variable('w_gate1', initializer=tf.random_normal([emb_dim, emb_dim], stddev=0.1))
            w_g2 = tf.get_variable('w_gate2', initializer=tf.random_normal([emb_dim, emb_dim], stddev=0.1))
            b = tf.get_variable('b_gate', initializer=tf.random_normal([emb_dim], stddev=0.1))
            #tf.nn.sigmoid(), y=1/(1+exp(-x))
            att = tf.nn.sigmoid(tf.expand_dims(tf.matmul(input1, w_g1),1) + tf.tensordot(input2, w_g2,axes=1) + b)
            #gi=sigma(w3u+w4p+b),
            att=tf.nn.softmax(att,axis=-1)
            att = att * mask
            return att * input2