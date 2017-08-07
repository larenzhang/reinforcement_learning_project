#!/usr/bin/env python
# coding=utf-8
import numpy as np
import tensorflow as tf

class policy_gradient:
    def __init__(
            self,
            n_features,
            n_actions,
            learning_rate,
            decay,
            is_restore
    ):
        self.n_features = n_features
        self.n_actions  = n_actions
        self.lr         = learning_rate
        self.gamma      = decay
        self.obs     = []
        self.acts    = []
        self.rewards = []
        self.episode_num = 0
        self.build_network()
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.saver = tf.train.Saver()
        
        if is_restore:
            checkpoint = tf.train.get_checkpoint_state("saved_networks")
            if checkpoint and checkpoint.model_checkpoint_path:
                self.saver.restore(self.sess, checkpoint.model_checkpoint_path)

    def build_network(self):
        self.tf_obs = tf.placeholder(tf.float32,[None,self.n_features])
        self.tf_act = tf.placeholder(tf.int32,[None,])
        self.tf_vt  = tf.placeholder(tf.float32,[None,])
            
        layer1 = tf.layers.dense(
            inputs=self.tf_obs,
            units=20,
            activation=tf.nn.relu,
            use_bias=True,
            kernel_initializer=tf.random_normal_initializer(mean=0,stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            trainable=True,
            name=None
        )

        action_outputs = tf.layers.dense(
            inputs=layer1,
            units=self.n_actions,
            activation=None,
            use_bias=True,
            kernel_initializer=tf.random_normal_initializer(mean=0,stddev=0.3),
            bias_initializer=tf.constant_initializer(0,1),
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            trainable=True,
            name=None
        )
        
        self.all_action_pro = tf.nn.softmax(action_outputs)
        #define loss
        #self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.tf_act,\
        #        logits=self.all_action_pro)
        self.cross_entropy = tf.reduce_sum(-tf.log(self.all_action_pro)*\
                tf.one_hot(self.tf_act,self.n_actions),axis=1)
        loss = tf.reduce_mean(self.cross_entropy*self.tf_vt)
        #define train step
        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(loss)
    
    def choose_action(self,obs):
        action_probability = self.sess.run(self.all_action_pro,\
                feed_dict={self.tf_obs:obs[np.newaxis,:]})
        action = np.random.choice(range(action_probability.shape[1]),p=action_probability.ravel())
        return action

    def state_value(self):
        value = 0
        discount_episode_rewards = np.zeros(len(self.rewards)) 
        for t in reversed(range(0,len(self.rewards))):
            value = value*self.gamma+self.rewards[t]
            discount_episode_rewards[t] = value
        return discount_episode_rewards
   
    def get_gt(self):
        value = 0
        gt_list = np.zeros(len(self.rewards))
        for t in reversed(range(0,len(self.rewards))):
            value += self.rewards[t]
            gt_list[t] = value
        gt_mean = np.mean(gt_list)
        gt_std  = np.std(gt_list)
        for i  in range(len(gt_list)):
            gt_list[i] = (gt_list[i]-gt_mean)/gt_std

        return gt_list

    def save_transition(self,observation,action,reward):
        self.obs.append(observation)
        self.acts.append(action)
        self.rewards.append(reward)

    def learn(self):
        gt = self.get_gt()
        #print("old obs:{0},acts:{1},vt:{2}".format(np.shape(self.obs),\
        #        np.shape(self.acts),np.shape(vt)))
        #print("obs:{0},acts:{1},vt:{2}".format(np.shape(np.vstack(self.obs)),\
        #        np.shape(np.array(self.acts)),np.shape(vt)))
        train,cross_entropy,tf_vt = self.sess.run([self.train_step,self.cross_entropy,self.tf_vt],feed_dict=\
                {self.tf_obs:np.vstack(self.obs),
                 self.tf_act:np.array(self.acts),
                 self.tf_vt:gt
                })

        #print("cross_entropy:{0},tf_vt:{1}".format(np.shape(cross_entropy),np.shape(tf_vt)))
        self.obs     = []
        self.acts    = []
        self.rewards = []
        self.episode_num += 1
        if self.episode_num%50 == 0:
            self.saver.save(self.sess,"saved_networks/"+"cart-pole-{0}th-episode".\
                    format(self.episode_num))
