# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 15:16:25 2018
朴素贝叶斯实现情感分析
@author: Will
"""

import numpy as np


#%%
def load_dataset():
	sent_list = [['蒙牛', '很', '牛'],
	['蒙牛', '又', '出来', '丢人', '了'],
	['奖品', '很', '给力', '为', '蒙牛', '为', '奖品'],
	['珍爱', '生命', '远离', '蒙牛'],
	['蒙牛', '大果粒', '就是', '好吃', '好吃', '好吃'],
	['好在', '一直', '不吃', '蒙牛']]
	
	class_vec = [1, -1, 1, -1, 1, -1]
	
	return sent_list, class_vec
#%%
 
def create_vocab_list(dataset):
	vocab_set = set([])
	
	for doc in dataset:    
		vocab_set = vocab_set | set(doc)
	
	return list(vocab_set)
#%%
def set_of_words2vec(vocab_list, input_set):
	return_vec = [0] * len(vocab_list)
	
	for word in input_set:
		if word in vocab_list:
			return_vec[vocab_list.index(word)] = 1
	
	return return_vec
#%%
def trainNB(train_matrix, train_catagory):
	num_train_docs = len(train_matrix)  #6
	num_words = len(train_matrix[0])    #19
	pos_num = 0
	for i in train_catagory:
		if i == 1:
			pos_num += 1
	pAbusive = pos_num / float(num_train_docs) # 3/6
	p0_num = np.ones(num_words)  #[1,1,1,1,1.....]  19  拉普拉斯平滑
	p1_num = np.ones(num_words)
	p0_demon = 2.0
	p1_demon = 2.0   #也是拉普拉斯平滑  λ*k
	
	for i in range(num_train_docs):
		if train_catagory[i] == 1:
			p1_num += train_matrix[i]
			p1_demon += sum(train_matrix[i])
		else:
			p0_num += train_matrix[i]
			p0_demon += sum(train_matrix[i])
	
	p1_vect = np.log(p1_num / p1_demon)
	p0_vect = np.log(p0_num / p0_demon)
	
	return p0_vect, p1_vect, pAbusive
#%%
def classifyNB(vec2classify, p0_vec, p1_vec, pClass1):
	p1 = sum(vec2classify * p1_vec) + np.log(pClass1)
	p0 = sum(vec2classify * p0_vec) + np.log(1.0 - pClass1)
	
	if p1 > p0:
		return 1
	elif p0 > p1:
		return -1
	else:
		return 0
#%%	
	
list_sents, list_classes = load_dataset()
my_vocab_list = create_vocab_list(list_sents)
train_mat = []
for sent_in_doc in list_sents:
	train_mat.append(set_of_words2vec(my_vocab_list, sent_in_doc))
 
p0V, p1V, pAb = trainNB(train_mat, list_classes)
test_entry1 = ['蒙牛', '真', '好吃', '好', '给力']
test_entry2 = ['再也', '不吃', '蒙牛', '了']
 
print(classifyNB(np.array(set_of_words2vec(my_vocab_list, test_entry1)), p0V, p1V, pAb))
print(classifyNB(np.array(set_of_words2vec(my_vocab_list, test_entry2)), p0V, p1V, pAb))
