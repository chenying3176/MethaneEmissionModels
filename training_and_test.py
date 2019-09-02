#coding:utf-8
import os
import xlrd
import math
import copy
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
import numpy
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import ensemble
from sklearn import neighbors
from sklearn import tree

import torch.nn as nn
import torch
from torch.autograd import Variable 
import torch.nn.functional as F
import torch.utils.data as Data

from sklearn import svm
from sklearn.tree import DecisionTreeRegressor
from sklearn import ensemble
from sklearn.neural_network import MLPRegressor

import numpy as np
import bagging_models



def load_CO2_CV_data(exp_data_dir_path, file_tail, j, region_type):
		cr_id_list = []
		cr_feat_list = []
		cr_tag_list = []
		for i in range(10):
			r = open(exp_data_dir_path + str(j) + '/' + str(i) + '/'+ file_tail, 'r')
			id_list = []
			feat_list = []
			tag_list = [] 
			lines_train = r.readlines()
			for line in lines_train:
				line = line.strip()
				tl = []
				strs = line.split('\t')
				id = float(strs[0])
				if region_type == 'LIT':
					if id >= 373.0 and id <= 381.0: continue   
				id_list.append(id)
				tag_list.append(float(strs[1]))
				for i in range(2, 15):   
					s = strs[i]
					tl.append(float(s)) 
				feat_list.append(tl)
			cr_id_list.append(id_list)
			cr_feat_list.append(feat_list)
			cr_tag_list.append(tag_list)

		return cr_id_list, cr_feat_list, cr_tag_list



def load_CH4_CV_data(exp_data_dir_path, file_tail, j, region_type):
		cr_id_list = []
		cr_feat_list = []
		cr_tag_list = []
		for i in range(10):
			r = open(exp_data_dir_path + str(j) + '/' + str(i) + '/'+ file_tail, 'r')
			id_list = []
			feat_list = []
			tag_list = []
			lines_train = r.readlines()
			for line in lines_train:
				line = line.strip()
				tl = []
				strs = line.split('\t')  
				id = float(strs[0])
				if region_type == 'LIT':
					if id >= 373.0 and id <= 381.0: continue  
				id_list.append(id)
				tag_list.append(float(strs[1])) 
				for i in range(2, 15):    
					s = strs[i]
					tl.append(float(s)) 
				 
				feat_list.append(tl)
			cr_id_list.append(id_list)
			cr_feat_list.append(feat_list)
			cr_tag_list.append(tag_list)

		return cr_id_list, cr_feat_list, cr_tag_list

def load_CO2_data(exp_data_dir_path, file_tail, region_type):
			cr_id_list = []
			cr_feat_list = []
			cr_tag_list = []
		 
			r = open(exp_data_dir_path  + '/'+ file_tail, 'r')
			id_list = []
			feat_list = []
			tag_list = [] 
			lines_train = r.readlines()
			for line in lines_train:
				line = line.strip()
				tl = []
				strs = line.split('\t')
				id = float(strs[0])
				if region_type == 'LIT':
					if id >= 373.0 and id <= 381.0: continue   
				id_list.append(id)
				tag_list.append(float(strs[1]))
				for i in range(2, 15):   
					s = strs[i]
					tl.append(float(s)) 
				feat_list.append(tl)
			cr_id_list.append(id_list)
			cr_feat_list.append(feat_list)
			cr_tag_list.append(tag_list)

			return cr_id_list, cr_feat_list, cr_tag_list



def load_CH4_data(exp_data_dir_path, file_tail,  region_type):
			cr_id_list = []
			cr_feat_list = []
			cr_tag_list = [] 

			r = open(exp_data_dir_path  + '/'+ file_tail, 'r')
			id_list = []
			feat_list = []
			tag_list = []
			lines_train = r.readlines()
			for line in lines_train:
				line = line.strip()
				tl = []
				strs = line.split('\t')  
				id = float(strs[0])
				if region_type == 'LIT':
					if id >= 373.0 and id <= 381.0: continue  
				id_list.append(id)
				tag_list.append(float(strs[1])) 
				for i in range(2, 15):    
					s = strs[i]
					tl.append(float(s)) 
				 
				feat_list.append(tl)
			cr_id_list.append(id_list)
			cr_feat_list.append(feat_list)
			cr_tag_list.append(tag_list)

			return cr_id_list, cr_feat_list, cr_tag_list


def data_normalization(train, val, test, y_train, y_val, y_test, all_feat_data, all_tag_data, region_type, y_normalization= False):
	 
	#scaler1 = preprocessing.MinMaxScaler(feature_range = (-1,1))
	scaler1 = preprocessing.MinMaxScaler(feature_range = (0,1))
	scaler1.fit(all_feat_data)
	train = scaler1.transform(train)
	test = scaler1.transform(test)
	val = scaler1.transform(val)  

	# train = numpy.exp(train)
	# test = numpy.exp(test)
	# val = numpy.exp(val)  

	if y_normalization:
		y_min = min(all_tag_data) -0.01 
		y_train = y_train - y_min 
		y_test = y_test - y_min
		y_val = y_val - y_min
		all_tag_data = all_tag_data - y_min 

	y_train = y_train.reshape(-1,1)
	y_test = y_test.reshape(-1,1)
	y_val = y_val.reshape(-1,1)
	all_tag_data = all_tag_data.reshape(-1,1)  

	return train, val, test, y_train, y_val, y_test


def select_features(data, include_feat_indeies, exclude_feat_indeies):
	new_data = []  
	column_num = len(data[0]) 
	for i in range(len(data)):
		s = []
		for j in range (column_num):
			if len(include_feat_indeies) > 0:
					if j not in include_feat_indeies: continue # nobagging
			else :
				if j in exclude_feat_indeies: continue # bagging
			s.append(data[i][j])
		
		new_data.append(s)  
	return new_data


def train_model(train, val, y_train,  y_val, train_type, model_type, randomstate): 
			if train_type == 'bagging':
				# model  = bagging_models.fit(train, val, y_train,  y_val, model_type,  randomstate)
				base_model = DecisionTreeRegressor(random_state=randomstate)  
				model = ensemble.BaggingRegressor(max_samples = 0.9, max_features = 1, warm_start = True, base_estimator = base_model, random_state = randomstate, n_estimators = 100, n_jobs = 50) 
				# model = ensemble.BaggingRegressor(max_samples = 0.85, max_features = 1, warm_start = True, base_estimator = base_model, random_state = randomstate, n_estimators = 100, n_jobs = 50)  
				model.fit(train, y_train)
				return model
			else: 
				if model_type == 'Linear':
					model = linear_model.LinearRegression(n_jobs = 3)
				elif model_type == 'SVR':
					model = svm.SVR(C=3.0, cache_size=50, degree=3, gamma='auto', kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
					# model = svm.SVR(C=3.0, cache_size=50, degree=2, gamma='auto', kernel='sigmoid', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
				elif model_type == 'DT':
					model = DecisionTreeRegressor(random_state=randomstate)  
				elif model_type == 'MLP':
					model = MLPRegressor(hidden_layer_sizes=4, activation='relu', random_state=randomstate)  
				elif model_type == 'poly':
					poly = preprocessing.PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
					train = poly.fit_transform(train)
					test = poly.fit_transform(test)
					val = poly.fit_transform(val)
					model = linear_model.LinearRegression(n_jobs = 3) 
				elif model_type == 'LinearGAM':
					model = LinearGAM() 
				elif model_type == 'GammaGAM':
					model = GammaGAM() 
				elif model_type == 'InvGaussGAM':
					model = InvGaussGAM() 
				elif model_type == 'LogisticGAM':
					model = LogisticGAM() 
				elif model_type == 'PoissonGAM':
					model = PoissonGAM() 
				elif model_type == 'ExpectileGAM':
					model = ExpectileGAM()  

				model.fit(train, y_train)
				return model

def train_n_cv_split_models(trian_10, test_10, val_10, y_train_10, y_test_10, y_val_10, region_type, tag_type, train_type, model_type, include_feat_indeies, exclude_feat_indeies):
	r2_sum_n = 0
	r2_average_n = 0 
	total_nomodel_folder_num = 0
	total_folder_num = 0 

	exp_model_list = []

	for china_train, china_test, china_val, china_y_train, china_y_test, china_y_val in zip(trian_10, test_10, val_10, y_train_10, y_test_10, y_val_10):
		r2_sum = 0
		r2_average = 0
		test_raw = []
		test_predict = []
		n_cv_split_models = []
		all_predict = []
		all_val = []
		for train, test, val, y_train, y_test, y_val in zip(china_train, china_test, china_val, china_y_train, china_y_test, china_y_val): 
			randomstate = 1111111
			total_folder_num += 1
			all_feat_data = train + val + test 
			all_tag_data = y_train + y_val + y_test   

			all_val.extend(y_val)
			# select features  
			train = select_features(train, include_feat_indeies, exclude_feat_indeies)
			val = select_features(val, include_feat_indeies, exclude_feat_indeies)
			test = select_features(test, include_feat_indeies, exclude_feat_indeies)
			all_feat_data = select_features(all_feat_data, include_feat_indeies, exclude_feat_indeies)
			train = numpy.array(train)
			test = numpy.array(test)
			val = numpy.array(val) 
			y_train = numpy.array(y_train)
			y_test = numpy.array(y_test)
			y_val = numpy.array(y_val)

			all_feat_data =  numpy.array(all_feat_data) 
			all_tag_data =  numpy.array(all_tag_data)  
			# print(train.shape, test.shape, val.shape, y_train.shape, y_test.shape, y_val.shape) 

			#normalization
			if model_type.endswith('GAM') :
				train, val, test, y_train, y_val, y_test = data_normalization(train, val, test, y_train, y_val, y_test, all_feat_data, all_tag_data, region_type, True)
			else:
				train, val, test, y_train, y_val, y_test = data_normalization(train, val, test, y_train, y_val, y_test, all_feat_data, all_tag_data, region_type)
				
			if train_type == 'bagging':
				# model_list, test = bagging_models.fit(train, test, val, y_train, y_test, y_val, randomstate)
				model = train_model(train, val, y_train,  y_val, train_type, model_type, randomstate)
				# predict = bagging_models.predict(model, val)
				predict = model.predict(val) 
				n_cv_split_models.append(model)
			else:
				model = train_model(train, val, y_train,  y_val, train_type, model_type, randomstate) 
				predict = model.predict(val)    
				n_cv_split_models.append(model)
			 
			all_predict.extend(predict)
			#r2_sum = r2_sum + float('%.2f' % r2_score(y_val, predict))  
			
		#r2_average = r2_sum/10
		#r2_sum_n = r2_sum_n + r2_average 
		r2_sum_n += float('%.2f' % r2_score(all_val, all_predict))  
	 
	r2_average_n = r2_sum_n/10 

	return r2_average_n, n_cv_split_models


def feat_increase_search(trian_10, test_10, val_10, y_train_10, y_test_10, y_val_10, region_type, tag_type, train_type, model_type, skip_feat_indeies, total_feat_num, max_feat_len ):

	max_feats = []
	max_r2 = -10000
	max_models = []
	for i in range(max_feat_len):   
		cur_max_r2 = max_r2
		cur_max_models = []
		cur_max_feats = []
		start_feats = [] 
		if len(max_feats) > 0: 
			start_feats = max_feats[-1].copy()
			cur_max_models = max_models[-1]

		print('cur iter FeatLen %.2f' % (len(start_feats) +1))
		for j in range(total_feat_num):
			cur_feats = start_feats.copy() 
			if j in skip_feat_indeies:
				continue
			if j in start_feats:
				continue
			
			cur_feats.append(j)
			include_feat_indeies = cur_feats
			exclude_feat_indeies = []
		
			r2_average, n_cv_split_models = train_n_cv_split_models(trian_10, test_10, val_10, y_train_10, y_test_10, y_val_10, region_type, tag_type, train_type, model_type, include_feat_indeies, exclude_feat_indeies)

			print('cur r2: %.2f cur max r2: %.2f' % (r2_average, cur_max_r2))
			if r2_average > cur_max_r2:
				cur_max_r2 = r2_average
				cur_max_models = n_cv_split_models
				cur_max_feats = cur_feats

		print('cur iter max r2: %.2f max r2: %.2f' % (cur_max_r2, max_r2))
		print('cur iter features ' + str(cur_max_feats))
		if (cur_max_r2 - max_r2) >= 0.0001: 
			max_r2 = cur_max_r2
			max_models.append(cur_max_models)
			max_feats.append(cur_max_feats) 
			
		else:
			break
	
	return max_r2, max_feats, max_models


def feat_bin_search(trian_10, test_10, val_10, y_train_10, y_test_10, y_val_10, region_type, tag_type, train_type, model_type, skip_feat_indeies, total_feat_num, bin_size):
	bin_feats_list = [[]]
	bin_feats_r2_list = [-10000]
	searched_feats_list =  {}
	max_feats = []
	max_r2 = -10000 
	max_models = []
	feats_num = 0
	while True:
		changed = 0    
		print('pre iter Feat List',  str(bin_feats_list))   
		for start_feats in bin_feats_list: 
			for j in range(total_feat_num):
				cur_feats = start_feats.copy() 
				# print(str(cur_feats))
				if j in skip_feat_indeies:
					continue
				if j in start_feats:
					continue
				
				cur_feats.append(j)
				# print(str(cur_feats))
				cur_feats.sort()
				str_cur_feats = [str(k) for k in cur_feats]
				s = '_'.join(str_cur_feats)
				if s in searched_feats_list: continue
				searched_feats_list[s]  = 1

				include_feat_indeies = cur_feats
				exclude_feat_indeies = [] 
				r2_average, n_cv_split_models = train_n_cv_split_models(trian_10, test_10, val_10, y_train_10, y_test_10, y_val_10, region_type, tag_type, train_type, model_type, include_feat_indeies, exclude_feat_indeies)
				
				if len(bin_feats_list) == 1 and bin_feats_r2_list[0] == -10000:
					bin_feats_list[0] = cur_feats
					bin_feats_r2_list[0] = r2_average
					changed = 1
				elif len(bin_feats_list) < bin_size:
					inserted = 0
					for k in range(len(bin_feats_list), 0, -1): 
						if bin_feats_r2_list[k-1] < r2_average:
							bin_feats_r2_list.insert(k, r2_average)
							bin_feats_list.insert(k, cur_feats)
							print(k)
							print(bin_feats_r2_list)
							print(bin_feats_list) 
							inserted = 1
							break
					if (inserted == 0):
						bin_feats_r2_list.insert(0, r2_average)
						bin_feats_list.insert(0, cur_feats)
						print(k)
						print(bin_feats_r2_list)
						print(bin_feats_list)
					changed = 1
				else:
					for k in range(len(bin_feats_list), 0, -1): 
						if bin_feats_r2_list[k-1] < r2_average:
							bin_feats_r2_list.insert(k, r2_average)
							bin_feats_list.insert(k, cur_feats)
							print(k)
							print(bin_feats_r2_list)
							print(bin_feats_list) 
							break
					if  len(bin_feats_list) > bin_size: 
						bin_feats_r2_list = bin_feats_r2_list[1:]
						bin_feats_list = bin_feats_list[1:]
						changed = 1
		if (changed == 0): break

	max_r2 = bin_feats_r2_list[-1]
	max_feats = bin_feats_list[-1]

	return max_r2, max_feats, max_models

def feat_decrease_search(trian_10, test_10, val_10, y_train_10, y_test_10, y_val_10, region_type, tag_type, train_type, model_type, skip_feat_indeies, total_feat_num, min_feat_len ): 
	max_feats = []
	max_r2 = -10000
	max_models = []
	iter_num = total_feat_num - min_feat_len

	cur_feats = []
	for j in range(total_feat_num):
		if j in skip_feat_indeies:
				continue 
		cur_feats.append(j) 
	include_feat_indeies = cur_feats
	exclude_feat_indeies = []
	r2_average, n_cv_split_models = train_n_cv_split_models(trian_10, test_10, val_10, y_train_10, y_test_10, y_val_10, region_type, tag_type, train_type, model_type, include_feat_indeies, exclude_feat_indeies)
	max_r2 = r2_average
	max_models.append(n_cv_split_models)
	max_feats.append(cur_feats) 


	for i in range(iter_num):   
		cur_max_r2 = max_r2
		cur_max_models = []
		cur_max_feats = []
		start_feats = [] 
		if len(max_feats) > 0: 
			start_feats = max_feats[-1].copy()
			cur_max_models = max_models[-1]

		if (len(start_feats) -1) < min_feat_len:
			break

		print('cur iter FeatLen %.2f' % (len(start_feats) -1))
		for j in range(total_feat_num):
			cur_feats = start_feats.copy() 
			if j in skip_feat_indeies:
				continue
			if j not in start_feats:
				continue
			
			cur_feats.remove(j)  
			include_feat_indeies = cur_feats
			exclude_feat_indeies = []
		
			r2_average, n_cv_split_models = train_n_cv_split_models(trian_10, test_10, val_10, y_train_10, y_test_10, y_val_10, region_type, tag_type, train_type, model_type, include_feat_indeies, exclude_feat_indeies)

			print('cur r2: %.2f cur max r2: %.2f' % (r2_average, cur_max_r2))
			if r2_average > cur_max_r2:
				cur_max_r2 = r2_average
				cur_max_models = n_cv_split_models
				cur_max_feats = cur_feats

		print('cur iter max r2: %.2f max r2: %.2f' % (cur_max_r2, max_r2))
		print('cur iter features ' + str(max_feats))
		if (cur_max_r2 - max_r2) >= 0.0001: 
			max_r2 = cur_max_r2
			max_models.append(cur_max_models)
			max_feats.append(cur_max_feats) 
			
		else:
			break
	
	return max_r2, max_feats, max_models

def all_feats_search(trian_10, test_10, val_10, y_train_10, y_test_10, y_val_10, region_type, tag_type, train_type, model_type, skip_feat_indeies, total_feat_num, max_feat_len ):
	all_feats_list = [[[]]]
	max_feats = []
	max_r2 = -10000
	max_models = []
	feats_num = 0
	for i in range(max_feat_len):   
		print('cur iter FeatLen %.2f' % (i +1))  
		pre_iter_feats_list = all_feats_list[-1] 
		# print('pre iter Feat List',  str(pre_iter_feats_list))  
		cur_iter_feats_list = []
		for start_feats in pre_iter_feats_list: 
			for j in range(total_feat_num):
				cur_feats = start_feats.copy() 
				# print(str(cur_feats))
				if j in skip_feat_indeies:
					continue
				if j in start_feats:
					continue
				
				cur_feats.append(j)
				# print(str(cur_feats))

				include_feat_indeies = cur_feats
				exclude_feat_indeies = [] 
				r2_average, n_cv_split_models = train_n_cv_split_models(trian_10, test_10, val_10, y_train_10, y_test_10, y_val_10, region_type, tag_type, train_type, model_type, include_feat_indeies, exclude_feat_indeies)

				# print('cur r2: %.2f cur max r2: %.2f' % (r2_average, max_r2))
				if (r2_average - max_r2) >= 0.0001: 
					max_r2 = r2_average 
					max_feats = cur_feats
					max_models = n_cv_split_models
				cur_iter_feats_list.append(cur_feats)
				# print(str(cur_iter_feats_list))
				feats_num += 1 

		print('cur iter max r2: %.2f ' % (max_r2))
		print('cur iter max features ' + str(max_feats))
			
		all_feats_list.append(cur_iter_feats_list)
		# print(str(all_feats_list))
	

	print('all feats num ' + str(feats_num))
	
	return max_r2, max_feats, max_models

def seach_features(region_type, tag_type, train_type, model_type,feat_search_method, exp_data_dir_path, model_file_path,skip_feat_indeies):   
	trian_10 = []
	test_10 = []
	val_10 = []
	y_train_10 = []
	y_test_10 = []
	y_val_10 = []  
	id_train_10 = []  
	id_test_10 = []  
	id_val_10 = []  

	total_feat_num = 0
	max_feat_len = 0

	for j in range(10):  
		if tag_type == 'CH4':
			china_id_train, china_train, china_y_train = load_CH4_CV_data(exp_data_dir_path, region_type + '_train_' + tag_type + '.txt', j, region_type )
			china_id_test, china_test, china_y_test = load_CH4_CV_data(exp_data_dir_path, region_type + '_test_' + tag_type + '.txt', j, region_type )
			china_id_val, china_val, china_y_val = load_CH4_CV_data(exp_data_dir_path, region_type + '_val_' + tag_type + '.txt', j, region_type ) 
			total_feat_num = len(china_train[0][0])
		else:
			china_id_train, china_train, china_y_train = load_CO2_CV_data(exp_data_dir_path, region_type + '_train_' + tag_type + '.txt', j, region_type )
			china_id_test, china_test, china_y_test = load_CO2_CV_data(exp_data_dir_path, region_type + '_test_' + tag_type + '.txt', j, region_type )
			china_id_val, china_val, china_y_val = load_CO2_CV_data(exp_data_dir_path, region_type + '_val_' + tag_type + '.txt', j, region_type ) 
			total_feat_num = len(china_train[0][0])

		trian_10.append(china_train)
		test_10.append(china_test)
		val_10.append(china_val)
		y_train_10.append(china_y_train)
		y_test_10.append(china_y_test)
		y_val_10.append(china_y_val)  
		id_train_10.append(china_id_train)
		id_test_10.append(china_id_test)
		id_val_10.append(china_id_val)  

	print(skip_feat_indeies)
	
	if feat_search_method == 'increase':
		max_feat_len = total_feat_num  - len(skip_feat_indeies)
		print('total feat #: %.2f max feat #: %.2f' % (total_feat_num, max_feat_len))
		max_r2, max_feats, max_models = feat_increase_search(trian_10, test_10, val_10, y_train_10, y_test_10, y_val_10, region_type, tag_type, train_type, model_type, skip_feat_indeies, total_feat_num, max_feat_len) 
	elif feat_search_method == 'decrease':
		min_feat_len = 1
		print('total feat #: %.2f min feat #: %.2f' % (total_feat_num, min_feat_len))
		max_r2, max_feats, max_models = feat_decrease_search(trian_10, test_10, val_10, y_train_10, y_test_10, y_val_10, region_type, tag_type, train_type, model_type, skip_feat_indeies, total_feat_num, min_feat_len) 
	elif feat_search_method == 'bin':
		bin_size = 20
		print('total feat #: %.2f bin feat #: %.2f' % (total_feat_num,bin_size))
		max_r2, max_feats, max_models = feat_bin_search(trian_10, test_10, val_10, y_train_10, y_test_10, y_val_10, region_type, tag_type, train_type, model_type, skip_feat_indeies, total_feat_num, bin_size) 
	else:
		max_feat_len = total_feat_num  - len(skip_feat_indeies)
		max_r2, max_feats, max_models = all_feats_search(trian_10, test_10, val_10, y_train_10, y_test_10, y_val_10, region_type, tag_type, train_type, model_type, skip_feat_indeies, total_feat_num, max_feat_len) 

	# print('Final max r2: %.2f' % (max_r2))
	# print('Final features ' + str(max_feats))
	return max_feats, max_r2

def train_models_for_fixed_feats(trian_10, test_10, val_10, y_train_10, y_test_10, y_val_10, id_train_10, id_test_10, id_val_10, region_type, tag_type, train_type, model_type, include_feat_indeies, exclude_feat_indeies, result_r2_10_file_path, result_r2_100_file_path, result_tag_file_path, test_data_type):   
	r2_sum_10 = 0
	r2_average_10 = 0  
	total_nomodel_folder_num = 0
	total_folder_num = 0
	r2_10_writer = open(result_r2_10_file_path, 'w')
	r2_100_writer = open(result_r2_100_file_path, 'w')
	tag_writer = open(result_tag_file_path, 'w')
	for china_train, china_test, china_val, china_y_train, china_y_test, china_y_val, china_id_train, china_id_test, china_id_val in zip(trian_10, test_10, val_10, y_train_10, y_test_10, y_val_10, id_train_10, id_test_10, id_val_10):
		r2_sum = 0
		r2_average = 0 
		cv_y_test = []
		cv_predict = []
		for train, test, val, y_train, y_test, y_val, id_train, id_test, id_val in zip(china_train, china_test, china_val, china_y_train, china_y_test, china_y_val, china_id_train, china_id_test, china_id_val):
			randomstate = 1111111
			total_folder_num += 1
			all_feat_data = train + val + test 
			all_tag_data = y_train + y_val + y_test 
			
			# y_test = y_val
			# test = val
			# if region_type == 'DW' and tag_type == 'CH4':
			# 	y_test = BHB_DW_CH4_Y_test
			# 	test = BHB_DW_CH4_X_test
			# elif region_type == 'DW' and tag_type == 'CO2':
			# 	y_test = BHB_DW_CO2_Y_test
			# 	test = BHB_DW_CO2_X_test
			# elif region_type == 'LIT' and tag_type == 'CH4':
			# 	y_test = BHB_LIT_CH4_Y_test
			# 	test = BHB_LIT_CH4_X_test
			# elif region_type == 'LIT' and tag_type == 'CO2':
			# 	y_test = BHB_LIT_CO2_Y_test
			# 	test = BHB_LIT_CO2_X_test

			# if region_type == 'DW' and tag_type == 'CH4':
			# 	y_test = YDS_DW_CH4_Y_test
			# 	test = YDS_DW_CH4_X_test
			# elif region_type == 'DW' and tag_type == 'CO2':
			# 	y_test = YDS_DW_CO2_Y_test
			# 	test = YDS_DW_CO2_X_test
			# elif region_type == 'LIT' and tag_type == 'CH4':
			# 	y_test = YDS_LIT_CH4_Y_test
			# 	test = YDS_LIT_CH4_X_test
			# elif region_type == 'LIT' and tag_type == 'CO2':
			# 	y_test = YDS_LIT_CO2_Y_test
			# 	test = YDS_LIT_CO2_X_test 

			# select features  
			train = select_features(train, include_feat_indeies, exclude_feat_indeies)
			val = select_features(val, include_feat_indeies, exclude_feat_indeies)
			test = select_features(test, include_feat_indeies, exclude_feat_indeies)
			all_feat_data = select_features(all_feat_data, include_feat_indeies, exclude_feat_indeies)
			train = numpy.array(train)
			test = numpy.array(test)
			val = numpy.array(val) 
			y_train = numpy.array(y_train)
			y_test = numpy.array(y_test)
			y_val = numpy.array(y_val)

			all_feat_data =  numpy.array(all_feat_data) 
			all_tag_data =  numpy.array(all_tag_data)  
			# print(train.shape, test.shape, val.shape, y_train.shape, y_test.shape, y_val.shape) 

			#normalization
			if model_type.endswith('GAM') :
				train, val, test, y_train, y_val, y_test = data_normalization(train, val, test, y_train, y_val, y_test, all_feat_data, all_tag_data, region_type, True)
			else:
				train, val, test, y_train, y_val, y_test = data_normalization(train, val, test, y_train, y_val, y_test, all_feat_data, all_tag_data, region_type)
			
			
			if train_type == 'bagging':
				# model_list, test = bagging_models.fit(train, test, val, y_train, y_test, y_val, randomstate)
				# model = ensemble.BaggingRegressor(max_samples = 1.0, max_features = 0.8, warm_start = True, base_estimator = model, random_state = randomstate, n_estimators = 100, n_jobs = 50) 
				model = train_model(train, val, y_train,  y_val, train_type, model_type, randomstate)
				# predict = bagging_models.predict(model, test)  
				if test_data_type == 'training_data':
					test = train
					y_test = y_train
					id_test = id_train
				elif test_data_type == 'val_data':
					test = val
					y_test = y_val
					id_test = id_val
				predict = model.predict(test) 
			else: 
				model = train_model(train, val, y_train,  y_val, train_type, model_type, randomstate)  
				if test_data_type == 'training_data':
					test = train
					y_test = y_train
					id_test = id_train
				elif test_data_type == 'val_data':
					test = val
					y_test = y_val
					id_test = id_val
				predict = model.predict(test)      


			# if train_type == 'bagging':
			# 	model_list  = bagging_models.fit(train, val, y_train,  y_val, model_type, randomstate)
			# 	# model = ensemble.BaggingRegressor(max_samples = 1.0, max_features = 0.8, warm_start = True, base_estimator = model, random_state = randomstate, n_estimators = 100, n_jobs = 50) 
			# 	predict = bagging_models.predict(model_list, test, y_test)
			# else:
			# 	# poly = preprocessing.PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
			# 	# train = poly.fit_transform(train)
			# 	# test = poly.fit_transform(test)
			# 	# val = poly.fit_transform(val)
			# 	# model = linear_model.LinearRegression(n_jobs = 3) 

			# 	if model_type == 'Linear':
			# 		model = linear_model.LinearRegression(n_jobs = 3)
			# 	elif model_type == 'SVR':
			# 		model = svm.SVR(C=3.0, cache_size=50, degree=3, gamma='auto', kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
			# 		# model = svm.SVR(C=3.0, cache_size=50, degree=2, gamma='auto', kernel='sigmoid', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
			# 	elif model_type == 'DT':
			# 		model = DecisionTreeRegressor(random_state=randomstate)  
			# 	elif model_type == 'MLP':
			# 		model = MLPRegressor(random_state=randomstate)  

			# 	model.fit(train, y_train)
			# 	predict = model.predict(test)   
				# print(model.coef_, model.intercept_)

			# print(len(predict), len(y_val))
			# for p_y, g_y in zip(predict, y_test):
			# 	print(p_y, g_y)  

			cv_y_test.extend(y_test)
			cv_predict.extend(predict)
			print('%.2f' % r2_score(y_test, predict) )
			try: 
				r2_average = float('%.2f' % r2_score(y_test, predict))
				r2_100_writer.write('%.2f' % r2_average + '\n')
				r2_sum = r2_sum + r2_average 
				
				for i in range(len(id_test)): 
					id = id_test[i]
					g_tag = y_test[i][0]
					if model_type.endswith('Linear') :
						p_tag = predict[i][0]
					else:
						p_tag = predict[i]
					# print(str(id) + '\t' + str(g_tag)  + '\t' + str(p_tag))
					tag_writer.write(str(id) + '\t' + str(g_tag)  + '\t' + str(p_tag)  + '\n')
				tag_writer.write('\n')
			
			except Exception as e:
				print((e))
				total_nomodel_folder_num += 1
		
		# r2_average = r2_sum/10
		r2_average = float('%.2f' % r2_score(cv_y_test, cv_predict)) 
		r2_10_writer.write('%.2f' % r2_average + '\n')
		r2_sum_10 = r2_sum_10 + r2_average
		print(r2_average , r2_sum_10)
	r2_average_10 = r2_sum_10/10 
	
	r2_10_writer.flush()
	r2_10_writer.close()
	r2_100_writer.flush()
	r2_100_writer.close()
	tag_writer.flush()
	tag_writer.close()
	print(tag_type , total_nomodel_folder_num, total_folder_num) 
	return r2_average_10


def training_test_for_fixed_feats(region_type, tag_type, train_type, model_type, include_feat_indeies, exclude_feat_indeies, exp_data_dir_path, result_r2_10_file_path, result_r2_100_file_path, result_tag_file_path, test_data_type):   
	trian_10 = []
	test_10 = []
	val_10 = []
	y_train_10 = []
	y_test_10 = []
	y_val_10 = [] 
	id_train_10 = []
	id_test_10 = []
	id_val_10 = [] 

	for j in range(10):  
		if tag_type == 'CH4':
			china_id_train, china_train, china_y_train = load_CH4_CV_data(exp_data_dir_path, region_type + '_train_' + tag_type + '.txt', j, region_type )
			china_id_test, china_test, china_y_test = load_CH4_CV_data(exp_data_dir_path, region_type + '_test_' + tag_type + '.txt', j, region_type )
			china_id_val, china_val, china_y_val = load_CH4_CV_data(exp_data_dir_path, region_type + '_val_' + tag_type + '.txt', j, region_type ) 
		else:
			china_id_train, china_train, china_y_train = load_CO2_CV_data(exp_data_dir_path, region_type + '_train_' + tag_type + '.txt', j, region_type )
			china_id_test, china_test, china_y_test = load_CO2_CV_data(exp_data_dir_path, region_type + '_test_' + tag_type + '.txt', j, region_type )
			china_id_val, china_val, china_y_val = load_CO2_CV_data(exp_data_dir_path, region_type + '_val_' + tag_type + '.txt', j, region_type ) 


		trian_10.append(china_train)
		test_10.append(china_test)
		val_10.append(china_val)
		y_train_10.append(china_y_train)
		y_test_10.append(china_y_test)
		y_val_10.append(china_y_val)  
		id_train_10.append(china_id_train)
		id_test_10.append(china_id_test)
		id_val_10.append(china_id_val)  

	r2_average = train_models_for_fixed_feats(trian_10, test_10, val_10, y_train_10, y_test_10, y_val_10, id_train_10, id_test_10, id_val_10, region_type, tag_type, train_type, model_type, include_feat_indeies, exclude_feat_indeies, result_r2_10_file_path, result_r2_100_file_path, result_tag_file_path, test_data_type)
	return r2_average

def duplicate_test_data(id_list, feat_list, tag_list):
		cr_id_list = []
		cr_feat_list = []
		cr_tag_list = []
		for i in range(10): 
			id_list = copy.deepcopy(id_list)
			feat_list = copy.deepcopy(feat_list)
			tag_list = copy.deepcopy(tag_list)
			cr_id_list.append(id_list)
			cr_feat_list.append(feat_list)
			cr_tag_list.append(tag_list)

		return cr_id_list, cr_feat_list, cr_tag_list

def training_test_for_fixed_feats_fixed_testdata(region_type, tag_type, train_type, model_type, include_feat_indeies, exclude_feat_indeies, traing_exp_data_dir_path, test_exp_data_dir_path,  result_r2_10_file_path, result_r2_100_file_path, result_tag_file_path, test_data_type):   
	trian_10 = []
	test_10 = []
	val_10 = []
	y_train_10 = []
	y_test_10 = []
	y_val_10 = [] 
	id_train_10 = []
	id_test_10 = []
	id_val_10 = [] 

	for j in range(10):  
		if tag_type == 'CH4':
			china_id_train, china_train, china_y_train = load_CH4_CV_data(traing_exp_data_dir_path, region_type + '_train_' + tag_type + '.txt', j, region_type )
			china_id_test, china_test, china_y_test = load_CH4_data(test_exp_data_dir_path, region_type + '_test_' + tag_type + '.txt', region_type )
			china_id_test, china_test, china_y_test = duplicate_test_data(china_id_test[0], china_test[0], china_y_test[0])
			china_id_val, china_val, china_y_val = load_CH4_CV_data(exp_data_dir_path, region_type + '_val_' + tag_type + '.txt', j, region_type ) 
		else:
			china_id_train, china_train, china_y_train = load_CO2_CV_data(traing_exp_data_dir_path, region_type + '_train_' + tag_type + '.txt', j, region_type )
			china_id_test, china_test, china_y_test = load_CO2_data(test_exp_data_dir_path, region_type + '_test_' + tag_type + '.txt',  region_type )
			china_id_test, china_test, china_y_test = duplicate_test_data(china_id_test[0], china_test[0], china_y_test[0])
			china_id_val, china_val, china_y_val = load_CO2_CV_data(exp_data_dir_path, region_type + '_val_' + tag_type + '.txt', j, region_type ) 
			


		trian_10.append(china_train)
		test_10.append(china_test)
		val_10.append(china_val)
		y_train_10.append(china_y_train)
		y_test_10.append(china_y_test)
		y_val_10.append(china_y_val)  
		id_train_10.append(china_id_train)
		id_test_10.append(china_id_test)
		id_val_10.append(china_id_val)  

	r2_average = train_models_for_fixed_feats(trian_10, test_10, val_10, y_train_10, y_test_10, y_val_10, id_train_10, id_test_10, id_val_10, region_type, tag_type, train_type, model_type, include_feat_indeies, exclude_feat_indeies, result_r2_10_file_path, result_r2_100_file_path, result_tag_file_path, test_data_type)
	return r2_average

def calculate_performances_one_reservoirs(region_type, all_result_tag_file_path, result_r2_10_file_path, result_r2_100_file_path,  result_tag_file_path,  beg_index, end_index):    
	r = open(all_result_tag_file_path, 'r')
	r2_10_writer = open(result_r2_10_file_path, 'w')
	r2_100_writer = open(result_r2_100_file_path, 'w')
	tag_writer = open(result_tag_file_path, 'w') 
	 
	g_tag_list = []
	p_tag_list = []  
	num = 0
	real_num = 0
	r2_sum_10 = 0 
	r2_sum = 0
	lines_train = r.readlines() 
	for line in lines_train:
			if line != '\n':
				line = line.strip() 
				strs = line.split('\t')  
				id = int(float(strs[0]))
				# if (id >= end_index):
				# 	print(line)
				if id >= beg_index and id < end_index:   
				# if True:
					g_tag_list.append(float(strs[1]))
					p_tag_list.append(float(strs[2]))
					tag_writer.write(line +'\n')
			else: 
				tag_writer.write('\n')
				if len(g_tag_list) > 0:  
					# print(g_tag_list)
					# print(p_tag_list)
					r2_aver =  r2_score(g_tag_list, p_tag_list) 
					print('one %.2f' % r2_aver)
					try: 
						r2_100_writer.write('%.2f' % r2_aver + '\n')  
					except Exception as e:
						print((e))  

					real_num += 1
					r2_sum += r2_aver 
				
				# g_tag_list = []
				# p_tag_list = []  

				num += 1
				if num == 10:     
					# aver = r2_sum/real_num 
					r2_aver = r2_score(g_tag_list, p_tag_list) 
					r2_10_writer.write('%.2f' % r2_aver + '\n')  
					r2_sum_10 = r2_sum_10 + r2_aver
					print(r2_sum, real_num, r2_sum/real_num,r2_sum_10 )
					r2_sum = 0 
					num = 0
					real_num = 0
	r2_average_10 = r2_sum_10/10 
	r2_10_writer.flush()
	r2_10_writer.close()
	r2_100_writer.flush()
	r2_100_writer.close()
	tag_writer.flush()
	tag_writer.close() 
	return r2_average_10

def get_output_filepathes(result_dir, training_data_name,  test_data_name, region_type,  tag_type, train_type, model_type):
	result_r2_10_file_path = result_dir + '/' + training_data_name + '_2_' + test_data_name + '_' + region_type + '_' + tag_type  +'_'+ train_type  +'_'+ model_type +  '_r2_10.txt' 
	result_r2_100_file_path = result_dir + '/' + training_data_name + '_2_' + test_data_name + '_' + region_type + '_' + tag_type  +'_'+ train_type  +'_'+ model_type +  '_r2_100.txt' 
	result_tag_file_path = result_dir + '/' + training_data_name + '_2_' + test_data_name + '_' + region_type + '_' + tag_type  +'_'+ train_type  +'_'+ model_type +  '_tag.txt'  
	
	return result_tag_file_path, result_r2_10_file_path, result_r2_100_file_path


if __name__ == '__main__': 
	train_type='nobagging'   # nobagging bagging
	region_type = 'DW' # DW LIT
	model_type = 'DT' # Linear  SVR  DT  MLP
	tag_type= 'C02'  # CO2  CH4 N2O 
	bank_name='MY' # MY,  all
	feat_search_method = 'increase' # increase decrease all bin
	if region_type == 'DW':
		if bank_name == 'all':
			if tag_type == 'N2O':
				exp_data_dir_path = '/home/cxy/zym/exp_data/beijing/N2O/'
			else:
				exp_data_dir_path = '/home/cxy/zym/exp_data/beijing/DW/' 
		else:
			if tag_type == 'N2O':
				exp_data_dir_path = '/home/cxy/zym/exp_data/beijing_others/MY/N2O/'
				bhb_exp_data_dir_path = '/home/cxy/zym/exp_data/beijing_others/BHB/N2O/' 
				yds_exp_data_dir_path = '/home/cxy/zym/exp_data/beijing_others/YDS/N2O/' 
			else:
				exp_data_dir_path = '/home/cxy/zym/exp_data/beijing_others/MY/DW/' 
				bhb_exp_data_dir_path = '/home/cxy/zym/exp_data/beijing_others/BHB/DW/' 
				yds_exp_data_dir_path = '/home/cxy/zym/exp_data/beijing_others/YDS/DW/' 
			
		
	elif region_type == 'LIT':

		if bank_name == 'all':
			if tag_type == 'N2O':
				exp_data_dir_path = '/home/cxy/zym/exp_data/beijing/N2O/'
			else:
				exp_data_dir_path = '/home/cxy/zym/exp_data/beijing/LIT/'
		else:
			if tag_type == 'N2O':
				exp_data_dir_path = '/home/cxy/zym/exp_data/beijing_others/MY/N2O/'
				bhb_exp_data_dir_path = '/home/cxy/zym/exp_data/beijing_others/BHB/N2O/' 
				yds_exp_data_dir_path = '/home/cxy/zym/exp_data/beijing_others/YDS/N2O/' 
			else:
				exp_data_dir_path = '/home/cxy/zym/exp_data/beijing_others/MY/LIT/'	
				bhb_exp_data_dir_path = '/home/cxy/zym/exp_data/beijing_others/BHB/LIT/' 
				yds_exp_data_dir_path = '/home/cxy/zym/exp_data/beijing_others/YDS/LIT/' 
		
	

	skip_feat_indeies = []
	if region_type == 'DW':
			skip_feat_indeies = [6, 7, 8, 9]
			# skip_feat_indeies = []

	include_feat_indeies = []
	exclude_feat_indeies = []

	
	if bank_name == 'all':
		model_file_path = '/home/cxy/zym/result/beijing_val_CO2_'+region_type +'_'+ train_type + '.txt' 
		training_data_name = 'beijing'
	else:
		model_file_path = '/home/cxy/zym/result/beijing_my_val_CO2_'+region_type +'_'+ train_type + '.txt' 
		training_data_name = 'beijing_MY'
	max_feats, max_r2 = seach_features(region_type, tag_type, train_type, model_type,feat_search_method, exp_data_dir_path, model_file_path,skip_feat_indeies)

 

	
	include_feat_indeies = max_feats[-1] 
	# include_feat_indeies = [4, 10, 5, 11]

	result_dir = '/home/cxy/zym/result'

	test_data_type = 'training_data'
	if bank_name == 'all':
		test_data_name = 'beijing_training'
	else:
		test_data_name = 'beijing_MY_training'  
	result_tag_file_path, result_r2_10_file_path, result_r2_100_file_path = get_output_filepathes(result_dir, training_data_name,  test_data_name, region_type,  tag_type, train_type, model_type)
	training_r2_average = training_test_for_fixed_feats(region_type, tag_type, train_type, model_type, include_feat_indeies, exclude_feat_indeies, exp_data_dir_path, result_r2_10_file_path, result_r2_100_file_path, result_tag_file_path, test_data_type)

	test_data_type = 'val_data'
	if bank_name == 'all':
		test_data_name = 'beijing_val'
	else:
		test_data_name = 'beijing_MY_val'
	result_tag_file_path, result_r2_10_file_path, result_r2_100_file_path = get_output_filepathes(result_dir, training_data_name,  test_data_name, region_type,  tag_type, train_type, model_type)
	val_r2_average = training_test_for_fixed_feats(region_type, tag_type, train_type, model_type, include_feat_indeies, exclude_feat_indeies, exp_data_dir_path, result_r2_10_file_path, result_r2_100_file_path, result_tag_file_path, test_data_type)

	test_data_type = 'test_data'
	if bank_name == 'all':
		test_data_name = 'beijing_test'
	else:
		test_data_name = 'beijing_MY_test'
	result_tag_file_path, result_r2_10_file_path, result_r2_100_file_path = get_output_filepathes(result_dir, training_data_name,  test_data_name, region_type,  tag_type, train_type, model_type)
	test_r2_average = training_test_for_fixed_feats(region_type, tag_type, train_type, model_type, include_feat_indeies, exclude_feat_indeies, exp_data_dir_path, result_r2_10_file_path, result_r2_100_file_path, result_tag_file_path, test_data_type)



	test_data_type = 'test_data'
	if bank_name == 'MY': 
		test_data_name = 'beijing_BHB'
		result_tag_file_path, result_r2_10_file_path, result_r2_100_file_path = get_output_filepathes(result_dir, training_data_name,  test_data_name, region_type,  tag_type, train_type, model_type)
		bhb_test_r2_average = training_test_for_fixed_feats_fixed_testdata(region_type, tag_type, train_type, model_type, include_feat_indeies, exclude_feat_indeies, exp_data_dir_path, bhb_exp_data_dir_path, result_r2_10_file_path, result_r2_100_file_path, result_tag_file_path, test_data_type)
		

		test_data_name = 'beijing_YDS'
		result_tag_file_path, result_r2_10_file_path, result_r2_100_file_path = get_output_filepathes(result_dir, training_data_name,  test_data_name, region_type,  tag_type, train_type, model_type)
		yds_test_r2_average = training_test_for_fixed_feats_fixed_testdata(region_type, tag_type, train_type, model_type, include_feat_indeies, exclude_feat_indeies, exp_data_dir_path, yds_exp_data_dir_path, result_r2_10_file_path, result_r2_100_file_path, result_tag_file_path, test_data_type)
		
	elif bank_name == 'all': 
		test_data_name = 'beijing_test'
		all_result_tag_file_path = result_dir + '/' + training_data_name + '_2_' + test_data_name + '_' + region_type + '_' + tag_type  +'_'+ train_type  +'_'+ model_type +  '_tag.txt'  

		if region_type == 'LIT':
			test_data_name = 'beijing_MY'
			result_tag_file_path, result_r2_10_file_path, result_r2_100_file_path = get_output_filepathes(result_dir, training_data_name,  test_data_name, region_type,  tag_type, train_type, model_type)
			my_test_r2_average = calculate_performances_one_reservoirs(region_type, all_result_tag_file_path,  result_r2_10_file_path, result_r2_100_file_path,  result_tag_file_path, 1, 355)

			test_data_name = 'beijing_BHB'
			result_tag_file_path, result_r2_10_file_path, result_r2_100_file_path = get_output_filepathes(result_dir, training_data_name,  test_data_name, region_type,  tag_type, train_type, model_type)
			bhb_test_r2_average = calculate_performances_one_reservoirs(region_type, all_result_tag_file_path,  result_r2_10_file_path, result_r2_100_file_path,  result_tag_file_path, 391, 409)

			test_data_name = 'beijing_YDS'
			result_tag_file_path, result_r2_10_file_path, result_r2_100_file_path = get_output_filepathes(result_dir, training_data_name,  test_data_name, region_type,  tag_type, train_type, model_type)
			yds_test_r2_average = calculate_performances_one_reservoirs(region_type, all_result_tag_file_path,  result_r2_10_file_path, result_r2_100_file_path,  result_tag_file_path, 364, 382) 
				 
		else:
			test_data_name = 'beijing_MY'
			result_tag_file_path, result_r2_10_file_path, result_r2_100_file_path = get_output_filepathes(result_dir, training_data_name,  test_data_name, region_type,  tag_type, train_type, model_type)
			my_test_r2_average = calculate_performances_one_reservoirs(region_type, all_result_tag_file_path,  result_r2_10_file_path, result_r2_100_file_path,  result_tag_file_path, 1, 175)
			print(my_test_r2_average) 

			test_data_name = 'beijing_BHB'
			result_tag_file_path, result_r2_10_file_path, result_r2_100_file_path = get_output_filepathes(result_dir, training_data_name,  test_data_name, region_type,  tag_type, train_type, model_type)
			bhb_test_r2_average = calculate_performances_one_reservoirs(region_type, all_result_tag_file_path,  result_r2_10_file_path, result_r2_100_file_path,  result_tag_file_path, 382, 391)

			test_data_name = 'beijing_YDS'
			result_tag_file_path, result_r2_10_file_path, result_r2_100_file_path = get_output_filepathes(result_dir, training_data_name,  test_data_name, region_type,  tag_type, train_type, model_type)
			yds_test_r2_average = calculate_performances_one_reservoirs(region_type, all_result_tag_file_path,  result_r2_10_file_path, result_r2_100_file_path,  result_tag_file_path, 355, 364) 



	print(region_type + '_' + tag_type  +'_'+ train_type  +'_'+ model_type)
	print('Final features ' + str(max_feats)) 
	print('Val r2: %.2f' % (max_r2)) 
	print('Training r2: %.2f' % training_r2_average)  
	print('Val r2: %.2f' % val_r2_average)  
	print('Test r2: %.2f' % test_r2_average) 

	if bank_name == 'MY':
		print('BHB Test r2: %.2f' % bhb_test_r2_average) 
		print('YDS Test r2: %.2f' % yds_test_r2_average) 
	elif bank_name == 'all':
		print('MY Test r2: %.2f' % my_test_r2_average) 
		print('BHB Test r2: %.2f' % bhb_test_r2_average) 
		print('YDS Test r2: %.2f' % yds_test_r2_average) 
