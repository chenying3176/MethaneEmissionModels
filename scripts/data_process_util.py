#coding:utf-8
import os
import random 
from sklearn.model_selection import cross_val_score
import numpy
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from n_cross_validation import NCrossValidation


def get_n_cv_split(X_data,Y_data, exp_data_dir_path, filename_head,filename_tail, exp_n):
	randomstate=1111111

	for j in range(exp_n):    
		random.seed(randomstate)
		numpy.random.seed(randomstate) 
		ncv = NCrossValidation(n_splits=10,shuffle=True, random_state=randomstate)  
		cv_data_list =  ncv.split(X_data,Y_data)
		randomstate+=5

		for i in range(len(cv_data_list)): 
			[X_train ,X_val , X_test, y_train ,y_val, y_test] = cv_data_list[i]

			y_train = numpy.array(y_train).tolist()
			y_val = numpy.array(y_val).tolist()
			y_test = numpy.array(y_test).tolist()

			if not os.path.exists(exp_data_dir_path + '/' + str(j)):
			    os.makedirs(exp_data_dir_path + '/' + str(j))
			if not os.path.exists(exp_data_dir_path + '/' + str(j) + '/' + str(i)):
			    os.makedirs(exp_data_dir_path + '/' + str(j) + '/' + str(i))

			wtrain = open(exp_data_dir_path + '/' + str(j) + '/' + str(i) + '/' + filename_head + '_train_'+ filename_tail + '.txt', 'w')
			for x,y in zip(X_train, y_train):
			    for yy in y:
				    wtrain.write( str(yy) + '\t')
			    x = numpy.array(x).tolist()
			    for xx in x:
				    wtrain.write(str(xx) + '\t')
			    wtrain.write('\n')
			wtest = open(exp_data_dir_path + '/' + str(j) + '/' + str(i)+ '/' + filename_head + '_test_'+ filename_tail + '.txt', 'w')
			for x,y in zip(X_test, y_test):
			    for yy in y:
				    wtest.write( str(yy) + '\t')
			    x = numpy.array(x).tolist()
			    for xx in x:
				    wtest.write(str(xx) + '\t')
			    wtest.write('\n')
			wval = open(exp_data_dir_path + '/'+ str(j) + '/' + str(i) + '/' + filename_head + '_val_'+ filename_tail + '.txt', 'w')
			for x,y in zip(X_val, y_val):
			    for yy in y:
				    wval.write( str(yy) + '\t')
			    x = numpy.array(x).tolist()
			    for xx in x:
				    wval.write(str(xx) + '\t')
			    wval.write('\n')
			wtrain.flush()
			wtrain.close()
			wtest.flush()
			wtest.close()
			wval.flush()
			wval.close() 


def output_data(X_CH4, Y_CH4, output_file_path):
			wtest = open(output_file_path, 'w')
			for x,y in zip(X_CH4, Y_CH4):
				for yy in y:
					wtest.write( str(yy) + '\t')
				x = numpy.array(x).tolist()
				for xx in x:
					wtest.write(str(xx) + '\t')
				wtest.write('\n')
			wtest.flush()
			wtest.close()        	
            
def insert_missing_feat(X_data):
    column_num = len(X_data[0])
    for j in range(0,column_num):
	    sum = 0
	    num = 0
	    print( len(X_data))
	    for i in range(len(X_data)):
			# print( i, j, X_data[i][j])
		    if X_data[i][j] != 'null':
			    sum += X_data[i][j]
			    num += 1
	    print(j, num, sum)
	    if num >0:
		    average = sum/num
	    else:
		    average = 0.0
		

	    for i in range(len(X_data)):
		    if X_data[i][j] == 'null':
		        X_data[i][j] = average 

    return X_data

def delete_missing_feat_instances(X_data,Y_data,):
    new_X_data = []
    new_Y_data = []
     
    for i in range(len(X_data)):
        find = 0
        for j in range(len(X_data[i])):
            if X_data[i][j] == 'null': 
                find = 1
                break
        if find == 0:
            new_X_data.append(X_data[i])
            new_Y_data.append(Y_data[i])

    return new_X_data, new_Y_data
