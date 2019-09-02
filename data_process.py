#coding:utf-8
import os
import xlrd
import random
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
import numpy 
from sklearn import preprocessing
from  data_process_util import get_n_cv_split, insert_missing_feat,delete_missing_feat_instances,output_data

excel = '/exp_data/000.Res_GHGs_yc.xlsx'

bank_name='MY' # MY, BHB, YDS, BHB_YDS, all
if bank_name == 'all':
	exp_data_dir_path = '/home/yc/Code/projects/SoilCarbonAnalysis/exp_data/beijing/' 
elif bank_name == 'MY':
	exp_data_dir_path = '/home/yc/Code/projects/SoilCarbonAnalysis/exp_data/beijing_others/MY/' 
elif bank_name == 'BHB':
	exp_data_dir_path = '/home/yc/Code/projects/SoilCarbonAnalysis/exp_data/beijing_others/BHB/' 
elif bank_name == 'YDS':
	exp_data_dir_path = '/home/yc/Code/projects/SoilCarbonAnalysis/exp_data/beijing_others/YDS/' 
elif bank_name == 'BHB_YDS':
	exp_data_dir_path = '/home/yc/Code/projects/SoilCarbonAnalysis/exp_data/beijing_others/BHB_YDS/' 


def load_DW_CH4_subdata(DW_CH4, beg_index, end_index):
	X_CH4 = []
	Y_CH4 = []
	row_num = DW_CH4.nrows 
	print("DW_CH4.nrows ", DW_CH4.nrows )
	for i in range(beg_index, end_index):
		try:
			hang = []
			idi = DW_CH4.cell(i,0).value
			hang.append(int(idi))
			yv = DW_CH4.cell(i,8).value
			# print(idi)
			hang.append(float(yv))
			Y_CH4.append(hang)

			x_ = []
			for j in range(9,22):
				v = DW_CH4.cell(i,j).value
				try:
					x_.append(float(v))
				except Exception as e:
					x_.append('null')
			# x_ = numpy.array(x_)
			# x_ = x_.reshape(1,31,1)
			X_CH4.append(x_)
			 
		except Exception as e:
			print(e)
			print(i)
			continue

	column_num = len(X_CH4[0])
	row_num = len(X_CH4)
	print(row_num, column_num)
	X_CH4 = insert_missing_feat(X_CH4)

	return X_CH4, Y_CH4
	

def load_DW_CO2_subdata(DW_CO2, beg_index, end_index):
	X_CO2 = []
	Y_CO2 = []
	row_num = DW_CO2.nrows 
	print("DW_CO2.nrows ", DW_CO2.nrows )
	for i in range(beg_index, end_index):
		try:
			hang = []
			idi = DW_CO2.cell(i,0).value
			hang.append(int(idi))
			yv = DW_CO2.cell(i,8).value
			# print(yv)
			hang.append(float(yv))
			Y_CO2.append(hang)

			x_ = []
			for j in range(9,22):
				v = DW_CO2.cell(i,j).value
				try:
					x_.append(float(v))
				except Exception as e:
					x_.append('null')
			# x_ = numpy.array(x_)
			# x_ = x_.reshape(1,31,1)
			X_CO2.append(x_)
			 
		except Exception as e:
			print(e)
			continue

	column_num = len(X_CO2[0])
	row_num = len(X_CO2)
	print(row_num, column_num)
	X_CO2 = insert_missing_feat(X_CO2)

	return X_CO2, Y_CO2

def load_DW_data(DW_CH4, DW_CO2): 
	X_CH4 = []
	Y_CH4 = []
	row_num = DW_CH4.nrows 

	print("load data") 
	X_CH4_MY, Y_CH4_MY = load_DW_CH4_subdata(DW_CH4, 1, 116)
	X_CO2_MY, Y_CO2_MY = load_DW_CO2_subdata(DW_CO2, 1, 116)

	X_CH4_YDS, Y_CH4_YDS = load_DW_CH4_subdata(DW_CH4, 116, 125)
	X_CO2_YDS, Y_CO2_YDS = load_DW_CO2_subdata(DW_CO2,116, 125)

	X_CH4_BHB, Y_CH4_BHB = load_DW_CH4_subdata(DW_CH4, 125, 133)
	X_CO2_BHB, Y_CO2_BHB = load_DW_CO2_subdata(DW_CO2,125, 133)

	print("merge data")
	#CH4
	if bank_name == 'all':
		X_CH4 = X_CH4_MY + X_CH4_YDS + X_CH4_BHB
		Y_CH4 = Y_CH4_MY + Y_CH4_YDS + Y_CH4_BHB 
	elif bank_name == 'MY':
		X_CH4 = X_CH4_MY 
		Y_CH4 = Y_CH4_MY 
	elif bank_name == 'BHB':
		X_CH4 = X_CH4_BHB
		Y_CH4 = Y_CH4_BHB 
	elif bank_name == 'YDS':
		X_CH4 = X_CH4_YDS 
		Y_CH4 = Y_CH4_YDS 
	elif bank_name == 'BHB_YDS':
		X_CH4 = X_CH4_YDS + X_CH4_BHB
		Y_CH4 = Y_CH4_YDS + Y_CH4_BHB 
		


	X_CH4 = numpy.array(X_CH4) 
	Y_CH4 = numpy.array(Y_CH4)
	# X_CH4 = preprocessing.scale(X_CH4)
	# Y_CH4 = preprocessing.scale(Y_CH4)

	print("split data")
	if bank_name == 'all' or  bank_name == 'MY':
		get_n_cv_split(X_CH4,Y_CH4, exp_data_dir_path+'/DW' ,'DW','CH4', 10)
	else:
		output_data(X_CH4,Y_CH4, exp_data_dir_path + '/DW/' + 'DW_test_CH4.txt')

	#CO2 
	if bank_name == 'all':
		X_CO2 = X_CO2_MY + X_CO2_YDS + X_CO2_BHB
		Y_CO2 = Y_CO2_MY + Y_CO2_YDS + Y_CO2_BHB
	elif bank_name == 'MY':
		X_CO2 = X_CO2_MY  
		Y_CO2 = Y_CO2_MY  
	elif bank_name == 'BHB':
		X_CO2 = X_CO2_BHB
		Y_CO2 = Y_CO2_BHB
	elif bank_name == 'YDS':
		X_CO2 = X_CO2_YDS  
		Y_CO2 = Y_CO2_YDS 
	elif bank_name == 'BHB_YDS':
		X_CO2 = X_CO2_YDS + X_CO2_BHB
		Y_CO2 = Y_CO2_YDS + Y_CO2_BHB

	X_CO2 = numpy.array(X_CO2) 
	Y_CO2 = numpy.array(Y_CO2)
	# X_CO2 = preprocessing.scale(X_CO2)
	# Y_CO2 = preprocessing.scale(Y_CO2)

	if bank_name == 'all' or  bank_name == 'MY':
		get_n_cv_split(X_CO2,Y_CO2, exp_data_dir_path+'/DW' ,'DW', 'CO2',10)
	else:
		output_data(X_CO2,Y_CO2, exp_data_dir_path + '/DW/' + 'DW_test_CO2.txt')


def load_LIT_CH4_subdata(LIT_CH4, beg_index, end_index):
	X_CH4 = []
	Y_CH4 = []
	row_num = LIT_CH4.nrows
	for i in range(beg_index, end_index):
		try:
			hang = []
			idi = LIT_CH4.cell(i,0).value
			hang.append(idi)
			yv = LIT_CH4.cell(i,8).value
			# print(yv)
			hang.append(float(yv))
			Y_CH4.append(hang)
			x_ = []
			for j in range(9,22):
				v = LIT_CH4.cell(i,j).value
				try:
					x_.append(float(v))
				except Exception as e:
					x_.append('null')
			# x_ = numpy.array(x_)
			# x_ = x_.reshape(1,31,1)
			X_CH4.append(x_)
		except Exception as e:
			print(e)
			continue
	X_CH4 = insert_missing_feat(X_CH4)

	return 	X_CH4, Y_CH4

def load_LIT_CO2_subdata(LIT_CO2, beg_index, end_index):  
	X_CO2 = []
	Y_CO2 = []
	
	row_num = LIT_CO2.nrows
	for i in range(beg_index, end_index):
		try:
			hang = []
			idi = LIT_CO2.cell(i,0).value
			hang.append(idi)
			yv = LIT_CO2.cell(i,8).value
			# print(yv)
			hang.append(float(yv))
			Y_CO2.append(hang)
			x_ = []
			for j in range(9,22):
				v = LIT_CO2.cell(i,j).value
				try:
					x_.append(float(v))
				except Exception as e:
					x_.append('null')
			# X_CO2.append(numpy.array(x_))
			X_CO2.append(x_)
		except Exception as e:
			print(e)
			continue

	X_CO2 = insert_missing_feat(X_CO2)

	return X_CO2, Y_CO2

def load_LIT_data(LIT_CH4, LIT_CO2):  
	X_CH4_MY, Y_CH4_MY = load_LIT_CH4_subdata(LIT_CH4, 1, 188) 
	X_CO2_MY, Y_CO2_MY = load_LIT_CO2_subdata(LIT_CO2, 1, 188)

	X_CH4_YDS, Y_CH4_YDS = load_LIT_CH4_subdata(LIT_CH4, 188, 206)  
	X_CO2_YDS, Y_CO2_YDS = load_LIT_CO2_subdata(LIT_CO2,188, 206)  

	X_CH4_BHB, Y_CH4_BHB = load_LIT_CH4_subdata(LIT_CH4, 206, 223) 
	X_CO2_BHB, Y_CO2_BHB = load_LIT_CO2_subdata(LIT_CO2, 206, 223) 


	#CH4
	X_CH4 = X_CH4_MY + X_CH4_YDS + X_CH4_BHB
	Y_CH4 = Y_CH4_MY + Y_CH4_YDS + Y_CH4_BHB 

	if bank_name == 'all':
		X_CH4 = X_CH4_MY + X_CH4_YDS + X_CH4_BHB
		Y_CH4 = Y_CH4_MY + Y_CH4_YDS + Y_CH4_BHB 
	elif bank_name == 'MY':
		X_CH4 = X_CH4_MY  
		Y_CH4 = Y_CH4_MY 
	elif bank_name == 'BHB':
		X_CH4 = X_CH4_BHB
		Y_CH4 = Y_CH4_BHB 
	elif bank_name == 'YDS':
		X_CH4 = X_CH4_YDS  
		Y_CH4 = Y_CH4_YDS
	elif bank_name == 'BHB_YDS':
		X_CH4 = X_CH4_YDS + X_CH4_BHB
		Y_CH4 = Y_CH4_YDS + Y_CH4_BHB 

	X_CH4 = numpy.array(X_CH4) 
	Y_CH4 = numpy.array(Y_CH4)
	# X_CH4 = preprocessing.scale(X_CH4)
	# Y_CH4 = preprocessing.scale(Y_CH4)

	if bank_name == 'all' or  bank_name == 'MY':
		get_n_cv_split(X_CH4,Y_CH4, exp_data_dir_path+'/LIT' ,'LIT','CH4', 10)
	else:
		output_data(X_CH4,Y_CH4, exp_data_dir_path + '/LIT/' + 'LIT_test_CH4.txt')

			
	#CO2  
	if bank_name == 'all':
		X_CO2 = X_CO2_MY + X_CO2_YDS + X_CO2_BHB
		Y_CO2 = Y_CO2_MY + Y_CO2_YDS + Y_CO2_BHB
	elif bank_name == 'MY':
		X_CO2 = X_CO2_MY  
		Y_CO2 = Y_CO2_MY  
	elif bank_name == 'BHB':
		X_CO2 = X_CO2_BHB
		Y_CO2 = Y_CO2_BHB
	elif bank_name == 'YDS':
		X_CO2 = X_CO2_YDS 
		Y_CO2 = Y_CO2_YDS 
	elif bank_name == 'BHB_YDS':
		X_CO2 = X_CO2_YDS + X_CO2_BHB
		Y_CO2 = Y_CO2_YDS + Y_CO2_BHB

	X_CO2 = numpy.array(X_CO2) 
	Y_CO2 = numpy.array(Y_CO2)
	# X_CO2 = preprocessing.scale(X_CO2)
	# Y_CO2 = preprocessing.scale(Y_CO2) 

	if bank_name == 'all' or  bank_name == 'MY':
		get_n_cv_split(X_CO2,Y_CO2, exp_data_dir_path+'/LIT' ,'LIT', 'CO2', 10)
	else:
		output_data(X_CO2,Y_CO2, exp_data_dir_path + '/LIT/' + 'LIT_test_CO2.txt')

	
def load_LIT_N2O_subdata(LIT_N2O, beg_index, end_index):
	X_N2O = []
	Y_N2O = []
	
	row_num = LIT_N2O.nrows
	for i in range(beg_index, end_index):
		try:
			hang = []
			idi = LIT_N2O.cell(i,0).value
			hang.append(idi)
			yv = LIT_N2O.cell(i,8).value
			# print(yv)
			hang.append(float(yv))
			Y_N2O.append(hang)
			x_ = []
			for j in range(9,22):
				v = LIT_N2O.cell(i,j).value
				try:
					x_.append(float(v))
				except Exception as e:
					x_.append('null')
			# X_N2O.append(numpy.array(x_))
			X_N2O.append(x_)
		except Exception as e:
			continue

	X_N2O = insert_missing_feat(X_N2O)

	return 	X_N2O, Y_N2O
 

def load_N2O_data(LIT_N2O):  
	X_N2O_MY, Y_N2O_MY = load_LIT_N2O_subdata(LIT_N2O, 1, 188)  
	X_N2O_YDS, Y_N2O_YDS = load_LIT_N2O_subdata(LIT_N2O, 188, 206)  
	X_N2O_BHB, Y_N2O_BHB = load_LIT_N2O_subdata(LIT_N2O, 206, 223) 


	if bank_name == 'all':
		X_N2O = X_N2O_MY + X_N2O_YDS + X_N2O_BHB
		Y_N2O = Y_N2O_MY + Y_N2O_YDS + Y_N2O_BHB 
	elif bank_name == 'MY':
		X_N2O = X_N2O_MY  
		Y_N2O = Y_N2O_MY  
	elif bank_name == 'BHB':
		X_N2O = X_N2O_BHB
		Y_N2O = Y_N2O_BHB 
	elif bank_name == 'YDS':
		X_N2O = X_N2O_YDS 
		Y_N2O = Y_N2O_YDS
	elif bank_name == 'BHB_YDS':
		X_N2O = X_N2O_YDS + X_N2O_BHB
		Y_N2O = Y_N2O_YDS + Y_N2O_BHB 

	X_N2O = numpy.array(X_N2O) 
	Y_N2O = numpy.array(Y_N2O)
	# X_N2O = preprocessing.scale(X_N2O)
	# Y_N2O = preprocessing.scale(Y_N2O)
	if bank_name == 'all' or  bank_name == 'MY':
		get_n_cv_split(X_N2O,Y_N2O, exp_data_dir_path+'/N2O' ,'LIT', 'N2O', 10)
	else:
		output_data(X_N2O,Y_N2O, exp_data_dir_path + '/N2O/' + 'LIT_test_N2O.txt')

def linear(excel):
	workbook = xlrd.open_workbook(excel)
	DW_CH4 = workbook.sheet_by_name('CH4_T3_DW_n132')
	DW_CO2 = workbook.sheet_by_name('CO2_T3_DW_n132')
	LIT_CH4 = workbook.sheet_by_name('CH4_T3_LIT_n222')
	LIT_CO2 = workbook.sheet_by_name('CO2_T3_LIT_n222')
	LIT_N2O = workbook.sheet_by_name('N2O_T3_LIT_n222')
	load_DW_data(DW_CH4, DW_CO2)
	load_LIT_data(LIT_CH4, LIT_CO2)
	load_N2O_data(LIT_N2O) 

if __name__ == '__main__': 
	linear(excel)
	 
