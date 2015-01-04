#include<iostream>
#include<string>
#include<queue>
#include<vector>
#include <algorithm>
#include<time.h>
#include<fstream>
#include <math.h>

#include "Neural_Network.h"
using namespace std;

// w1： 输入层到隐层权重， w2：隐层到输出层权重, 下标从 1 开始
double w1[InputNode][HiddenNode];
double w2[HiddenNode][OutputNode];

string train_file;      // 训练集文件名 
int Times;
int input_px[InputNode];       //输入的 64 个像素值，隐藏层的输入
double hidden_out[HiddenNode]; // 隐藏层的输出，即输出层的的输入
double output_num[OutputNode];                 // 输出层的最后输出
int expect_out[OutputNode];
int goal_num;                      // 样本的输出数字，即目标数字

int train_match;
double e;  // 单个样本方差估计
double etemp; // 上一个样本的方差
double etempnext; // 当前样本的方差
// 自适应学习率
double alpha1;
double alpha2;
double MSE;  // 全局均方差，判断学习终止的条件之一
double sum1[HiddenNode]; // 输入层到隐藏层的求和
double sum2[OutputNode];             // 隐藏层到输出层的求和
double delta_Output[OutputNode];     // delta_Output = Err * g'(in)
double delta_Hidden[HiddenNode];  //   delta_Hidden[j] = g'(sum1[j]) *  w2[j][OutputNode] * delta_Output

int test_input_px[InputNode]; 
int test_expect_out[OutputNode];
int test_goal_num;

 // 初始化变量值
void init(string tra_file, int num)       
{
	train_file = tra_file;
	Times = num;
	memset(w1, 0, sizeof(w1));
	memset(w2, 0, sizeof(w2));
	memset(sum1, 0, sizeof(sum1));
	memset(sum2, 0, sizeof(sum2));
	memset(input_px, 0, sizeof(input_px));
    memset(output_num, 0, sizeof(output_num));
	memset(expect_out, 0, sizeof(expect_out));
	memset(test_input_px, 0, sizeof(test_input_px));
	memset(hidden_out, 0, sizeof(hidden_out));
	memset(delta_Hidden, 0, sizeof(delta_Hidden));
	weight_init();
	e = 0.0;
	etemp = 3.0;
	alpha1 = Alpha1;
	alpha2 = Alpha2;
}
// 初始化，随机给权重赋初值
void weight_init() 
{
	int i, j;
	// 第 1 层权重初始化
	for (i = 0; i < InputNode; i++) 
		for (j = 0; j < HiddenNode; j++) 
			w1[i][j] = (0.2*(double)rand()/RAND_MAX)-0.1; //rand()/(double)(RAND_MAX)*(0.5) - 0.25;

	// 第 2 层权重初始化
	for (i = 0; i < HiddenNode; i++) 
		for (j = 0; j < OutputNode; j++) 
			w2[i][j] =(0.2*(double)rand()/RAND_MAX)-0.1;  // rand()/(double)(RAND_MAX)*(0.5) - 0.25;

}

// 计算权值
void cal_weight()                             
{
	int i, time;
	char x;
	time = 0;
	// 打开训练集文件，逐行读数据，对每行数据都校正一次权重
	ifstream traindata;
	traindata.open(train_file);
	while (1) {
		for(i = 0; i < InputNode; i++) 
			traindata >> input_px[i] >> x;
		traindata >> goal_num;
		for(i = 0; i < OutputNode; i++) 
			expect_out[i] = (i == goal_num )? 1:0;		
		one_sample_train() ;
		time++;
		//alpha1 = A *exp(lamda * time);
		MSE = e /((double)time * (double) OutputNode);
		if ( MSE < EPS || time > Times) 
			break;
		
	}
	traindata.close();
	cout << "Trainning finished, The time is :" << time << "    The error is :" << MSE  << endl;
	cout << "alpha1: " << alpha1 << "  alpha2:  " << alpha2 <<endl;

	
} 
// 对每个样本进行迭代
void one_sample_train()                     
{
	int i, j, k;
	// 求出隐藏层各个单元的输出
	for (j = 0; j < HiddenNode; j++) {
		sum1[j] = 0.0;
        for (i = 0; i < InputNode; i++) 
			sum1[j] = sum1[j] + (w1[i][j] * (double)input_px[i]);
		hidden_out[j] = 1.0/(1.0 + exp(-sum1[j]));	
	}
	// 求出输出层各个单元的输出
	for (j = 0; j < OutputNode; j++) {
		sum2[j] = 0.0;
		for (i = 0; i < HiddenNode; i++) 
			sum2[j] = sum2[j] + (w2[i][j] * hidden_out[i] );
		output_num[j] = 1.0/(1.0 + exp(-sum2[j]));
	}
	//计算方差, 调整学习率
	
	etempnext = 0.0; // 当前样本方差
	for (i = 0; i < OutputNode; i++)
		etempnext = etempnext +  (expect_out[i] - output_num[i]) * (expect_out[i] - output_num[i]);
	// debug cout << etemp << endl;
	if (etempnext < 0.7 *etemp) {
		alpha1 = 1.0001 * alpha1;
		alpha2 = 1.0002 * alpha2;
	}
	else if (etempnext > 14 * etemp) {
		alpha1 = 0.995 * alpha1;
		alpha2 = 0.999 * alpha2;
	}
	

	// 计算误差，校正隐藏层到输出层的权重值. i: 输出层； j：隐藏层
	/*
	   Wj,i <- Wj,i + α2 * [error[i] * g'(sum2[i])] * hidden_out[j]
	   double error[i] = expect_out[i] - output_num[i];
	   double g'(sum2[i]) = exp(-sum2[i]) / ((1.0 + exp(-sum2[i])) * (1.0 + exp(-sum2[i])))
	   因为 output_num[i] = 1.0/(1.0 + exp(-sum2[i]));
	   所以 g'(sum2[i]) = (1 - output_num[i]) * output_num[i];

	   令 delta_Output =  error * g'(sum2)  ;
	*/
	for (i = 0; i < OutputNode; i++) {
		delta_Output[i] = (expect_out[i] - output_num[i]) * (1.0 - output_num[i]) * output_num[i];
		for (j = 0; j < HiddenNode; j++) 
		   w2[j][i] = w2[j][i] + alpha2 * delta_Output[i] * hidden_out[j];	
	}


	// 误差后向传播，校正输入层到隐藏层的权重值   k: 输入层； j：隐藏层
	/*
	    Wk,j <-  Wk,j + α1 * delta_Hidden[j] * input_px[k] 
	    delta_Hidden[j] = g'(sum1[j]) * ∑i  w2[j][i] * delta_Output[i] ;
		 g'(sum1[j]) = (1 - hidden_out[j]) * hidden_out[j]
	*/
	for (j = 0; j < HiddenNode; j++) {
		delta_Hidden[j] = 0.0;
		for (i = 0; i < OutputNode; i++)
			delta_Hidden[j] = delta_Hidden[j] + w2[j][i] * delta_Output[i];
		delta_Hidden[j] = delta_Hidden[j] * (1 - hidden_out[j]) * hidden_out[j];

		for (k = 0; k < InputNode; k++) 
			w1[k][j] = w1[k][j] + alpha1 * delta_Hidden[j] * (double)input_px[k];
	}


	etemp = etempnext;
    e += etemp;  // 全局方差和
} 

// 利用测试集数据测试准确度
double test_accuracy()  
{
	int i, testnum, match;
	double accuracy;
	char x;
	ifstream testdata;
	testdata.open("digitstest.txt");
	testnum = match = 0;
	while(testnum < Testnum) {
		for(i = 0; i < InputNode; i++) {
			testdata >> test_input_px[i] >> x;
		}
		testdata >> test_goal_num;
		if (recognize() == test_goal_num)
			match++;
		testnum++;	  
	}
	accuracy = (double) match / (double) Testnum;
	testdata.close();
	return accuracy;
}

int recognize()
{
    int i, j, result;
	double std_out = 0.0;
	// 求出隐藏层各个单元的输出
	for (j = 0; j < HiddenNode; j++) {
		sum1[j] = 0.0;
        for (i = 0; i < InputNode; i++) 
			sum1[j] = sum1[j] + (w1[i][j] * (double)test_input_px[i]);
		hidden_out[j] = 1.0/(1.0 + exp(-sum1[j]));	
	}
	// 求出输出层各个单元的输出
	for (j = 0; j < OutputNode; j++) {
		sum2[j] = 0.0;
		for (i = 0; i < HiddenNode; i++) 
			sum2[j] = sum2[j] + (w2[i][j] * hidden_out[i] );
		output_num[j] = 1.0/(1.0 + exp(-sum2[j]));
		if (output_num[j] > std_out) {
			result = j;
			std_out = output_num[j];
		}
	}
	return result;
}
