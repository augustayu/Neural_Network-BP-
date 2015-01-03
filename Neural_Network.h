/* 
   三层神经网络：
     1 输入层 64 个单元（对应64个像素值）
	 2 隐藏层 8 个单元
	 3 输出层 1 个单元（对应数字）
*/
#include<iostream>
#include<string>
#include<queue>
#include<vector>
#include <algorithm>
#include<time.h>
#include<fstream>
using namespace std;
#define Alpha1 0.02 //权值学习率α  输入层到隐层
#define Alpha2 0.98  //权值学习率α  隐层到输出层
#define EPS 0.00001   // 终止学习条件，误差精度判断
#define Times 3822    // 终止学习条件，学习迭代次数判断
#define Testnum 1517        // 测试样例个数
// 神经网络各层节点数   
#define InputNode 64
#define HiddenNode 48
#define OutputNode 10



void init(string tra_file);         // 初始化变量值
void weight_init();  // 初始化，随机给权重赋初值
void one_sample_train();                        // 对每个样本进行迭代
void cal_weight();                              // 计算权值
void write_weights_ToFile();                    // 将最后得到的权值写入文件
int recognize();
void test_accuracy();                           // 利用测试集数据测试准确度

