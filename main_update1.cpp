#include<iostream>
#include<string>
#include<queue>
#include<vector>
#include <algorithm>
#include<time.h>
#include<fstream>
#include "Neural_Network.h"
using namespace std;
int main()
{
	int i;
	ofstream accurcise;
	accurcise.open("test_accuracy_x.txt",ios::app);
	
	for(i = 500; i <= 3500; i = i+500) {
	    init("digitstra.txt",i);
	    cal_weight();
		accurcise << Alpha1 << " " << Alpha2 << " " << HiddenNode <<endl;
		accurcise << i << " " <<   test_accuracy() << endl;
	}

		
	return 0;

}
