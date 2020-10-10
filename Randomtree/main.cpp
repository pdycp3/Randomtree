#include"RandomForest.h"
#include"MnistPreProcess.h"
#include "Mymath.h"
#include "Global.h"
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>


#include <vector>
#include <map>
#include <fstream>
using namespace std;
int main(int argc, const char * argv[])
{
    //1. prepare data
	float**trainset;
	float** testset;
	float*trainlabels;
	float*testlabels;

	//训练数据集60000个float数组
	//验证样本数据集有10000个，float数组
	//每一个特征有784维
	const char * strNodeFile = "D:/3DOpenSource/pdycp-random-forests555-master/RandomForestNode.txt";

	const char * strModelFile = "D:/3DOpenSource/pdycp-random-forests555-master/RandomForesttest.Model";
	trainset=new float*[TRAIN_NUM];
	testset=new float*[TEST_NUM];
	trainlabels=new float[TRAIN_NUM];
	testlabels=new float[TEST_NUM];
	for(int i=0;i<TRAIN_NUM;++i)
	{trainset[i]=new float[FEATURE];}
	for(int i=0;i<TEST_NUM;++i)
	{testset[i]=new float[FEATURE];}
	//60000个样本，每个样本维度784，验证样本有10000
    //readData(trainset,trainlabels,argv[1],argv[2]);
    //readData(testset,testlabels,argv[3],argv[4]);
	string strInputPath = "D:\\data\\colmatest\\features.ibx";
	readDescriptor(trainset, trainlabels, strInputPath.c_str());
	//readData(trainset,trainlabels,
 //       "D:/3DOpenSource/pdycp-random-forests555-master/train-images.idx3-ubyte",
	//	"D:/3DOpenSource/pdycp-random-forests555-master/train-labels.idx1-ubyte");
	//readData(testset,testlabels,
 //       "D:/3DOpenSource/pdycp-random-forests555-master/t10k-images.idx3-ubyte",
	//	"D:/3DOpenSource/pdycp-random-forests555-master/t10k-labels.idx1-ubyte");
    
    //2. create RandomForest class and set some parameters
	//100棵树，每棵树最大深度不能超多10个，每个叶节点不能少于10个，最小信息为0；
	//初始化随机森林参数
	RandomForest randomForest(NUMTREES,DEPTHTREE,MINNUM,0);
 //   
	////3. start to train RandomForest
	//randomForest.train(trainset,trainlabels,TRAIN_NUM,FEATURE,10,true,56);//regression
	printf("start to train RandomForest....................\n");
    randomForest.train(trainset,trainlabels,TRAIN_NUM,FEATURE,NUMIMAGE,false);//classification
	//
 //   //restore model from file and save model to file
	printf("restore model from file and save model to file....................\n");
	randomForest.saveModel(strModelFile,strNodeFile,trainset);
   // randomForest.readModel(strModelFile);
	//RandomForest randomForest("D:/3DOpenSource/pdycp-random-forests555-master/RandomForest2.Model");
    
    //predict single sample
//  float resopnse;
//	randomForest.predict(testset[0],resopnse);
    //构造检索特征
	printf("start make test samples....................\n");
	for (int i=0;i<TEST_NUM;++i)
	{
		//int nid = rand() % TRAIN_NUM;
		int nid = i;
		for (int j=0;j<128;++j)
		{
			testset[i][j] = trainset[nid][j];
			testlabels[i] = trainlabels[nid];
		}
	}
	printf("start predict test samples....................\n");
    //predict a list of samples 输出树的id和节点id；
    float*resopnses=new float[TEST_NUM*2*NUMTREES];
	randomForest.predict(testset,TEST_NUM,resopnses);
	//float * P=resopnses;
	//int nNodeMax = pow(2, DEPTHTREE);
	//for (int i=0;i<TEST_NUM*2*NUMTREES;i=i+2)
	//{
	//	if (P[i]<0||P[i]>NUMTREES)
	//	{
	//		printf("tree id is unnormal\n");
	//	}

	//	if (P[i+1]<0||P[i+1]>nNodeMax)
	//	{
	//		printf("node id is unnormal\n");
	//	}
	//}
	//predict a list of samples_use
	//float ** resopnses_features = new float*[TEST_NUM];
	//for (int i=0;i<TEST_NUM;++i)
	//	resopnses_features[i] = new float[NUMTREES*NUMSELECT];
	//randomForest.predict(testset,resopnses);
	//将导出的树的id和预测的节点导出测试数据
	//创建模型节点树
	printf("start read node file ....................\n");
	vector<map<int, int>> vecNodeTree;
	vecNodeTree.reserve(NUMTREES);
	//读取节点文件
	int nCapcity = 1,nRows=0;
	ifstream inFile;
	inFile.open(strNodeFile);
	string strCurretnLine;
	vector<vector<int>> vecNode;
	vecNode.reserve(37000000);
	//构建树节点索引和叶节点索引
	vector<vector<int>> vecTreeIndex;
	int nTreeStart = 0, nTreeEnd = 0,nCurrentTreeId=0;
	while (!inFile.eof())
	{
		printf("......read %d th row....................\n",nRows);
		vector<int> vecCurrentNode;
		vecCurrentNode.reserve(3);
		getline(inFile, strCurretnLine);
		vector<string> strList;
		boost::split(strList, strCurretnLine,boost::is_any_of( ","),boost::token_compress_off);
		if (strList.size()==3)
		{
			int nTreeid = std::atoi(strList[0].c_str());
			if (nTreeid != nCurrentTreeId)
			{
				vector<int> vecRange;
				nTreeEnd = nRows - 1;
				vecRange.push_back(nTreeStart);
				vecRange.push_back(nTreeEnd);
				vecTreeIndex.push_back(vecRange);
				nTreeStart = nRows;
				nCurrentTreeId++;
			}
			for (int i = 0; i < 3; ++i)
				vecCurrentNode.push_back(std::atoi(strList[i].c_str()));
			vecNode.push_back(vecCurrentNode);
		/*	if (nRows + 2 < nCapcity * 1000000)
				
			else
			{
				nCapcity++;
				vecNode.reserve(nCapcity * 1000000);
				vecNode.push_back(vecCurrentNode);
			}*/
			nRows++;
		}
	}
	vector<int> vecRange;
	nTreeEnd = nRows - 1;
	vecRange.push_back(nTreeStart);
	vecRange.push_back(nTreeEnd);
	vecTreeIndex.push_back(vecRange);
	nTreeStart = nRows;
	nCurrentTreeId++;
	//开辟结果空间
	printf("make memory for result....................\n");
	float** dij = new float*[TEST_NUM];
	int ** pij = new int *[TEST_NUM];
	for (int i=0;i<TEST_NUM;++i)
	{
		dij[i] = new float[NUMIMAGE+1];
		pij[i] = new int[NUMIMAGE+1];
	}
	//进行检索计算
	printf("start find pij and dij....................\n");
	for (int i=0;i<TEST_NUM;++i)
	{
		printf("........find %d th pij  dij ....................\n",i);
		float * p = new float[NUMTREES * 2];
		memcpy(p, resopnses + (i*NUMTREES * 2), sizeof(float)*NUMTREES * 2);
	/*	for (int i = 0; i < 2 * NUMTREES; i = i + 2)
		{
			if (p[i]<0 || p[i]>NUMTREES)
			{
				printf("tree id is unnormal\n");
			}

			if (p[i + 1]<0 || p[i + 1]>nNodeMax)
			{
				printf("node id is unnormal\n");
			}
		}*/
		int * sample_id=nullptr;
		int nSamplenumber=0;
		GetSampleid(vecNode,vecTreeIndex, p, NUMTREES * 2, &sample_id, nSamplenumber);
		printf("sample %d th have %d neighbor\n",i,nSamplenumber);
		if (nSamplenumber>0)
		{
			float score_distance = 0;
			distance_id * score_array = new distance_id[nSamplenumber];
			int *pdataset = sample_id;
			for (int j = 0; j < nSamplenumber; ++j)
			{
				int nId = *pdataset;
			/*	if (nId>=66075)
				{
					printf("--------样本越界>------\n");
				}
				if (nId<0)
				{
					printf("--------样本越界<------\n");
				}*/
				float distance_curret;
				float * ptestSample = testset[i];
				float * pcurrentsample = trainset[nId];
				distance_curret = DistanceL2(ptestSample, pcurrentsample, 128);
				if (nId<0||nId>=TRAIN_NUM)
				{
					printf("************current SAM ID =%f****************\n", nId);
				}
				if (distance_curret<0||distance_curret>1000)
				{
					printf("************current SAM DL =%f  is unnormal****************\n", distance_curret);
				}
				
				score_array[j].distance = distance_curret;
				score_array[j].id_sample = nId;
				score_array[j].lable = trainlabels[nId];
				pdataset++;
			}
			qsort(score_array,  nSamplenumber, sizeof(distance_id), my_compare_low);

			//整理数据
			map<int, int> map_comjacent;
			pij[i][NUMIMAGE] = testlabels[i];
			//初始化数据输出
			for (int n = 0; n < NUMIMAGE; ++n)
			{
				pij[i][n] = 0;
				dij[i][n] = 0;
			}
				
			int nSave = nSamplenumber > NUMCLASSSAVE ? NUMCLASSSAVE : nSamplenumber;
			for (int k = 0; k < nSave; ++k)
			{
				int nlable = score_array[k].lable;
				float fdistance = score_array[k].distance;
				if (fdistance>=0&&fdistance<=10)
				{

				}
				else
				{
					printf("!!!!!!!!!Current Distance=%f is unnormal!!!!!!!!!!!!\n",fdistance);
					printf("current test sample id is testset[%d]:", i);
					for (int l = 0; l < 128; ++l)
					{
						printf("%f,", testset[i][l]);
					}
					printf("\n");
					printf("current node sample id is traindata[%d]:", score_array[k].id_sample);
					for (int l = 0; l < 128; ++l)
					{
						printf("%f,", trainset[score_array[k].id_sample][l]);
					}
					printf("\n");

				}
				if (nlable >= 0 && nlable < NUMIMAGE)
				{
					pij[i][nlable] += 1;
					if (pij[i][nlable]>0)
					{
						dij[i][nlable] += (fdistance - dij[i][nlable]) / pij[i][nlable];
					}
					else
					{
						dij[i][nlable] += fdistance;
					}
					
				}
			/*	else {
				
					printf("!!!!!!!!!Current Label is unnormal!!!!!!!!!!!!\n");

				
				}*/
			}
			if (score_array != NULL)
				delete[] score_array;
		}
		if (p != NULL)
			delete[] p;
	}
	//TODO:统计值计算
	printf("calculate all the pij and dij count....................\n");
	int **PijCount = new int *[NUMIMAGE];
	float **DijCount = new float *[NUMIMAGE];
	float **SijCount = new float *[NUMIMAGE];
	for (int i=0;i<NUMIMAGE;++i)
	{
		PijCount[i] = new int[NUMIMAGE];
		DijCount[i] = new float[NUMIMAGE];
		SijCount[i] = new float[NUMIMAGE];
		for (int j = 0; j < NUMIMAGE; ++j)
		{
			PijCount[i][j] = 0;
			DijCount[i][j] = 0;
			SijCount[i][j] = 0;
		}
	}
	//计算所有的查询样本
	for (int i=0;i<TEST_NUM;++i)
	{
		int nfind_i = pij[i][NUMIMAGE];
		for (int j=0;j<NUMIMAGE;++j)
		{
			PijCount[nfind_i][j] += pij[i][j];
			
			DijCount[nfind_i][j] += dij[i][j];
			//printf("///////////pij=%d,dij=%f////////\n", pij[i][j],dij[i][j]);
		}
	}
	//系数计算//选择前r项，计算评价指标计算公式：sij=log(10)(pij)*exp(-dij/pij);
	for (int i = 0; i < NUMIMAGE; ++i)
	{
		for (int j = 0; j < NUMIMAGE; ++j)
		{
			if (PijCount[i][j]<=0)
			{
				SijCount[i][j] = 0;
			}
			else
			{
				SijCount[i][j] = log10f(PijCount[i][j] * exp(-DijCount[i][j] / PijCount[i][j]));
			}
			printf("S%d%d=%f,p=%d,d=%f\n", i, j, SijCount[i][j],PijCount[i][j],DijCount[i][j]);
		}
		printf("\n");
			
	}

//	float errorRate = 0;
//	for (int i = 0; i < TEST_NUM; ++i)
//	{
//		if (resopnses[i] != testlabels[i])
//		{
//			errorRate += 1.0f;
//		}
//		//for regression
////		float diff=abs(resopnses[i]-testlabels[i]);
////		errorRate+=diff;
//	}
//	errorRate /= TEST_NUM;
//	printf("the total error rate is:%f\n", errorRate);
	printf("program was executed successfully\n");
	delete[] resopnses;
	for(int i=0;i<TRAIN_NUM;++i)
	{delete[] trainset[i];}
	for(int i=0;i<TEST_NUM;++i)
	{delete[] testset[i];}
	for (int i=0;i<NUMIMAGE;++i)
	{
		delete[] PijCount[i];
		delete[] DijCount[i];
		delete[] SijCount[i];
	}
	delete[] PijCount;
    delete[] DijCount;
    delete[] SijCount;
	delete[] trainlabels;
	delete[] testlabels;
	delete[] trainset;
	delete[] testset;
	return 0;
};
