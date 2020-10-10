#pragma once
#ifndef MYMATH_H
#define MYMATH_H
#include <math.h>
#include <vector>
#include <unordered_map>
using namespace std;
struct distance_id 
{
	int id_sample;
	float lable;
	float distance;
};
static int my_compare_low( const void  *T1,const void * T2)
{
	distance_id * p1 = (distance_id*)T1;
	distance_id * p2 = (distance_id*)T2;
	return p1->distance - p2->distance;
	/*if ((p1->distance - p2->distance)>0)
	{
		return 1;
	}
	return  0;*/
}
void GetSampleid(vector<vector<int>>& vecNodeFile, vector<vector<int>>&vecTreeIndex,float * id_tree_node,int nLength,int **sample_id,int &nSamplenum)
{
	int nvec_size = vecNodeFile.size();
	unordered_map<int, int> m_return;
	for (int i=0;i<nLength;i=i+2)
	{
		//printf("............CAL %d th TREE..................\n",i);
		int nTreeid = id_tree_node[i];
		int nNodeid = id_tree_node[i + 1];
		int nStart = vecTreeIndex[nTreeid][0];
		int nEnd = vecTreeIndex[nTreeid][1];
		for (int j=nStart;j<=nEnd;++j)
		{
			/*if (vecNodeFile[j][0]==nTreeid)
			{*/
			if (vecNodeFile[j][1] == nNodeid)
			{
				if (!m_return[vecNodeFile[j][2]])
					m_return[vecNodeFile[j][2]] = 1;
				
			}
			//}
		}
	}
	nSamplenum = m_return.size();
	if (nSamplenum>0)
	{
		int * temp = new int[nSamplenum];
		*sample_id = temp;
		for (auto & t : m_return)
		{
			*temp = t.first;
			temp++;
		}
	}
	
}

void GetSampleId(float * sample_predict,float lable,float * train_lables,int nReturnnum,int * res_class,int * sampleId,float * res_score)
{
	
	vector<distance_id> vecOut;









}










float DistanceL2(const float * T1, const float * T2, int nDim)
{
	float dReturn=0;
	//πÈ“ªªØ
	float dT1L2_model = 0, dT2L2_model = 0;
	for (int i=0;i<nDim;++i)
	{
		dT1L2_model += (T1[i] * T1[i]);
		dT2L2_model += (T2[i] * T2[i]);
	}
	dT1L2_model = sqrt(dT1L2_model);
	dT2L2_model = sqrt(dT2L2_model);
	for (int i = 0; i < nDim; ++i)
		dReturn += (T1[i]/dT1L2_model - T2[i]/dT2L2_model)*(T1[i]/dT1L2_model - T2[i]/dT2L2_model);


	dReturn = sqrt(dReturn);
	return dReturn;
}



#endif
