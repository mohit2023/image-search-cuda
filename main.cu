#include <bits/stdc++.h>

using namespace std;

__global__
void GPUFunc(float *data,float *query,int dR,int dC,int qR,int qC,float t1,float t2, int n, float qSummary, float *dOutput, int *dLock){	
	
	int xCor = threadIdx.x + blockIdx.x * blockDim.x;
	int yCor = threadIdx.y + blockIdx.y * blockDim.y;

	if(blockIdx.z == 0){
		if(xCor+qC < dC && yCor+qR < dR){
			float summary = 0.0;
			for(int j=0;j<qR;j++){
				for(int i=0;i<qC;i++){
					//int qIndex = ((qR-1-j)*qC+i)*3;
					int dIndex = ((dR-1-j-yCor)*dC + i+xCor)*3;
					summary += (data[dIndex] + data[dIndex+1] + data[dIndex+2])/3;
				}
			}
			summary = summary/(qC*qR);
			if(abs(summary-qSummary) < t2){
				float rmsd = 0.0;
				for(int j=0;j<qR;j++){
					for(int i=0;i<qC;i++){
						int qIndex = ((qR-1-j)*qC+i)*3;
						int dIndex = ((dR-1-j-yCor)*dC + i+xCor)*3;
						rmsd += (data[dIndex] - query[qIndex])*(data[dIndex] - query[qIndex]);
						rmsd += (data[dIndex+1] - query[qIndex+1])*(data[dIndex+1] - query[qIndex+1]);
						rmsd += (data[dIndex+2] - query[qIndex+2])*(data[dIndex+2] - query[qIndex+2]);
					}
				}
				rmsd = rmsd/(qR*qC*3);
				rmsd = sqrt(rmsd);
				bool flag = true;
				while(rmsd < dOutput[4*(n-1)] && flag){
					if (atomicExch(dLock,1) == 0) {
						if(rmsd < dOutput[4*(n-1)]){
							int i = n-1;
							while(i > 0){
								if(rmsd < dOutput[4*(i-1)]){
									dOutput[4*i] = dOutput[4*(i-1)];
									dOutput[4*i+1] = dOutput[4*(i-1)+1];
									dOutput[4*i+2] = dOutput[4*(i-1)+2];
									dOutput[4*i+3] = 0;
									i--;
								}
								else{
									break;
								}
							}
							dOutput[4*i] = rmsd;
							dOutput[4*i+1] = yCor;
							dOutput[4*i+2] = xCor;
							dOutput[4*i+3] = 0;
						}
						flag = false;
						atomicExch(dLock,0);
					}
				}
			}
		}
	}

	else if(blockIdx.z==1){

		float sinAngle = -1/(sqrtf(2));
		float cosAngle = -1*sinAngle;

		int lEnd = 0;
		int rEnd = ceilf((qC-1)*cosAngle - (qR-1)*sinAngle);
		int uEnd = ceilf((qR-1)*cosAngle);
		int dEnd = floorf((qC-1)*sinAngle);

		if(rEnd + xCor < dC && uEnd + yCor < dR && dEnd + yCor >= 0){
			float summary = 0.0;
			for(int j=dEnd;j<uEnd;j++){
				for(int i=lEnd;i<rEnd;i++){
					//int qIndex = ((qR-1-j)*qC+i)*3;
					int dIndex = ((dR-1-j-yCor)*dC + i+xCor)*3;
					summary += (data[dIndex] + data[dIndex+1] + data[dIndex+2])/3;
				}
			}
			summary = summary/((rEnd-lEnd)*(uEnd-dEnd));
			if(abs(summary-qSummary) < t2){
				float rmsd = 0.0;
				for(int j=0;j<qR;j++){
					for(int i=0;i<qC;i++){
						int qIndex = ((qR-1-j)*qC+i)*3;
						float rxCor = xCor + i*cosAngle - j*sinAngle;
						float ryCor = yCor + i*sinAngle + j*cosAngle;

						int dIndex00 = ((dR-1-int(ryCor))*dC + int(rxCor))*3;
						int dIndex10 = ((dR-1-int(ryCor))*dC + int(rxCor)+1)*3;
						int dIndex01 = ((dR-1-int(ryCor)-1)*dC + int(rxCor))*3;
						int dIndex11 = ((dR-1-int(ryCor)-1)*dC + int(rxCor)+1)*3;

						float temp;
						temp = data[dIndex00]*(1-rxCor+floorf(rxCor))*(1+floorf(ryCor)-ryCor) + data[dIndex10]*(rxCor-floorf(rxCor))*(1+floorf(ryCor)-ryCor) + data[dIndex01]*(1-rxCor+floorf(rxCor))*(ryCor-floorf(ryCor)) + data[dIndex11]*(rxCor-floorf(rxCor))*(ryCor-floorf(ryCor))  - query[qIndex];
						temp = temp*temp;
						rmsd += temp;

						temp = data[dIndex00+1]*(1-rxCor+floorf(rxCor))*(1+floorf(ryCor)-ryCor) + data[dIndex10+1]*(rxCor-floorf(rxCor))*(1+floorf(ryCor)-ryCor) + data[dIndex01+1]*(1-rxCor+floorf(rxCor))*(ryCor-floorf(ryCor)) + data[dIndex11+1]*(rxCor-floorf(rxCor))*(ryCor-floorf(ryCor))  - query[qIndex+1];
						temp = temp*temp;
						rmsd += temp;

						temp = data[dIndex00+2]*(1-rxCor+floorf(rxCor))*(1+floorf(ryCor)-ryCor) + data[dIndex10+2]*(rxCor-floorf(rxCor))*(1+floorf(ryCor)-ryCor) + data[dIndex01+2]*(1-rxCor+floorf(rxCor))*(ryCor-floorf(ryCor)) + data[dIndex11+2]*(rxCor-floorf(rxCor))*(ryCor-floorf(ryCor))  - query[qIndex+2];
						temp = temp*temp;
						rmsd += temp;
					}
				}
				rmsd = rmsd/(qR*qC*3);
				rmsd = sqrt(rmsd);
				bool flag = true;
				while(rmsd < dOutput[4*(n-1)] && flag){
					if (atomicExch(dLock,1) == 0) {
						if(rmsd < dOutput[4*(n-1)]){
							int i = n-1;
							while(i > 0){
								if(rmsd < dOutput[4*(i-1)]){
									dOutput[4*i] = dOutput[4*(i-1)];
									dOutput[4*i+1] = dOutput[4*(i-1)+1];
									dOutput[4*i+2] = dOutput[4*(i-1)+2];
									dOutput[4*i+3] = -45;
									i--;
								}
								else{
									break;
								}
							}
							dOutput[4*i] = rmsd;
							dOutput[4*i+1] = yCor;
							dOutput[4*i+2] = xCor;
							dOutput[4*i+3] = -45;
						}
						flag = false;
						atomicExch(dLock,0);
					}
				}
			}
		}
	}

	else if(blockIdx.z == 2){

		float sinAngle = 1/(sqrtf(2));
		float cosAngle = sinAngle;

		int lEnd = floorf((1-qR)*sinAngle);
		int rEnd = ceilf((qC-1)*cosAngle);
		int uEnd = ceilf((qC-1)*sinAngle + (qR-1)*cosAngle);
		int dEnd = 0;

		if(rEnd + xCor < dC && uEnd + yCor < dR && lEnd + xCor >= 0){
			float summary = 0.0;
			for(int j=dEnd;j<uEnd;j++){
				for(int i=lEnd;i<rEnd;i++){
					//int qIndex = ((qR-1-j)*qC+i)*3;
					int dIndex = ((dR-1-j-yCor)*dC + i+xCor)*3;
					summary += (data[dIndex] + data[dIndex+1] + data[dIndex+2])/3;
				}
			}
			summary = summary/((rEnd-lEnd)*(uEnd-dEnd));
			if(abs(summary-qSummary) < t2){
				float rmsd = 0.0;
				for(int j=0;j<qR;j++){
					for(int i=0;i<qC;i++){
						int qIndex = ((qR-1-j)*qC+i)*3;
						float rxCor = xCor + i*cosAngle - j*sinAngle;
						float ryCor = yCor + i*sinAngle + j*cosAngle;

						int dIndex00 = ((dR-1-int(ryCor))*dC + int(rxCor))*3;
						int dIndex10 = ((dR-1-int(ryCor))*dC + int(rxCor)+1)*3;
						int dIndex01 = ((dR-1-int(ryCor)-1)*dC + int(rxCor))*3;
						int dIndex11 = ((dR-1-int(ryCor)-1)*dC + int(rxCor)+1)*3;

						float temp;
						temp = data[dIndex00]*(1-rxCor+floorf(rxCor))*(1+floorf(ryCor)-ryCor) + data[dIndex10]*(rxCor-floorf(rxCor))*(1+floorf(ryCor)-ryCor) + data[dIndex01]*(1-rxCor+floorf(rxCor))*(ryCor-floorf(ryCor)) + data[dIndex11]*(rxCor-floorf(rxCor))*(ryCor-floorf(ryCor))  - query[qIndex];
						temp = temp*temp;
						rmsd += temp;

						temp = data[dIndex00+1]*(1-rxCor+floorf(rxCor))*(1+floorf(ryCor)-ryCor) + data[dIndex10+1]*(rxCor-floorf(rxCor))*(1+floorf(ryCor)-ryCor) + data[dIndex01+1]*(1-rxCor+floorf(rxCor))*(ryCor-floorf(ryCor)) + data[dIndex11+1]*(rxCor-floorf(rxCor))*(ryCor-floorf(ryCor))  - query[qIndex+1];
						temp = temp*temp;
						rmsd += temp;

						temp = data[dIndex00+2]*(1-rxCor+floorf(rxCor))*(1+floorf(ryCor)-ryCor) + data[dIndex10+2]*(rxCor-floorf(rxCor))*(1+floorf(ryCor)-ryCor) + data[dIndex01+2]*(1-rxCor+floorf(rxCor))*(ryCor-floorf(ryCor)) + data[dIndex11+2]*(rxCor-floorf(rxCor))*(ryCor-floorf(ryCor))  - query[qIndex+2];
						temp = temp*temp;
						rmsd += temp;
					}
				}
				rmsd = rmsd/(qR*qC*3);
				rmsd = sqrt(rmsd);
				bool flag = true;
				while(rmsd < dOutput[4*(n-1)] && flag){
					if (atomicExch(dLock,1) == 0) {
						if(rmsd < dOutput[4*(n-1)]){
							int i = n-1;
							while(i > 0){
								if(rmsd < dOutput[4*(i-1)]){
									dOutput[4*i] = dOutput[4*(i-1)];
									dOutput[4*i+1] = dOutput[4*(i-1)+1];
									dOutput[4*i+2] = dOutput[4*(i-1)+2];
									dOutput[4*i+3] = 45;
									i--;
								}
								else{
									break;
								}
							}
							dOutput[4*i] = rmsd;
							dOutput[4*i+1] = yCor;
							dOutput[4*i+2] = xCor;
							dOutput[4*i+3] = 45;
						}
						flag = false;
						atomicExch(dLock,0);
					}
				}
			}
		}
	}
}

int main(int argc, char* argv[]){

	string data_image = argv[1];
	string query_image = argv[2];
	float t1 = stof(argv[3]);
	float t2 = stof(argv[4]);
	int n = stoi(argv[5]);

	int dR,dC,qR,qC;

	ifstream MyReadFile;
	MyReadFile.open(data_image);
	MyReadFile >> dR;
	MyReadFile >> dC;
	int dataSize = dR*dC*3;
	float* hData = (float*)malloc(dataSize*sizeof(float));
	for(int i =0;i<dataSize;i++){
		
		MyReadFile >> hData[i];
	}
	MyReadFile.close();

	float qSummary = 0.0;
	MyReadFile.open(query_image);
	MyReadFile >> qR;
	MyReadFile >> qC;
	int querySize = qR*qC*3;
	float* hQuery = (float*)malloc(querySize*sizeof(float));
	for(int i =0;i<querySize;i++){

		MyReadFile >> hQuery[i];
		qSummary += hQuery[i];
	}
	MyReadFile.close();
	qSummary = qSummary/(3*qC*qR);

	int outputSize = 4*n*sizeof(float);
	float* hOutput = (float*)malloc(outputSize);
	int *hLock = (int*)malloc(sizeof(int));
	hLock[0] = 0;
	for(
	int i=0;i<n;i++){
		hOutput[4*i] = t1;
	}
	
	int *dLock;

	float *dQuery, *dData, *dOutput;
	float* hSummary = (float*)malloc(dataSize*sizeof(float));

	dim3 blockGrid(ceil(dC/32),ceil(dR/32),3);
	dim3 threadGrid(32,32);

	cudaMalloc((void**)&dData, dataSize*sizeof(float));
	cudaMalloc((void**)&dQuery, querySize*sizeof(float));
	cudaMalloc((void**)&dOutput,outputSize);
	cudaMalloc((void**)&dLock,sizeof(int));

	cudaMemcpy(dData, hData, dataSize*sizeof(float), cudaMemcpyDefault);
	cudaMemcpy(dQuery, hQuery, querySize*sizeof(float), cudaMemcpyDefault);
	cudaMemcpy(dOutput, hOutput, outputSize, cudaMemcpyDefault);
	cudaMemcpy(dLock, hLock, sizeof(int), cudaMemcpyDefault);

	GPUFunc<<<blockGrid,threadGrid>>>(dData,dQuery,dR,dC,qR,qC,t1,t2,n,qSummary,dOutput,dLock);

	cudaDeviceSynchronize();

	cudaMemcpy(hOutput, dOutput, outputSize, cudaMemcpyDeviceToHost);

	for(int i=0;i<n;i++){
		cout<<hOutput[4*i+1]<<" "<<hOutput[4*i+2]<<" "<<hOutput[4*i+3]<<"\n";
	}

	cudaFree(dOutput);
	cudaFree(dData);
	cudaFree(dQuery);
	cudaFree(dLock);
	return 0;
}
