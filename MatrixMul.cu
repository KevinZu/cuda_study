#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
using namespace std;



void MatrixMul_host(float *a, int a_rows, int a_cols, float *b, int b_rows, int b_cols, float *c) {
    for (int i = 0; i < a_rows; i++) {
        for (int j = 0; j < b_cols; j++) {
            float t = 0;
            for (int k = 0; k < b_rows; k++) {
                t += a[i*a_cols+k]*b[k*b_cols+j];
            }
            c[i*b_cols+j] = t;
        }
    }
}


void MatrixRandBin(float *mat, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if ((float)rand()/RAND_MAX > 0.5) {
                mat[i*cols+j] = 1.0f;
            }else {
                mat[i*cols+j] = -1.0f;
            }
        }
    }
}


float MatrixCompare(float *a,float *b,int rows,int cols){
    float err=0;
    for (int i=0;i<rows;i++){
        for (int j=0;j<cols;j++){
            err+=abs(a[i*cols+j]-b[i*cols+j]);  
        }
    }
    return err;
}


__global__ void MatrixMul_device(float *a, int a_rows, int a_cols, float *b, int b_rows, int b_cols, float *c) {
    int tix = threadIdx.x;
    int tiy = threadIdx.y;
    int bix = blockIdx.x;
    int biy = blockIdx.y;
    int bdx = blockDim.x;
    int bdy = blockDim.y;
    int gdx = gridDim.x;
    int gdy = gridDim.y;

    for (int i = tix; i < b_cols; i += bdx) {
        float sum = 0;
        for (int k = 0; k < a_cols; k++) {
            sum += a[bix*a_rows+k]*b[k*b_cols+i];
        }
        c[bix*a_cols+i] = sum;
    }
}



int main()
{
	int Matrixsize=1000;
	float *a_host;
	float *a_device;
	float *b_host;
	float *b_device;
	float *result_host;
	float *result_device;
	float *result_cpu;
	a_host = (float*) malloc(sizeof(float) * Matrixsize * Matrixsize);
	b_host = (float*) malloc(sizeof(float) * Matrixsize * Matrixsize);
	result_host = (float*) malloc(sizeof(float) * Matrixsize * Matrixsize);
	result_cpu = (float*) malloc(sizeof(float) * Matrixsize * Matrixsize);
	srand(0);
	MatrixRandBin(a_host,Matrixsize,Matrixsize);
	MatrixRandBin(b_host,Matrixsize,Matrixsize);
	cudaMalloc((void**)&a_device,sizeof(float) *Matrixsize * Matrixsize);
	cudaMalloc((void**)&b_device,sizeof(float) *Matrixsize * Matrixsize);
	cudaMalloc((void**)&result_device,sizeof(float) *Matrixsize * Matrixsize);
	cudaMemcpy(a_device,a_host,sizeof(float) *Matrixsize * Matrixsize,cudaMemcpyHostToDevice);
	cudaMemcpy(b_device,b_host,sizeof(float) *Matrixsize * Matrixsize,cudaMemcpyHostToDevice);


	cudaEvent_t start_device, stop_device;
	float time_device;
	cudaEventCreate(&start_device);
	cudaEventCreate(&stop_device);
	cudaEventRecord( start_device, 0 );
	dim3 gridsize(1000,1,1);
	dim3 blocksize(256,1,1);
	MatrixMul_device<<<gridsize,blocksize>>>(a_device,Matrixsize,Matrixsize,b_device,Matrixsize,Matrixsize,result_device);
	cudaEventRecord( stop_device, 0 );
	cudaEventSynchronize( stop_device );
	cudaEventElapsedTime( &time_device, start_device, stop_device );
	cudaEventDestroy( start_device );
	cudaEventDestroy( stop_device );
	cout<<"gputime="<<time_device<<"ms"<<endl;

	cudaMemcpy(result_host, result_device,sizeof(float) *Matrixsize * Matrixsize,cudaMemcpyDeviceToHost);
	cudaFree(a_device);
	cudaFree(b_device);
	cudaFree(result_device);
	clock_t start_host = clock();
	MatrixMul_host(a_host,Matrixsize,Matrixsize,b_host,Matrixsize,Matrixsize,result_cpu);
	cout<<"cputime="<<(double)(clock() - start_host)/1000<<"ms"<<endl;
	float err=MatrixCompare(result_cpu,result_host,Matrixsize,Matrixsize);
	cout<<"err in gpu and cpu = "<<err<<endl;
}

