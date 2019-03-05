#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
using namespace std;

void MatrixPrint(float *mat, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            cout << setw(2) << mat[i*cols+j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

__global__ void addone(float *a) {
    int tix = threadIdx.x;
    int tiy = threadIdx.y;
    int bdx = blockDim.x;
    int bdy = blockDim.y;
    a[tix*bdy+tiy] += 1;
}
int main()
{
    int size = 5;
    float *a = (float*)malloc(sizeof(float)*size*size);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            a[i*size+j] = 1.0f;
        }
    }
    MatrixPrint(a,size,size);
    float *a_cuda;
    cudaMalloc((void**)&a_cuda,sizeof(float)*size*size);
    cudaMemcpy(a_cuda,a,sizeof(float)*size*size,cudaMemcpyHostToDevice);

    dim3 grid(1, 1, 1), block(5, 5, 1);
    addone<<<grid,block>>>(a_cuda);
    cudaMemcpy(a,a_cuda,sizeof(float)*size*size,cudaMemcpyDeviceToHost);
    MatrixPrint(a,size,size);
    return 0;
}
