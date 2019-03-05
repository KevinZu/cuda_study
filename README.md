# cuda_study
This is just a cuda programming exercise

## 第一个cuda程序：basic.cu
```
nvcc -gencode=arch=compute_50,code=\"sm_50,compute_50\" -o basic basic.cu
```


## 矩阵乘法：MatrixMul.cu



```
nvcc -gencode=arch=compute_50,code=\"sm_50,compute_50\" -o MatrixMul MatrixMul.cu
```
