/*
!pip install git+https://github.com/andreinechaev/nvcc4jupyter.git
%load_ext nvcc_plugin
*/

// %%cu

#include <bits/stdc++.h>
#include <cuda.h>
using namespace std;

const int total = 9;
const int row1 = 4;
const int col1 = 5;
const int col2 = 3;

__global__ void matrixMultiplication(int *mat1, int *mat2, int *ans)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < row1 && col < col2)
    {
        int sum = 0;
        for (int i = 0; i < col1; ++i)
        {
            sum += mat1[row * col1 + i] + mat2[i * col2 + col];
        }
        ans[row * col2 + col] = sum;
    }
}

void generateRandomMatrix(int *matrix, int row, int col)
{
    for (int r = 0; r < row; ++r)
    {
        for (int c = 0; c < col; ++c)
        {
            matrix[r * col + c] = rand() % 10;
        }
    }
}

void printMatrix(int *matrix, int row, int col, int id)
{

    cout << "Result[" << id << "] =\n";
    for (int r = 0; r < row; ++r)
    {
        for (int c = 0; c < col; ++c)
        {
            cout << matrix[r * col + c] << "\t";
        }
        cout << endl;
    }
    cout << "\n\n";
}

int main()
{
    // Host (CPU) matrix
    int *hMat1;
    int *hMat2;
    int *hAns;

    // Host memory allocation
    hMat1 = (int *)malloc(total * row1 * col1 * sizeof(int));
    hMat2 = (int *)malloc(total * col1 * col2 * sizeof(int));
    hAns = (int *)malloc(total * row1 * col2 * sizeof(int));

    srand(time(nullptr));

    // Random Matrix Generation
    for (int i = 0; i < total; ++i)
    {
        generateRandomMatrix(hMat1 + i * row1 * col1, row1, col1);
        generateRandomMatrix(hMat2 + i * col1 * col2, col1, col2);
    }

    // Device (GPU) matrix
    int *dMat1;
    int *dMat2;
    int *dAns;

    // Device memory allocation
    cudaMalloc(&dMat1, total * row1 * col1 * sizeof(int));
    cudaMalloc(&dMat2, total * col1 * col2 * sizeof(int));
    cudaMalloc(&dAns, total * row1 * col2 * sizeof(int));

    cudaEvent_t startTime;
    cudaEventCreate(&startTime);

    cudaEvent_t endTime;
    cudaEventCreate(&endTime);

    // Copy data from Host to Device
    cudaMemcpy(dMat1, hMat1, total * row1 * col1 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dMat2, hMat2, total * col1 * col2 * sizeof(int), cudaMemcpyHostToDevice);

    // Define grid dimension & block dimension of thread
    dim3 blockDim(16, 16);
    dim3 gridDim((col2 + blockDim.x - 1) / blockDim.x, (row1 + blockDim.y) / blockDim.y);

    cudaEventRecord(startTime);

    // Matrix multiplication kernel
    for (int i = 0; i < total; ++i)
    {
        matrixMultiplication<<<gridDim, blockDim>>>(dMat1 + i * row1 * col1, dMat2 + i * col1 * col2, dAns + i * row1 * col2);
    }

    cudaEventRecord(endTime);
    cudaEventSynchronize(endTime);

    // Copy the result back to the host
    cudaMemcpy(hAns, dAns, total * row1 * col2 * sizeof(int), cudaMemcpyDeviceToHost);

    float timeTaken = 0;
    cudaEventElapsedTime(&timeTaken, startTime, endTime);
    cout << "Time taken  to execute the Matrix Multiplication: " << timeTaken << "ms\n\n";

    //... Print the result
    for (int i = 0; i < total; i++)
    {
        printMatrix(hAns + i * row1 * col2, row1, col2, i);
    }

    cudaFree(dMat1);
    cudaFree(dMat2);
    cudaFree(dAns);
    free(hMat1);
    free(hMat2);
    free(hAns);

    return 0;
}
