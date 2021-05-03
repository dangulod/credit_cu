#ifndef __CREDIT_CU__
#define __CREDIT_CU__

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>
#include "credit.cuh"
#include <iostream>

void handleCudaError(cudaError_t cudaERR)
{
    if (cudaERR != cudaSuccess)
    {
        printf("CUDA ERROR : %s\n", cudaGetErrorString(cudaERR));
    }
}


__global__ void simulation(double * d_out, Credit * cartera, int c, int n)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    curandState state;
    curand_init(12345 + i, 0, 0, &state);

    Portfolio p(cartera, c);

    if (i < n)
    {
        double fg = curand_normal(&state);
        double fl = curand_normal(&state);

        d_out[i] += p.loss(fg, fl, &state);

        i += gridDim.x * blockDim.x;
    }
}

double * credit_simulation(Credit * cartera, int c, int n)
{
    double * out, * d_out;
    Credit * d_cartera;

    size_t size_d = sizeof(double) * n;
    size_t size_c = sizeof(Credit) * c;

    out = new double[n];
    handleCudaError(cudaMalloc(&d_out, size_d));
    handleCudaError(cudaMalloc(&d_cartera, size_c));

    handleCudaError(cudaMemcpy(d_cartera, cartera, size_c, cudaMemcpyHostToDevice));

    simulation<<<0xFFFF, 1024>>>(d_out, d_cartera, c, n);

    handleCudaError(cudaMemcpy(out, d_out, size_d, cudaMemcpyDeviceToHost));

    handleCudaError(cudaFree(d_cartera));
    handleCudaError(cudaFree(d_out));

    return out;
}

#endif
