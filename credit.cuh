#ifndef CREDIT_H
#define CREDIT_H

#include <cuda_runtime.h>
#include <math.h>
#include <curand_kernel.h>
#include <random>

class Credit
{
    double PD, EAD, LGD, FG, FL, FI, rho;

public:
    Credit();
    Credit(double PD, double EAD, double LGD, double FG, double FL);
    ~Credit();

    __device__ double loss(double rG, double rL, curandState * state);
};

class Portfolio
{
    Credit * carte;
    int n;

public:
    __device__ Portfolio(Credit * cartera, int n);
    __device__ ~Portfolio();

    __device__ double loss(double fg, double fl, curandState * state);
};



#endif // CREDIT_H
