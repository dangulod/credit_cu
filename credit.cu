#include "credit.cuh"

__device__ double RationalApproximation(double t)
{
    double c[] = {2.515517, 0.802853, 0.010328};
    double d[] = {1.432788, 0.189269, 0.001308};
    return t - ((c[2]*t + c[1])*t + c[0]) /
               (((d[2]*t + d[1])*t + d[0])*t + 1.0);
}

__device__ double qNorm(double p)
{
    if ( (p < 0.5) & (p > 0))
    {
        return -RationalApproximation( sqrt(-2.0*log(p)) );
    }
    else if ( (p >= 0.5) & (p < 1))
    {
        return RationalApproximation( sqrt(-2.0*log(1-p)) );
    }
}

Credit::Credit() {}

Credit::~Credit() {}

Credit::Credit(double PD, double EAD, double LGD, double FG, double FL) : PD(PD), EAD(EAD), LGD(LGD), FG(FG), FL(FL)
{
    this->rho = pow(FG, 2) + pow(FL, 2);
    this->FI  = sqrt( 1 - this->rho );
}

__device__ double Credit::loss(double rG, double rL, curandState * state)
{
    double Y = sqrt(this->rho) * ( this->FG * rG + FL * rL ) + sqrt( 1 - this->rho ) * curand_normal(state);
    return (qNorm(this->PD) >= Y) * EAD * LGD;
}

__device__ Portfolio::Portfolio(Credit *cartera, int n): carte(cartera), n(n) {}

__device__ Portfolio::~Portfolio() {}

__device__ double Portfolio::loss(double fg, double fl, curandState *state)
{
    double loss = 0;
    for (int i = 0; i < this->n; i++)
    {
        loss += this->carte[i].loss(fg, fl, state);
    }
    return loss;
}
