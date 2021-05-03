#include <iostream>
#include "credit.cuh"
#include <chrono>

extern double * credit_simulation(Credit * cartera, int c, int n);

int main(int argc, char *argv[])
{
    int n = 1e2;

    unsigned long l = 2e4;
    Credit * counter = new Credit[l];

    for (unsigned long  i = 0; i < l; i++)
    {
        counter[i] = Credit(0.1, 1, 0.3, 0.14, 0.2);
    }

    auto dx = std::chrono::high_resolution_clock::now();

    double * out = credit_simulation(counter, l, n);

    auto dy = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dif = dy - dx;
    std::cout << dif.count() << " seconds" << std::endl;

    for (int i = 0; i < n; i++)
    {
        std::cout << out[i] << std::endl;
    }

    delete [] out;
    delete [] counter;


    return 0;
}

