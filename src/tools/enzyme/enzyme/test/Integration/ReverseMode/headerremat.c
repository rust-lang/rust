// RUN: %clang -std=c11 -O0 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -std=c11 -fno-unroll-loops -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -std=c11 -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -std=c11 -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -std=c11 -O0 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme --enzyme-inline=1 -S | %lli - 
// RUN: %clang -std=c11 -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme --enzyme-inline=1 -S | %lli - 
// RUN: %clang -std=c11 -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme --enzyme-inline=1 -S | %lli - 
// RUN: %clang -std=c11 -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme --enzyme-inline=1 -S | %lli - 

#include <stdio.h>
#include <math.h>
#include <assert.h>

#include "test_utils.h"


#include <stdio.h>

__attribute__((noinline))
int evaluate_integrand(const int nr,
    const int dtheta)
{
    return nr * dtheta;
}

double integrate_image(double dr, double* out)
{
    double dtheta = 1;

    {
        double I_estimate;

        for (int k=0; k<10; k++)
        {

            int nr = (int)(dr);
            int ntheta = (int)(dtheta);

            double sum = evaluate_integrand(nr, ntheta);

            out[k] *= nr * ntheta;
            printf("dtheta=%d\n", nr * ntheta);
            I_estimate = sum * dr;

            // Update the step size
            dr /= 0.8;
            dtheta = dr / 4.0;
        }
        return I_estimate;
    }
}

void __enzyme_autodiff(double (*)(double, double*), ...);

int main()
{
    double out[10];
    double d_out[10];
    for(int i=0; i<10; i++)
        d_out[i] = 1.0;

    int answer[10] = {
        200,
        15500,
        24336,
        37830,
        59536,
        92720,
        144780,
        226814,
        355216,
        554280
    };

    __enzyme_autodiff(integrate_image, 200.0, out, d_out);

    for (int i = 0; i < 10; i++)
    {
        printf("dout[%d]=%f answer=%d\n", i, d_out[i], answer[i]);
        APPROX_EQ(d_out[i], answer[i], 1e-6);
    }

    return 0;
}
