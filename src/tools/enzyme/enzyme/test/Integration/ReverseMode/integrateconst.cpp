// A note to a potential future debugger, the -fno-use-cxa-atexit is needed solely for lli (the llvm interpreter) to work
//   Actually compiling it works fine, and moreover, the one place atexit is used is nowhere near the enzyme code

// RUN: %clang++ -fno-use-cxa-atexit -ffast-math -mllvm -force-vector-width=1 -ffast-math -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fno-exceptions -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang++ -fno-use-cxa-atexit -ffast-math -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fno-exceptions -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang++ -Xclang -new-struct-path-tbaa -fno-use-cxa-atexit -ffast-math -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fno-exceptions -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang++ -Xclang -new-struct-path-tbaa -fno-use-cxa-atexit -ffast-math -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fno-exceptions %O0TBAA %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang++ -fno-use-cxa-atexit -ffast-math -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fno-exceptions -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli - 
// RUN: %clang++ -fno-use-cxa-atexit -ffast-math -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fno-exceptions -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli - 
// RUN: %clang++ -fno-use-cxa-atexit -ffast-math -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fno-exceptions -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli -
// RUN: %clang++ -fno-use-cxa-atexit -ffast-math -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fno-exceptions %O0TBAA %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli - 

#include "test_utils.h"

#define BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
#define BOOST_NO_EXCEPTIONS
#include <iostream>
#include <boost/array.hpp>

#include <boost/numeric/odeint.hpp>

#include <boost/throw_exception.hpp>
void boost::throw_exception(std::exception const & e){
    //do nothing
}

using namespace std;
using namespace boost::numeric::odeint;

#include <stdio.h>

typedef boost::array< double , 1 > state_type;

void lorenz( const state_type &x , state_type &dxdt , double t )
{
    const double a = 1.2;
    dxdt[0] = -a * x[0];
}

    
double foobar(double t=10.0) {
    state_type x = { 1.0 }; // initial conditions

    //typedef controlled_runge_kutta< runge_kutta_dopri5< state_type , typename state_type::value_type , state_type , double > > stepper_type;
    typedef euler< state_type , typename state_type::value_type , state_type , double > stepper_type;
    size_t steps = integrate_const( stepper_type(), lorenz , x , 0.0 , t, t/3 );

    // x -a x t == x(1-at) = (1-1.2t) = (1 - 1.2) == -.2
    
    printf("final result t=%f x(t)=%f, -0.2=%f, steps=%zu\n", t, x[0], -.2, steps);
    return x[0];
}

extern "C" {
extern double __enzyme_autodiff(void*, double);
}

int main(int argc, char **argv)
{

    int n = 3;
    double a = 1.2;

    for(int i=1; i<=100; i++) {
        double t=i/10.;
        double mreal = -a;
        for(int i=0; i<n-1; i++) {
            mreal *= 1 - a * t/n;
        }
        double res = __enzyme_autodiff((void*)foobar, t);
        printf("t=%f d/dt(exp(-1.2*t))=%f, -1.2*exp(-1.2*t)=%f\n", t, res, mreal);
        // see if approximation is within 10%
        APPROX_EQ(res, mreal, max(fabs(mreal)/10., 2.0e-5) );
    }
}
