// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli -
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli -
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli -
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli -

#include <stdio.h>
#include "test_utils.h"

void __enzyme_autodiff(...);

template <typename... Args>
double __enzyme_autodiff(void *, Args...);

int enzyme_dup, enzyme_const, enzyme_out, enzyme_dupnoneed;

struct Object
{
    virtual double eval(double v) = 0;
    double val;
};

struct Object1 : Object
{
    double eval(double v)
    {
        return val * v;
    }
};

struct Object2 : Object
{
    double eval(double v)
    {
        return val + v;
    }
};

double eval(Object &o, double v)
{
    return o.eval(v);
}

double deval(Object &o, Object &d_o, double v)
{
    return __enzyme_autodiff((void *)(eval),
                      enzyme_dup, &o, &d_o,
                      v);
}

template<typename T>
T __enzyme_virtualreverse(T);

int main()
{
    double v = 10;
    Object *o = new Object1();
    o->val = 2;
    Object *d_o = new Object1();
    *((void**)d_o) = __enzyme_virtualreverse(*(void**)d_o);
    d_o->val = 0;
    double res;
    res = deval(*o, *d_o, v);
    printf("res=%f\n", res);
    APPROX_EQ(res, 2.0, 1e-7);
    
    o = new Object2();
    o->val = 2;
    d_o = new Object2();
    *((void**)d_o) = __enzyme_virtualreverse(*(void**)d_o);
    d_o->val = 0;
    res = deval(*o, *d_o, v);
    printf("res=%f\n", res);
    APPROX_EQ(res, 1.0, 1e-7);
}
