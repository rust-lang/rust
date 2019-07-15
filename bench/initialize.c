#include <math.h>

__attribute__((noinline))
void allocateAndSet(double ** arrayp, const double x, unsigned int n) {
    *arrayp = (double*)malloc(sizeof(double)*n);
    (*arrayp)[3] = x;
}

__attribute__((noinline))
double get(double* x, unsigned int i) {
    return x[i];
}

double function(const double x, unsigned int n) {
    double *array;
    allocateAndSet(&array, x, n);
    return get(array, 3);
}

__attribute__((noinline))
double derivative(const double x, unsigned int n) {
    return __builtin_autodiff(function, x, n);
}

#include <stdio.h>
#include <stdlib.h>
int main(int argc, char** argv) {
    double x = atof(argv[1]);
    double n = atof(argv[2]);
    double xp = derivative(x, n);
    printf("x=%f xp=%f\n", x, xp);
}
