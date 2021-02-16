// RUN: %clang++ -ffast-math -O0 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang++ -ffast-math -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang++ -ffast-math -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang++ -ffast-math -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang++ -ffast-math -O0 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli - 
// RUN: %clang++ -ffast-math -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli - 
// RUN: %clang++ -ffast-math -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli - 
// RUN: %clang++ -ffast-math -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli - 

//#include <math.h>

#include "test_utils.h"

#include <stdio.h>
#include <stdlib.h>

#include <math.h>
#include <complex>      // std::complex, std::abs, std::arg

int enzyme_dup;
int enzyme_out;
int enzyme_const;

void __enzyme_autodiff(...);

using namespace std;

double h( const complex<double>& c )
{
  //double theta = c.real()+3*c.imag();
  double theta = std::abs(c);
  //double theta = arg(c);
  return theta * theta;
}

double h2( const double& c )
{
  double theta = std::abs(c);
  return theta * theta;
}

int main ()
{
  std::complex<double> mycomplex (3.0,4.0);

  std::complex<double> dc(0.0,0.0);

  double x = -3.0;
  double dx = 0.0;
  //Works with real number
  __enzyme_autodiff(&h2, enzyme_dup, &x,&dx);

  APPROX_EQ(dx, -6.0, 1e-7);

  //Compilation fails for complex number
  //"attempting to differentiate function without definition"' failed.
  __enzyme_autodiff(&h, enzyme_dup, &mycomplex,&dc);

  printf("grad energy of particle is %f, %f\n", real(dc), imag(dc));

  APPROX_EQ(real(dc), 6.0, 1e-7);
  APPROX_EQ(imag(dc), 8.0, 1e-7);

  return 0;
}