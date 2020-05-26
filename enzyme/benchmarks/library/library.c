#include <stdio.h>

#include "mylib.h"

void __enzyme_autodiff(void*, ...);


void f(double* in, double* out) {
  set(in, out);
}





int main(int argc, char** argv) {
  double in = 2;
  double out = 3;
  double din = 0.0;
  double dout = 1.0;

  __enzyme_autodiff(f, &in, &din, &out, &dout);

  printf("in=%f din=%f out=%f, dout=%f\n", in, din, out, dout);

}
