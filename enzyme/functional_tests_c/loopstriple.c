#include <stdio.h>
#include <math.h>
#include <assert.h>

#define __builtin_autodiff __enzyme_autodiff
double __enzyme_autodiff(void*, ...);
void compute_loops(float* a, float* b, float* ret) {
  double sum0 = 0.0;
  for (int i = 0; i < 100; i++) {
    double sum1 = 0.0;
    for (int j = 0; j < 100; j++) {
      double sum2 = 0.0;
      for (int k = 0; k < 100; k++) {
        sum2 += *a+*b;
      }
      sum1 += sum2;
    }
    sum0 += sum1;
  }
  *ret = sum0;
}



int main(int argc, char** argv) {



  float a = 2.0;
  float b = 3.0;



  float da = 0;//(float*) malloc(sizeof(float));
  float db = 0;//(float*) malloc(sizeof(float));


  float ret = 0;
  float dret = 1.0;

  //compute_loops(&a, &b, &ret);

  __builtin_autodiff(compute_loops, &a, &da, &b, &db, &ret, &dret);


  assert(da == 100*100*100*1.0f);
  assert(db == 100*100*100*1.0f);

  printf("hello! %f, res2 %f, da: %f, db: %f\n", ret, ret, da,db);
  return 0;
}
