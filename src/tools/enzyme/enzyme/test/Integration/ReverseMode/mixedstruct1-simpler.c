// RUN: %clang -std=c11 %O0TBAA %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -std=c11 -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -std=c11 -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -std=c11 -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -std=c11 %O0TBAA %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli - 
// RUN: %clang -std=c11 -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli - 
// RUN: %clang -std=c11 -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli - 
// RUN: %clang -std=c11 -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli - 

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "test_utils.h"

#define __builtin_autodiff __enzyme_autodiff
double __enzyme_autodiff(void*, ...);


typedef struct {
  int i;
  float value1;
  float value2;
} structtest2;


typedef struct {
  int R_start;
  int R_end;
  int C_start;
  int C_end;
  float* data;
} WINDOW_FRAME;

float tile_multiply(float* data, int R_start, int R_end, int C_start, int C_end) {
  int R_midpoint = (R_start) + (R_end-R_start)/2;
  int C_midpoint = (C_start) + (C_end-C_start)/2;

  if (R_end - R_start > 2) {
      float sum1 = tile_multiply(data, R_start, R_midpoint, C_start, C_end);
      float sum2 = tile_multiply(data, R_midpoint, R_end, C_start, C_end);
      return sum1 + sum2;
  } else if (C_end - C_start > 2) {
      float sum1 = tile_multiply(data, R_start, R_end, C_start, C_midpoint);
      float sum2 = tile_multiply(data, R_start, R_end, C_midpoint, C_end);
      return sum1 + sum2;
  } else {
    float sum = 0.0;
    for (int r = R_start; r < R_end; r++) {
      for (int c = C_start; c < C_end; c++) {
        printf("data[%d*10 + %d]\n", r, c);
        sum += data[r*10 + c];//*window[(r+c)%10];
      }
    }
    return sum;
  }
}

int main(int argc, char** argv) {

  float* data = (float*) malloc(sizeof(float) * 10*10);
  float* d_data = (float*) malloc(sizeof(float) * 10*10);
  for (int i = 0; i < 10*10; i++) {
    data[i] = 1.0;
    d_data[i] = 0.0;
  }

  float loss = 0.0;
  float d_loss = 1.0;


  int R_start = 0;
  int R_end = 10;
  int C_start = 0;
  int C_end = 10;

  //tile_multiply_helper(window, frame, &loss);

  __enzyme_autodiff(tile_multiply, data, d_data, R_start, R_end, C_start, C_end); //frame, d_frame, &loss, &d_loss);

  for (int i = 0; i < 100; i++) {
    printf("gradient for %d is %f\n", i, d_data[i]);
    APPROX_EQ(d_data[i], 1.0, 1e-10);
  }

  return 0;
}
