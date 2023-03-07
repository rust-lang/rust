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


typedef float* WINDOW_FRAME;


float tile_multiply(WINDOW_FRAME* frame, int C_start, int C_end) {
    if ( C_end - C_start > 3) {
      int C_midpoint = C_start + (C_end-C_start)/2;
      WINDOW_FRAME* frame1 = malloc(sizeof(WINDOW_FRAME));
      *frame1 = *frame;

      float sum1 = tile_multiply(frame1, C_start, C_midpoint);
      float sum2 = tile_multiply(frame1, C_midpoint, C_end);
      free(frame1);
      return sum1 + sum2;
  } else {
    float sum = 0.0;
    long long int c = C_start;
    if (c < C_end)
    do{
        sum += (*frame)[c];//*window[(r+c)%10];
        c++;
      }while(c < C_end);
    return sum;
  }
}

void tile_multiply_helper(WINDOW_FRAME* frame, float* loss) {
  *loss = tile_multiply(frame, 0, 10);
}

int main(int argc, char** argv) {

  float* data = (float*) malloc(sizeof(float) * 10);
  float* d_data = (float*) malloc(sizeof(float) * 10);
  for (int i = 0; i < 10; i++) {
    data[i] = 1.0;
    d_data[i] = 0.0;
  }

  float loss = 0.0;
  float d_loss = 1.0;


  int C_start = 0;

  WINDOW_FRAME* frame = (WINDOW_FRAME*) malloc(sizeof(WINDOW_FRAME));
  WINDOW_FRAME* d_frame = (WINDOW_FRAME*) malloc(sizeof(WINDOW_FRAME));
  *frame = data;

  *d_frame = d_data; 

  //tile_multiply_helper(window, frame, &loss);

  __enzyme_autodiff(tile_multiply_helper, frame, d_frame, &loss, &d_loss);

  for (int i = 0; i < 10; i++) {
    printf("gradient for %d is %f\n", i, d_data[i]);
    APPROX_EQ(d_data[i], 1.0, 1e-10);
  }

  return 0;
}
