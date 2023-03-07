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


__attribute__((weak)) void compute_values_struct_passbyval(structtest2* value, float* loss) /*__attribute__((noinline))*/{
  *loss += value->value1*value->value2;
}

float tile_multiply(float* window, WINDOW_FRAME* frame) {
  if (frame->R_end - frame->R_start > 4 || frame->C_end - frame->C_start > 4) {
    int R_midpoint = (frame->R_start) + (frame->R_end-frame->R_start)/2;
    int C_midpoint = (frame->C_start) + (frame->C_end-frame->C_start)/2;

    if (frame->R_end - frame->R_start > frame->C_end - frame->C_start) {

      WINDOW_FRAME* frame1 = malloc(sizeof(WINDOW_FRAME));
      WINDOW_FRAME* frame2 = malloc(sizeof(WINDOW_FRAME));
      *frame1 = *frame;
      *frame2 = *frame;
      frame1->R_end = R_midpoint;
      frame2->R_start = R_midpoint;

      float sum1 = tile_multiply(window, frame1);
      float sum2 = tile_multiply(window, frame2);
      free(frame1);
      free(frame2);
      return sum1 + sum2;
    } else {
      WINDOW_FRAME* frame1 = malloc(sizeof(WINDOW_FRAME));
      WINDOW_FRAME* frame2 = malloc(sizeof(WINDOW_FRAME));
      *frame1 = *frame;
      *frame2 = *frame;
      frame1->C_end = C_midpoint;
      frame2->C_start = C_midpoint;

      float sum1 = tile_multiply(window, frame1);
      float sum2 = tile_multiply(window, frame2);
      free(frame1);
      free(frame2);
      return sum1 + sum2;
    }
  } else {
    float sum = 0.0;
    for (int r = frame->R_start; r < frame->R_end; r++) {
      for (int c = frame->C_start; c < frame->C_end; c++) {
        sum += frame->data[r*10 + c];//*window[(r+c)%10];
      }
    }
    return sum;
  }
}

void tile_multiply_helper(float* window, WINDOW_FRAME* frame, float* loss) {
  *loss = tile_multiply(window, frame);
}

int main(int argc, char** argv) {

  float* data = (float*) malloc(sizeof(float) * 10*10);
  float* d_data = (float*) malloc(sizeof(float) * 10*10);
  for (int i = 0; i < 10*10; i++) {
    data[i] = 1.0;
    d_data[i] = 0.0;
  }

  float* window = (float*) malloc(sizeof(float) * 10);
  float* d_window = (float*) malloc(sizeof(float) * 10);
  for (int i = 0; i < 10; i++) {
    window[i] = 2.0;
    d_window[i] = 0.0;
  }

  float loss = 0.0;
  float d_loss = 1.0;


  int R_start = 0;
  int R_end = 10;
  int C_start = 0;
  int C_end = 10;

  WINDOW_FRAME* frame = (WINDOW_FRAME*) malloc(sizeof(WINDOW_FRAME));
  WINDOW_FRAME* d_frame = (WINDOW_FRAME*) malloc(sizeof(WINDOW_FRAME));
  frame->R_start = R_start;
  frame->R_end = R_end;
  frame->C_start = C_start;
  frame->C_end = C_end;
  frame->data = data;

  *d_frame = *frame;
  d_frame->data = d_data; 

  //tile_multiply_helper(window, frame, &loss);

  __enzyme_autodiff(tile_multiply_helper, window, d_window, frame, d_frame, &loss, &d_loss);

  for (int i = 0; i < 100; i++) {
    printf("gradient for %d is %f\n", i, d_frame->data[i]);
    APPROX_EQ(d_frame->data[i], 1.0, 1e-10);
  }

  return 0;
}
