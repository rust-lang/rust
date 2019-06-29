#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdlib.h>
#include <math.h>
#include <inttypes.h>
#include <string.h>

extern int diffe_const;

float tdiff(struct timeval *start, struct timeval *end) {
  return (end->tv_sec-start->tv_sec) + 1e-6*(end->tv_usec-start->tv_usec);
}

#include <adept_source.h>
#include <adept_arrays.h>
using adept::adouble;
using adept::aMatrix;
using adept::aVector;

using adept::Vector;

static double sum(const double *x, size_t n) {
    double res = 0;
    for(int i=0; i<n; i++) {
        res+=x[i];
    }
    return res;
}

static double max(double x, double y) {
    return (x > y) ? x : y;
}

static adouble amax(adouble x, adouble y) {
    return (x > y) ? x : y;
}

static double logsumexp(const double *__restrict x, size_t n) {
  double A = x[0];
  for(int i=0; i<n; i++) {
    A = max(A, x[i]);
  }
  double sema = 0;
  for(int i=0; i<n; i++) {
    sema += exp(x[i] - A);
  }
  return log(sema) + A;
}


static adouble alogsumexp(const aVector &x, size_t n) {
  adouble A = x[0];
  for(int i=0; i<n; i++) {
    A = amax(A, x[i]);
  }
  adouble ema[n];
  for(int i=0; i<n; i++) {
    ema[i] = exp(x[i] - A);
  }
  adouble sema = 0;
  for(int i=0; i<n; i++)
    sema += ema[i];
  return log(sema) + A;
}

static adouble alogsumexp2(const aVector &x, size_t n) {
  adouble A = x[0];
  for(int i=0; i<n; i++) {
    A = amax(A, x[i]);
  }
  return adept::log(adept::sum(exp(x - A))) + A;
}

static void adept_sincos(double *input, double *inputp, unsigned long n, unsigned long repeat) {
  {
  struct timeval start, end;
  //gettimeofday(&start, NULL);
 
  adept::Stack stack;
 
  aVector inp(n);
  for(int i=0; i<n; i++) inp(i) = input[i];
  memset(inputp, 0, sizeof(double)*n);
  double total = 0;

  gettimeofday(&start, NULL);
  for (int iter = 0; iter < repeat; iter++) {
    stack.new_recording();
    adouble resa = alogsumexp(inp, n);
    stack.pause_recording();
    total += resa.value();
    stack.continue_recording();
  }
  gettimeofday(&end, NULL);

  stack.pause_recording();

  printf("adept forward (recording) %0.6f res'=%f\n", tdiff(&start, &end), total);
  }
  {
  struct timeval start, end;
  //gettimeofday(&start, NULL);
 
  adept::Stack stack;
 
  aVector inp(n);
  for(int i=0; i<n; i++) inp(i) = input[i];
  memset(inputp, 0, sizeof(double)*n);

  gettimeofday(&start, NULL);
  for (int iter = 0; iter < repeat; iter++) {
    stack.new_recording();
    adouble resa = alogsumexp(inp, n);
    resa.set_gradient(1.0);
    stack.reverse();
    stack.pause_recording();
    for (int i = 0; i < n; i++) {
        inputp[i] += inp(i).get_gradient();
    }
    stack.continue_recording();
  }
  gettimeofday(&end, NULL);

  stack.pause_recording();

  printf("adept forward reverse %0.6f res'=%f\n", tdiff(&start, &end), sum(inputp, n));
  }
}
static void adept2_sincos(double *input, double *inputp, unsigned long n, unsigned long repeat) {
  {
  struct timeval start, end;
  //gettimeofday(&start, NULL);
 
  adept::Stack stack;
 
  aVector inp(n);
  for(int i=0; i<n; i++) inp(i) = input[i];
  memset(inputp, 0, sizeof(double)*n);
  double total = 0;

  gettimeofday(&start, NULL);
  for (int iter = 0; iter < repeat; iter++) {
    stack.new_recording();
    adouble resa = alogsumexp2(inp, n);
    stack.pause_recording();
    total += resa.value();
    stack.continue_recording();
  }
  gettimeofday(&end, NULL);

  stack.pause_recording();

  printf("adept2 forward (recording) %0.6f res'=%f\n", tdiff(&start, &end), total);
  }
  {
  struct timeval start, end;
  //gettimeofday(&start, NULL);
 
  adept::Stack stack;
 
  aVector inp(n);
  for(int i=0; i<n; i++) inp(i) = input[i];
  memset(inputp, 0, sizeof(double)*n);

  gettimeofday(&start, NULL);
  for (int iter = 0; iter < repeat; iter++) {
    stack.new_recording();
    adouble resa = alogsumexp2(inp, n);
    resa.set_gradient(1.0);
    stack.reverse();
    stack.pause_recording();
    for (int i = 0; i < n; i++) {
        inputp[i] += inp(i).get_gradient();
    }
    stack.continue_recording();
  }
  gettimeofday(&end, NULL);

  stack.pause_recording();

  printf("adept2 forward reverse %0.6f res'=%f\n", tdiff(&start, &end), sum(inputp, n));
  }
}

static void my_sincos(double *input, double *inputp, unsigned long n, unsigned long repeat) {
    double realinput = input[0];
  {
  struct timeval start, end;
  gettimeofday(&start, NULL);
  double total = 0;
  for(int i=0; i<repeat; i++) {
    input[0] = realinput + (double)i/10000000;
    total += logsumexp(input, n);
  }

  gettimeofday(&end, NULL);
  printf("enzyme forward %0.6f res'=%f\n", tdiff(&start, &end), total);
  }
  {
      input[0] = realinput;
  struct timeval start, end;
  memset(inputp, 0, sizeof(double)*n);

  gettimeofday(&start, NULL);

  for(int i=0; i<repeat; i++) {
    __builtin_autodiff(logsumexp, input, inputp, n);
  }

  gettimeofday(&end, NULL);
  printf("enzyme forward and reverse %0.6f res'=%f\n", tdiff(&start, &end), sum(inputp, n));
  }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("usage %s n repeat\n", argv[0]);
        return 1;
    }
  unsigned long n = atoi(argv[1]);
  unsigned long repeat = atoi(argv[2]);

  double *input = new double[n];
  double *inputp = new double[n];
  for(int i=0; i<n; i++) {
    input[i] = 3.1415926535 / (i+1);
  }
  
  //adept_sincos(input, inputp, n, repeat);
  
  adept2_sincos(input, inputp, n, repeat);

  my_sincos(input, inputp, n, repeat);
}

