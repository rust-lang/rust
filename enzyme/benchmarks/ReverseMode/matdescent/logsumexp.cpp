#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdlib.h>
#include <math.h>
#include <inttypes.h>
#include <string.h>

extern int enzyme_const;
template<typename Return, typename... T>
Return __enzyme_autodiff(T...);

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


extern "C" {
#include <adBuffer.h>
}

/*
  Differentiation of logsumexp in reverse (adjoint) mode:
   gradient     of useful results: *x logsumexp
   with respect to varying inputs: *x
   RW status of diff variables: *x:incr logsumexp:in-killed
   Plus diff mem management of: x:in
*/
static void logsumexp_b(const double *__restrict x, double *xb, size_t n, double logsumexpb) {
    double A = x[0];
    double Ab = 0.0;
    int branch;
    double logsumexp;
    for (int i = 0; i < n; ++i)
        if (A < x[i]) {
            A = x[i];
            pushControl1b(0);
        } else {
            pushControl1b(1);
            A = A;
        }
    double sema = 0;
    double semab = 0.0;
    for (int i = 0; i < n; ++i)
        sema = sema + exp(x[i] - A);
    semab = logsumexpb/sema;
    Ab = logsumexpb;
    {
      double tempb;
      for (int i = n-1; i > -1; --i) {
          tempb = exp(x[i]-A)*semab;
          xb[i] = xb[i] + tempb;
          Ab = Ab - tempb;
      }
    }
    for (int i = n-1; i > -1; --i) {
        popControl1b(&branch);
        if (branch == 0) {
            xb[i] = xb[i] + Ab;
            Ab = 0.0;
        }
    }
    xb[0] = xb[0] + Ab;
}

adouble alogsumexp2(const aVector &x, size_t n) {
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

static void enzyme_sincos(double *input, double *inputp, unsigned long n, unsigned long repeat) {
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
    __enzyme_autodiff<void>(logsumexp, input, inputp, n);
  }

  gettimeofday(&end, NULL);
  printf("enzyme forward and reverse %0.6f res'=%f\n", tdiff(&start, &end), sum(inputp, n));
  }
}
static void tapenade_sincos(double *input, double *inputp, unsigned long n, unsigned long repeat) {
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
  printf("tapenade forward %0.6f res'=%f\n", tdiff(&start, &end), total);
  }
  {
      input[0] = realinput;
  struct timeval start, end;
  memset(inputp, 0, sizeof(double)*n);

  gettimeofday(&start, NULL);

  for(int i=0; i<repeat; i++) {
    logsumexp_b(input, inputp, n, 1.0);
  }

  gettimeofday(&end, NULL);
  printf("tapenade forward and reverse %0.6f res'=%f\n", tdiff(&start, &end), sum(inputp, n));
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

  tapenade_sincos(input, inputp, n, repeat);

  enzyme_sincos(input, inputp, n, repeat);
}

