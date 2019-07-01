#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdlib.h>
#include <math.h>
#include <inttypes.h>
#include <string.h>

float tdiff(struct timeval *start, struct timeval *end) {
  return (end->tv_sec-start->tv_sec) + 1e-6*(end->tv_usec-start->tv_usec);
}

#include <adept_source.h>
#include <adept.h>
using adept::adouble;

#define SINCOSN 10000000
static 
double sincos_real(double x) {
  double sum = 0;
  for(int i=1; i<=SINCOSN; i++) {
    sum += pow(x, i) / i;
  }
  return sum;
}

static void sincos_real_tapenade(double x, double *xb, double sincos_realb) {
    double sum = 0;
    double sumb = 0.0;
    double sincos_real;
    sumb = sincos_realb;
    for (int i = SINCOSN; i > 0; --i)
        if (!(x<=0.0&&(i==0.0||i!=(int)i)))
            *xb = *xb + pow(x, (i-1))*sumb;
}

#if 1
static
adouble sincos(adouble x) {
  adouble sum = 0;
  for(int i=1; i<=SINCOSN; i++) {
    sum += pow(x, i) / i;
  }
  return sum;
}

static
double sincos_and_gradient(double xin, double& xgrad) {
    adept::Stack stack;
    adouble x = xin;
    stack.new_recording();
    adouble y = sincos(x);
    y.set_gradient(1.0);
    stack.compute_adjoint();
    xgrad = x.get_gradient();
    return y.value();
}

static void adept_sincos(double inp) {
  {
  struct timeval start, end;
  gettimeofday(&start, NULL);

  double res = sincos_real(inp);

  gettimeofday(&end, NULL);
  printf("%0.6f res=%f\n", tdiff(&start, &end), res);
  }

  {
  struct timeval start, end;
  gettimeofday(&start, NULL);

  adept::Stack stack;
 // stack.new_recording();
  adouble resa = sincos(inp);
  double res = resa.value();

  gettimeofday(&end, NULL);
  printf("%0.6f res=%f\n", tdiff(&start, &end), res);
  }

  {
  struct timeval start, end;
  gettimeofday(&start, NULL);

  double res2 = 0;
  sincos_and_gradient(inp, res2);

  gettimeofday(&end, NULL);
  printf("%0.6f res'=%f\n", tdiff(&start, &end), res2);
  }
}
#endif

static void tapenade_sincos(double inp) {

  {
  struct timeval start, end;
  gettimeofday(&start, NULL);

  double res = sincos_real(inp);

  gettimeofday(&end, NULL);
  printf("tapenade %0.6f res=%f\n", tdiff(&start, &end), res);
  }

  {
  struct timeval start, end;
  gettimeofday(&start, NULL);

  double res = sincos_real(inp);

  gettimeofday(&end, NULL);
  printf("tapenade %0.6f res=%f\n", tdiff(&start, &end), res);
  }

  {
  struct timeval start, end;
  gettimeofday(&start, NULL);
  double res2 = 0;

  sincos_real_tapenade(inp, &res2, 1.0);

  gettimeofday(&end, NULL);
  printf("tapendade %0.6f res'=%f\n", tdiff(&start, &end), res2);
  }
}
static void my_sincos(double inp) {

  {
  struct timeval start, end;
  gettimeofday(&start, NULL);

  double res = sincos_real(inp);

  gettimeofday(&end, NULL);
  printf("%0.6f res=%f\n", tdiff(&start, &end), res);
  }

  {
  struct timeval start, end;
  gettimeofday(&start, NULL);

  double res = sincos_real(inp);

  gettimeofday(&end, NULL);
  printf("%0.6f res=%f\n", tdiff(&start, &end), res);
  }

  {
  struct timeval start, end;
  gettimeofday(&start, NULL);
  double res2;

  res2 = __builtin_autodiff(sincos_real, inp);

  gettimeofday(&end, NULL);
  printf("%0.6f res'=%f\n", tdiff(&start, &end), res2);
  }
}

int main(int argc, char** argv) {

  double inp = atof(argv[1]) ;
  adept_sincos(inp);
  tapenade_sincos(inp);
  my_sincos(inp);
}

#if 0
void matvec(size_t N, size_t M, double* mat, double* vec, double* out) {
  for(int i=0; i<N; i++) {
    out[i] = 0;
    for(int j=0; j<M; j++) {
        out[i] += mat[i*M+j] * vec[j];
    }
  }
}

int main(int argc, char** argv) {
  #define N 10ull
  #define M 4ull
  double mat[N*M];
  double matp[N*M];
  double vec[M];
  double vecp[M];
  double out[N];
  double outp[N];
  matvec(N, M, mat, vec, out);
  __builtin_autodiff(matvec, N, M, mat, matp, vec, vecp, out, outp);
}

static double max(double x, double y) {
    return (x > y) ? x : y;
}

__attribute__((noinline))
static double logsumexp(double *__restrict x, size_t n) {
  double A = x[0];
  for(int i=0; i<n; i++) {
    A = max(A, x[i]);
  }
  double ema[n];
  for(int i=0; i<n; i++) {
    ema[i] = exp(x[i] - A);
  }
  double sema = 0;
  for(int i=0; i<n; i++)
    sema += ema[i];
  return log(sema) + A;
}

int main(int argc, char** argv) {

  size_t size = 100000;
  double* rands = (double*)malloc(sizeof(double)*size);
  double* randsp = (double*)malloc(sizeof(double)*size);

  for(int i=0; i<size; i++) {
    rands[i] = i;
  }

  {
  struct timeval start, end;
  gettimeofday(&start, NULL);

  double res = logsumexp(rands, size);

  gettimeofday(&end, NULL);
  printf("%0.6f res=%f\n", tdiff(&start, &end), res);
  }

  {
  struct timeval start, end;
  gettimeofday(&start, NULL);

  double res = __builtin_autodiff(logsumexp, rands, randsp, size);

  gettimeofday(&end, NULL);
  printf("%0.6f res'=%f\n", tdiff(&start, &end), res);
  }
}
#endif

