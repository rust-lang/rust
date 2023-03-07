#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdlib.h>
#include <math.h>
#include <inttypes.h>
#include <string.h>

template<typename Return, typename... T>
Return __enzyme_autodiff(T...);

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

static void enzyme_sincos(double inp) {

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

  res2 = __enzyme_autodiff<double>(sincos_real, inp);

  gettimeofday(&end, NULL);
  printf("%0.6f res'=%f\n", tdiff(&start, &end), res2);
  }
}

int main(int argc, char** argv) {

  double inp = atof(argv[1]) ;
  printf("adept\n");
  adept_sincos(inp);
  printf("tapenade\n");
  tapenade_sincos(inp);
  printf("enzyme\n");
  enzyme_sincos(inp);
}
