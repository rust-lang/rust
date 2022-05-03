#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdlib.h>
#include <math.h>
#include <inttypes.h>
#include <string.h>
#include <adept_source.h>
#include <adept.h>
using adept::adouble;

static float tdiff(struct timeval *start, struct timeval *end) {
  return (end->tv_sec-start->tv_sec) + 1e-6*(end->tv_usec-start->tv_usec);
}

#define BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
#define BOOST_NO_EXCEPTIONS
#include <iostream>
#include <boost/array.hpp>

#include <boost/numeric/odeint.hpp>

#include <boost/throw_exception.hpp>

using namespace std;
using namespace boost::numeric::odeint;

double foobar(double t, uint64_t iters);

typedef boost::array< adouble , 1 > astate_type;

void alorenz( const astate_type &x , astate_type &dxdt , adouble t )
{
    const double a = 1.2;
    dxdt[0] = -a * x[0];
}

adouble afoobar(adouble t, uint64_t iters) {
    astate_type x = { 1.0 }; // initial conditions

    adouble start = 0.0;
    adouble step = t/adouble(iters);
    //typedef controlled_runge_kutta< runge_kutta_dopri5< astate_type , typename astate_type::value_type , astate_type , adouble > > stepper_type;
    typedef euler< astate_type , typename astate_type::value_type , astate_type , adouble > stepper_type;
    integrate_const( stepper_type(), alorenz , x , start , t, step );

    //x[0] += -1.2 * step * x[0];

    //printf("final result t=%f x(t)=%f, exp(-1.2* t)=%f\n", t, x[0], exp(- 1.2 * t));
    return x[0];
}

static
double afoobar_and_gradient(double xin, double& xgrad, uint64_t iters) {
    adept::Stack stack;
    adouble x = xin;
    stack.new_recording();
    adouble y = afoobar(x, iters);
    y.set_gradient(1.0);
    stack.compute_adjoint();
    xgrad = x.get_gradient();
    return y.value();
}

void adept_sincos(double inp, uint64_t iters) {
  {
  struct timeval start, end;
  gettimeofday(&start, NULL);

  double res = foobar(inp, iters);

  gettimeofday(&end, NULL);
  printf("Adept real %0.6f res=%f\n", tdiff(&start, &end), res);
  }

  {
  struct timeval start, end;
  gettimeofday(&start, NULL);

  adept::Stack stack;
 // stack.new_recording();
  adouble resa = afoobar(inp, iters);
  double res = resa.value();

  gettimeofday(&end, NULL);
  printf("Adept forward %0.6f res=%f\n", tdiff(&start, &end), res);
  }

  {
  struct timeval start, end;
  gettimeofday(&start, NULL);

  double res2 = 0;
  afoobar_and_gradient(inp, res2, iters);

  gettimeofday(&end, NULL);
  printf("Adept combined %0.6f res'=%f\n", tdiff(&start, &end), res2);
  }
}
