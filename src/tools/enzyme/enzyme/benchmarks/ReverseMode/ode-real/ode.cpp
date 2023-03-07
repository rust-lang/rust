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
#include <adept_arrays.h>
using adept::adouble;
using adept::aVector;

template<typename Return, typename... T>
Return __enzyme_autodiff(T...);
extern "C" {
  extern int enzyme_dup;
  extern int enzyme_const;
  extern int enzyme_dupnoneed;
}

float tdiff(struct timeval *start, struct timeval *end) {
  return (end->tv_sec-start->tv_sec) + 1e-6*(end->tv_usec-start->tv_usec);
}

#define BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
#define BOOST_NO_EXCEPTIONS
#include <iostream>
#include <boost/array.hpp>

#include <boost/numeric/odeint.hpp>

#include <boost/throw_exception.hpp>
void boost::throw_exception(std::exception const & e){
    //do nothing
}

using namespace std;
using namespace boost::numeric::odeint;

#define N 32
#define xmin 0.
#define xmax 1.
#define ymin 0.
#define ymax 1.

#include <assert.h>
#define RANGE(min, max, i, N) ((max-min)/(N-1)*i + min)
#define GETnb(x, i, j) (x)[N*i+j]
#define GET(x, i, j) GETnb(x, i, j)
//#define GET(x, i, j) ({ assert(i >=0); assert( j>=0); assert(j<N); assert(j<N); GETnb(x, i, j); })

template <typename T>
T brusselator_f(T x, T y, T t) {
  bool eq1 = ((x-0.3)*(x-0.3) + (y-0.6)*(y-0.6)) <= 0.1*0.1;
  bool eq2 = t >= 1.1;
  if (eq1 && eq2) {
    return T(5);
  } else {
    return T(0);
  }
}

void init_brusselator(double* __restrict u, double* __restrict v) {
  for(int i=0; i<N; i++) {
    for(int j=0; j<N; j++) {

      double x = RANGE(xmin, xmax, i, N);
      double y = RANGE(ymin, ymax, j, N);

      GETnb(u, i, j) = 22*(y*(1-y))*sqrt(y*(1-y));
      GETnb(v, i, j) = 27*(x*(1-x))*sqrt(x*(1-x));
    }
  }
}

__attribute__((noinline))
void brusselator_2d_loop(double* __restrict du, double* __restrict dv, const double* __restrict u, const double* __restrict v, const double* __restrict p, double t) {
  double A = p[0];
  double B = p[1];
  double alpha = p[2];
  double dx = (double)1/(N-1);

  alpha = alpha/(dx*dx);

  for(int i=0; i<N; i++) {
    for(int j=0; j<N; j++) {

      double x = RANGE(xmin, xmax, i, N);
      double y = RANGE(ymin, ymax, j, N);

      unsigned ip1 = (i == N-1) ? i : (i+1);
      unsigned im1 = (i == 0) ? i : (i-1);

      unsigned jp1 = (j == N-1) ? j : (j+1);
      unsigned jm1 = (j == 0) ? j : (j-1);

      double u2v = GET(u, i, j) * GET(u, i, j) * GET(v, i, j);

      GETnb(du, i, j) = alpha*( GET(u, im1, j) + GET(u, ip1, j) + GET(u, i, jp1) + GET(u, i, jm1) - 4 * GET(u, i, j))
                      + B + u2v - (A + 1)*GET(u, i, j) + brusselator_f(x, y, t);

      GETnb(dv, i, j) = alpha*( GET(v, im1, j) + GET(v, ip1, j) + GET(v, i, jp1) + GET(v, i, jm1) - 4 * GET(v, i, j))
                      + A * GET(u, i, j) - u2v;
    }
  }
}

typedef boost::array< double , 2 * N * N > state_type;

void lorenz( const state_type &x , state_type &dxdt , double t )
{
    // Extract the parameters
  double p[3] = { /*A*/ 3.4, /*B*/ 1, /*alpha*/10. };
  brusselator_2d_loop(dxdt.c_array(), dxdt.c_array() + N * N, x.data(), x.data() + N * N, p, t);
}

// init_brusselator(x.c_array(), x.c_array() + N*N)

double foobar(const double* p, const state_type x, const state_type adjoint, double t) {
    double dp[3] = { 0. };

    state_type dx = { 0. };

    state_type dadjoint_inp = adjoint;

    state_type dxdu;

    __enzyme_autodiff<void>(brusselator_2d_loop,
//                            enzyme_dup, dxdu.c_array(), dadjoint_inp.c_array(),
//                            enzyme_dup, dxdu.c_array() + N * N, dadjoint_inp.c_array() + N * N,
                            enzyme_dupnoneed, nullptr, dadjoint_inp.data(),
                            enzyme_dupnoneed, nullptr, dadjoint_inp.data() + N * N,
                            enzyme_dup, x.data(), dx.data(),
                            enzyme_dup, x.data() + N * N, dx.data() + N * N,
                            enzyme_dup, p, dp,
                            enzyme_const, t);

    return dx[0];
}

#undef GETnb
#define GETnb(x, i, j) (x)(N*i+j)

void abrusselator_2d_loop(aVector& du, aVector& dv, aVector& u, aVector& v, aVector& p, double t) {
  adouble A = p(0);
  adouble B = p(1);
  adouble alpha = p(2);
  adouble dx = (double)1/(N-1);

  alpha = alpha/(dx*dx);

  for(int i=0; i<N; i++) {
    for(int j=0; j<N; j++) {

      adouble x = RANGE(xmin, xmax, i, N);
      adouble y = RANGE(ymin, ymax, j, N);

      unsigned ip1 = (i == N-1) ? i : (i+1);
      unsigned im1 = (i == 0) ? i : (i-1);

      unsigned jp1 = (j == N-1) ? j : (j+1);
      unsigned jm1 = (j == 0) ? j : (j-1);

      adouble u2v = GET(u, i, j) * GET(u, i, j) * GET(v, i, j);

      GETnb(du, i, j) = alpha*( GET(u, im1, j) + GET(u, ip1, j) + GET(u, i, jp1) + GET(u, i, jm1) - 4 * GET(u, i, j))
                      + B + u2v - (A + 1)*GET(u, i, j) + brusselator_f<adouble>(x, y, t);

      GETnb(dv, i, j) = alpha*( GET(v, im1, j) + GET(v, ip1, j) + GET(v, i, jp1) + GET(v, i, jm1) - 4 * GET(v, i, j))
                      + A * GET(u, i, j) - u2v;
    }
  }
}

double afoobar(const double* p_in, const state_type x, const state_type adjoint, double t) {
    adept::Stack stack;

    aVector p(3);
    for(unsigned i=0; i<3; i++) p(i) = p_in[i];
    aVector ax(N*N);
    aVector ay(N*N);
    for(unsigned i=0; i<N*N; i++) {
      ax(i) = x[i];
      ay(i) = x[i+N*N];
    }

    aVector dxdu(N*N);
    aVector dydu(N*N);

    stack.new_recording();

    abrusselator_2d_loop(dxdu, dydu, ax, ay, p, t);

    for(unsigned i=0; i<N*N; i++) {
      dxdu(i).set_gradient(adjoint[i]);
      dydu(i).set_gradient(adjoint[i+N*N]);
    }
    stack.compute_adjoint();

    return ax(0).get_gradient();
}


//! Tapenade
extern "C" {
  /*        Generated by TAPENADE     (INRIA, Ecuador team)
    Tapenade 3.15 (master) -  8 Jan 2020 10:48
*/
#include <adBuffer.h>

/*
  Differentiation of get in reverse (adjoint) mode (with options i4 dr8 r4):
   gradient     of useful results: *x get
   with respect to varying inputs: *x
   Plus diff mem management of: x:in
*/
void get_b(const double *x, double *xb, unsigned int i, unsigned int j, double getb)
{
    double get;
    xb[N*i + j] = xb[N*i + j] + getb;
}

double get_nodiff(const double *x, unsigned int i, unsigned int j) {
    return x[N*i + j];
}

double brusselator_f_nodiff(double x, double y, double t) {
    if ((x-0.3)*(x-0.3) + (y-0.6)*(y-0.6) <= 0.1*0.1 && t >= 1.1)
        return 5.0;
    else
        return 0.0;
}

#if 1
void brusselator_2d_loop_b(double *du, double *dub, double *dv, double *dvb,
        const double *u, double *ub, const double *v, double *vb, const double *p, double *pb,
        double t) {
    double A = p[0];
    double Ab = 0.0;
    double B = p[1];
    double Bb = 0.0;
    double alpha = p[2];
    double alphab = 0.0;
    double dx = (double)1/(N-1);
    alpha = alpha/(dx*dx);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            double x = (xmax-xmin)/(N-1)*i + xmin;
            double y = (ymax-ymin)/(N-1)*j + ymin;
            unsigned int ip1 = (i == N - 1 ? i : i + 1);
            unsigned int im1 = (i == 0 ? i : i - 1);
            unsigned int jp1 = (j == N - 1 ? j : j + 1);
            unsigned int jm1 = (j == 0 ? j : j - 1);
            double u2v = u[N*i+j]*u[N*i+j]*v[N*i+j];
            double result1;
            pushInteger4(jm1);
            pushInteger4(jp1);
            pushInteger4(im1);
            pushInteger4(ip1);
        }
    *ub = 0.0;
    *vb = 0.0;
    alphab = 0.0;
    Ab = 0.0;
    Bb = 0.0;
    for (int i = N-1; i > -1; --i)
        for (int j = N-1; j > -1; --j) {
            double x;
            double y;
            unsigned int ip1;
            unsigned int im1;
            unsigned int jp1;
            unsigned int jm1;
            double u2v;
            double u2vb = 0.0;
            double result1;
            double temp;
            double tempb;
            popInteger4((int*)&ip1);
            popInteger4((int*)&im1);
            popInteger4((int*)&jp1);
            popInteger4((int*)&jm1);
            temp = u[N*i + j];
            alphab = alphab + (v[N*im1+j]+v[N*ip1+j]+v[N*i+jp1]+v[N*i+jm1]
                -4*v[N*i+j])*dvb[N*i+j] + (u[N*im1+j]+u[N*ip1+j]+u[N*i+
                jp1]+u[N*i+jm1]-4*u[N*i+j])*dub[N*i+j];
            tempb = alpha*dvb[N*i+j];
            Ab = Ab + u[N*i+j]*dvb[N*i+j] - u[N*i+j]*dub[N*i+j];
            ub[N*i + j] = ub[N*i + j] + A*dvb[N*i+j] - (A+1)*dub[N*i+j];
            u2vb = dub[N*i + j] - dvb[N*i + j];
            dvb[N*i + j] = 0.0;
            vb[N*im1 + j] = vb[N*im1 + j] + tempb;
            vb[N*ip1 + j] = vb[N*ip1 + j] + tempb;
            vb[N*i + jp1] = vb[N*i + jp1] + tempb;
            vb[N*i + jm1] = vb[N*i + jm1] + tempb;
            vb[N*i + j] = vb[N*i + j] + temp*temp*u2vb - 4*tempb;
            tempb = alpha*dub[N*i+j];
            Bb = Bb + dub[N*i + j];
            dub[N*i + j] = 0.0;
            ub[N*im1 + j] = ub[N*im1 + j] + tempb;
            ub[N*ip1 + j] = ub[N*ip1 + j] + tempb;
            ub[N*i + jp1] = ub[N*i + jp1] + tempb;
            ub[N*i + jm1] = ub[N*i + jm1] + tempb;
            ub[N*i + j] = ub[N*i + j] + 2*temp*v[N*i+j]*u2vb - 4*tempb;
        }
    alphab = alphab/(dx*dx);
    pb[2] = pb[2] + alphab;
    pb[1] = pb[1] + Bb;
    pb[0] = pb[0] + Ab;

}
#else
/*
  Differentiation of brusselator_2d_loop in reverse (adjoint) mode (with options i4 dr8 r4):
   gradient     of useful results: *du *dv
   with respect to varying inputs: *p *u *du *v *dv
   RW status of diff variables: *p:out *u:out *du:in-out *v:out
                *dv:in-out
   Plus diff mem management of: p:in u:in du:in v:in dv:in
*/
void brusselator_2d_loop_b(double *du, double *dub, double *dv, double *dvb,
        const double *u, double *ub, const double *v, double *vb, const double *p, double *pb,
        double t) {
    double A = p[0];
    double Ab = 0.0;
    double B = p[1];
    double Bb = 0.0;
    double alpha = p[2];
    double alphab = 0.0;
    double dx = (double)1/(N-1);
    alpha = alpha/(dx*dx);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            double x = (xmax-xmin)/(N-1)*i + xmin;
            double y = (ymax-ymin)/(N-1)*j + ymin;
            unsigned int ip1 = (i == N - 1 ? i : i + 1);
            unsigned int im1 = (i == 0 ? i : i - 1);
            unsigned int jp1 = (j == N - 1 ? j : j + 1);
            unsigned int jm1 = (j == 0 ? j : j - 1);
            double u2v;
            double result1;
            double result2;
            double result3;
            double result4;
            double result5;
            double result6;
            double result7;
            result1 = get_nodiff(u, i, j);
            result2 = get_nodiff(u, i, j);
            result3 = get_nodiff(v, i, j);
            pushReal8(result1);
            result1 = get_nodiff(u, im1, j);
            pushReal8(result2);
            result2 = get_nodiff(u, ip1, j);
            pushReal8(result3);
            result3 = get_nodiff(u, i, jp1);
            result4 = get_nodiff(u, i, jm1);
            result5 = get_nodiff(u, i, j);
            result6 = get_nodiff(u, i, j);
            pushReal8(result1);
            result1 = get_nodiff(v, im1, j);
            pushReal8(result2);
            result2 = get_nodiff(v, ip1, j);
            pushReal8(result3);
            result3 = get_nodiff(v, i, jp1);
            pushReal8(result4);
            result4 = get_nodiff(v, i, jm1);
            pushReal8(result5);
            result5 = get_nodiff(v, i, j);
            pushReal8(result6);
            result6 = get_nodiff(u, i, j);
            pushInteger4(jm1);
            pushInteger4(jp1);
            pushReal8(result6);
            pushReal8(result5);
            pushReal8(result4);
            pushReal8(result3);
            pushReal8(result2);
            pushReal8(result1);
            pushInteger4(im1);
            pushInteger4(ip1);
        }
    *ub = 0.0;
    *vb = 0.0;
    alphab = 0.0;
    Ab = 0.0;
    Bb = 0.0;
    for (int i = N-1; i > -1; --i)
        for (int j = N-1; j > -1; --j) {
            double x;
            double y;
            unsigned int ip1;
            unsigned int im1;
            unsigned int jp1;
            unsigned int jm1;
            double u2v;
            double u2vb;
            double result1;
            double result1b;
            double result2;
            double result2b;
            double result3;
            double result3b;
            double result4;
            double result4b;
            double result5;
            double result5b;
            double result6;
            double result6b;
            double result7;
            double tempb;
            popInteger4((int*)&ip1);
            popInteger4((int*)&im1);
            popReal8(&result1);
            popReal8(&result2);
            popReal8(&result3);
            popReal8(&result4);
            popReal8(&result5);
            popReal8(&result6);
            popInteger4((int*)&jp1);
            popInteger4((int*)&jm1);
            alphab = alphab + (result1+result2+result3+result4-4*result5)*dvb[
                N*i+j];
            tempb = alpha*dvb[N*i+j];
            Ab = Ab + result6*dvb[N*i+j];
            result6b = A*dvb[N*i+j];
            u2vb = dub[N*i + j] - dvb[N*i + j];
            dvb[N*i + j] = 0.0;
            result1b = tempb;
            result2b = tempb;
            result3b = tempb;
            result4b = tempb;
            result5b = -(4*tempb);
            popReal8(&result6);
            get_b(u, ub, i, j, result6b);
            popReal8(&result5);
            get_b(v, vb, i, j, result5b);
            popReal8(&result4);
            get_b(v, vb, i, jm1, result4b);
            popReal8(&result3);
            get_b(v, vb, i, jp1, result3b);
            popReal8(&result2);
            get_b(v, vb, ip1, j, result2b);
            popReal8(&result1);
            get_b(v, vb, im1, j, result1b);
            alphab = alphab + (result1+result2+result3+result4-4*result5)*dub[
                N*i+j];
            tempb = alpha*dub[N*i+j];
            Bb = Bb + dub[N*i + j];
            Ab = Ab - result6*dub[N*i+j];
            result6b = -((A+1)*dub[N*i+j]);
            dub[N*i + j] = 0.0;
            result1b = tempb;
            result2b = tempb;
            result3b = tempb;
            result4b = tempb;
            result5b = -(4*tempb);
            get_b(u, ub, i, j, result6b);
            get_b(u, ub, i, j, result5b);
            get_b(u, ub, i, jm1, result4b);
            popReal8(&result3);
            get_b(u, ub, i, jp1, result3b);
            popReal8(&result2);
            get_b(u, ub, ip1, j, result2b);
            popReal8(&result1);
            get_b(u, ub, im1, j, result1b);
            result1b = result2*result3*u2vb;
            result2b = result1*result3*u2vb;
            result3b = result1*result2*u2vb;
            get_b(v, vb, i, j, result3b);
            get_b(u, ub, i, j, result2b);
            get_b(u, ub, i, j, result1b);
        }
    alphab = alphab/(dx*dx);
    pb[2] = pb[2] + alphab;
    pb[1] = pb[1] + Bb;
    pb[0] = pb[0] + Ab;
}
#endif
}

double tfoobar(const double* p, const state_type x, const state_type adjoint, double t) {
    double dp[3] = { 0. };

    state_type dx = { 0. };

    state_type dadjoint_inp = adjoint;

    state_type dxdu;

    brusselator_2d_loop_b(nullptr, dadjoint_inp.data(),
                          nullptr, dadjoint_inp.data() + N * N,
                          x.data(), dx.data(),
                          x.data() + N * N, dx.data() + N * N,
                          p, dp,
                          t);

    return dx[0];
}

//! Main
int main(int argc, char** argv) {
  const double p[3] = { /*A*/ 3.4, /*B*/ 1, /*alpha*/10. };

  state_type x;
  init_brusselator(x.data(), x.data() + N * N);

  state_type adjoint;
  init_brusselator(adjoint.data(), adjoint.data() + N * N);

  double t = 2.1;

  {
  struct timeval start, end;
  gettimeofday(&start, NULL);

  double res;
  for(int i=0; i<10000; i++)
  res = afoobar(p, x, adjoint, t);

  gettimeofday(&end, NULL);
  printf("Adept combined %0.6f res=%f\n", tdiff(&start, &end), res);
  }

  {
  struct timeval start, end;
  gettimeofday(&start, NULL);

  double res;
  for(int i=0; i<10000; i++)
  res = tfoobar(p, x, adjoint, t);

  gettimeofday(&end, NULL);
  printf("Tapenade combined %0.6f res=%f\n", tdiff(&start, &end), res);
  }

  {
  struct timeval start, end;
  gettimeofday(&start, NULL);

  double res;
  for(int i=0; i<10000; i++)
  res = foobar(p, x, adjoint, t);

  gettimeofday(&end, NULL);
  printf("Enzyme combined %0.6f res=%f\n", tdiff(&start, &end), res);
  }
  //printf("res=%f\n", foobar(1000));
}


#if 0

typedef boost::array< double , 6 > state_type;

void lorenz( const state_type &x , state_type &dxdt , double t )
{
    // Extract the parameters
    double k1 = x[3];
    double k2 = x[4];
    double k3 = x[5];

    dxdt[0] = -k1 * x[0] + k3 * x[1] * x[2];
    dxdt[1] = k1 * x[0] - k2 * x[1] * x[1] - k3 * x[1] * x[2];
    dxdt[2] = k2 * x[1] * x[1];

    // Don't change the parameters p
    dxdt[3] = 0;
    dxdt[4] = 0;
    dxdt[5] = 0;
}

double foobar(double* p, uint64_t iters) {
    state_type x = { 1.0, 0, 0, p[0], p[1], p[2] }; // initial conditions
    double t = 1e5;
    typedef controlled_runge_kutta< runge_kutta_dopri5< state_type , typename state_type::value_type , state_type , double > > stepper_type;
    //typedef euler< state_type , typename state_type::value_type , state_type , double > stepper_type;
    integrate_const( stepper_type(), lorenz , x , 0.0 , t, t/iters );

    return x[0];
}

typedef boost::array< adouble , 6 > astate_type;

void alorenz( const astate_type &x , astate_type &dxdt , adouble t )
{
    // Extract the parameters
    adouble k1 = x[3];
    adouble k2 = x[4];
    adouble k3 = x[5];

    dxdt[0] = -k1 * x[0] + k3 * x[1] * x[2];
    dxdt[1] = k1 * x[0] - k2 * x[1] * x[1] - k3 * x[1] * x[2];
    dxdt[2] = k2 * x[1] * x[1];

    // Don't change the parameters p
    dxdt[3] = 0;
    dxdt[4] = 0;
    dxdt[5] = 0;
}

adouble afoobar(adouble* p, uint64_t iters) {
    astate_type x = { 1.0, 0, 0, p[0], p[1], p[2] }; // initial conditions
    double t = 1e5;
    typedef controlled_runge_kutta< runge_kutta_dopri5< astate_type , typename astate_type::value_type , astate_type , adouble > > stepper_type;
    //typedef euler< astate_type , typename astate_type::value_type , astate_type , adouble > stepper_type;
    integrate_const( stepper_type(), alorenz , x , 0.0 , t, t/iters );

    return x[0];
}

static
double afoobar_and_gradient(double* p_in, double* dp_out, uint64_t iters) {
    adept::Stack stack;
    adouble x[3] = { p_in[0], p_in[1], p_in[2] };
    stack.new_recording();
    adouble y = afoobar(x, iters);
    y.set_gradient(1.0);
    stack.compute_adjoint();
    for(int i=0; i<3; i++)
      dp_out[i] = x[i].get_gradient();
    return y.value();
}

static void adept_sincos(uint64_t iters) {
  {
  struct timeval start, end;
  gettimeofday(&start, NULL);

  double p[3] = { 0.04,3e7,1e4 };
  double res = foobar(p, iters);

  gettimeofday(&end, NULL);
  printf("Adept real %0.6f res=%f\n", tdiff(&start, &end), res);
  }

  {
  struct timeval start, end;
  gettimeofday(&start, NULL);

  adept::Stack stack;
  adouble p[3] = { 0.04,3e7,1e4 };
 // stack.new_recording();
  adouble resa = afoobar(p, iters);
  double res = resa.value();

  gettimeofday(&end, NULL);
  printf("Adept forward %0.6f res=%f\n", tdiff(&start, &end), res);
  }

  {
  struct timeval start, end;
  gettimeofday(&start, NULL);

  double p[3] = { 0.04,3e7,1e4 };
  double dp[3] = { 0 };
  afoobar_and_gradient(p, dp, iters);

  gettimeofday(&end, NULL);
  printf("Adept combined %0.6f res'=%f\n", tdiff(&start, &end), dp[0]);
  }
}

static void enzyme_sincos(double inp, uint64_t iters) {

  {
  struct timeval start, end;
  gettimeofday(&start, NULL);

  double p[3] = { 0.04,3e7,1e4 };
  double res = foobar(p, iters);

  gettimeofday(&end, NULL);
  printf("Enzyme real %0.6f res=%f\n", tdiff(&start, &end), res);
  }

  {
  struct timeval start, end;
  gettimeofday(&start, NULL);

  double p[3] = { 0.04,3e7,1e4 };
  double res = foobar(p, iters);

  gettimeofday(&end, NULL);
  printf("Enzyme forward %0.6f res=%f\n", tdiff(&start, &end), res);
  }

  {
  struct timeval start, end;
  gettimeofday(&start, NULL);

  double p[3] = { 0.04,3e7,1e4 };
  double dp[3] = { 0 };
  __enzyme_autodiff<void>(foobar, p, dp, iters);

  gettimeofday(&end, NULL);
  printf("Enzyme combined %0.6f res'=%f\n", tdiff(&start, &end), dp[0]);
  }
}

int main(int argc, char** argv) {

  int max_iters = atoi(argv[1]) ;
  double inp = 2.1;

  //for(int iters=max_iters/20; iters<=max_iters; iters+=max_iters/20) {
  auto iters = max_iters;
    printf("iters=%d\n", iters);
    adept_sincos(inp, iters);
    enzyme_sincos(inp, iters);
  //}
}
#endif
