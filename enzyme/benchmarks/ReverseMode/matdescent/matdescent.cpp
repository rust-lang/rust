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

#include <adept_arrays.h>
using adept::adouble;
using adept::aMatrix;
using adept::aVector;

using adept::Vector;

#define N 2000
#define M 2000
#define ITERS 1000
#define RATE 0.00000001

double matvec_real(double* mat, double* vec) {
  double *out = (double*)malloc(sizeof(double)*N);
  //double *out = new double[N];
  for(int i=0; i<N; i++) {
    out[i] = 0;
    for(int j=0; j<M; j++) {
        out[i] += mat[i*M+j] * vec[j];
    }
  }
  double sum = 0;
  for(int i=0; i<N; i++) {
    sum += out[i] * out[i];
  }
  free(out);
  //delete[] out;
  return sum;
}

/*
  Differentiation of matvec_real in reverse (adjoint) mode:
   gradient     of useful results: alloc(*out) matvec_real *mat
   with respect to varying inputs: alloc(*out) *mat
   RW status of diff variables: alloc(*out):in-out matvec_real:in-killed
                *mat:incr
   Plus diff mem management of: mat:in
*/
void matvec_real_b(double *mat, double *matb, double *vec, double matvec_realb) {
    double *out;
    double *outb;
    int ii1;
    double matvec_real;
    outb = (double *)malloc(sizeof(double)*N);
    for (ii1 = 0; ii1 < N; ++ii1)
        outb[ii1] = 0.0;
    out = (double *)malloc(sizeof(double)*N);
    //double *out = new double[N];
    for (int i = 0; i < N; ++i) {
        out[i] = 0;
        for (int j = 0; j < M; ++j)
            out[i] = out[i] + mat[i*M+j]*vec[j];
    }
    double sum = 0;
    double sumb = 0.0;
    sumb = matvec_realb;
    for (int i = N; i > -1; --i)
        outb[i] = outb[i] + 2*out[i]*sumb;
    for (int i = N; i > -1; --i) {
        for (int j = M; j > -1; --j)
            matb[i*M + j] = matb[i*M + j] + vec[j]*outb[i];
        outb[i] = 0.0;
    }
    free(out);
    free(outb);
}

#if 1
#include <vector>

static
adouble matvec(aMatrix& mat, Vector& vec) {
  //std::vector<adouble> out(N);

  aVector out = mat**vec;

#if 0
  for(int i=0; i<N; i++) {
    out[i] = 0;
    for(int j=0; j<M; j++) {
        out[i] +=
//        out[i] += mat[i*M+j] * vec[j];
    }
  }
#endif
  adouble sum = 0;
  for(int i=0; i<N; i++) {
    sum += out(i) * out(i);
  }
  //delete[] out;
  return sum;
}

#if 0
static
void sincos_and_gradient(double *Min, double *Mout, double *vecin, double *vecout) {
    //adouble *mat = new adouble[N*M];
    //std::vector<adouble> mat(N*M);// = new adouble[N*M];
    //adept::set_values(&mat[0], N*M, Min);
    //for(int i=0; i<N*M; i++) mat[i] = Min[i];
    //adouble *vec = new adouble[M];
    //std::vector<adouble> vec(M);// = new adouble[M];
    //adept::set_values(&vec[0], M, vecin);
    //for(int i=0; i<M; i++) vec[i] = vecin[i];

    //aMatrix M = aMatrix(N,M);
    //for (int i = 0; i < N; i++) {
    //  for (int j = 0; j < M; j++) {
    //    M.
    //  }
    //}

    adept::Stack stack;
    stack.new_recording();
    adouble loss = matvec(&mat[0], &vec[0]);
    loss.set_gradient(1.0);
    stack.compute_adjoint();

    adept::get_gradients(&mat[0], N*M, Mout);
    adept::get_gradients(&vec[0], M, vecout);
    //for(int i=0; i<N*M; i++) Mout[i] = mat[i].get_gradient();
    //for(int i=0; i<M; i++) vecout[i] = vec[i].get_gradient();

    //delete[] mat;
    //delete[] vec;
    //xgrad = x.get_gradient();
    //return y.value();
}
#endif

static void adept_sincos(double *Min, double *Mout, double *Vin, double *Vout) {

  {
  struct timeval start, end;

  double res2 = 0;
  adept::Stack stack;

    aMatrix mat(N,M);
    for(int i=0; i<N; i++) {
    for(int j=0; j<M; j++) {
        mat(i, j) = Min[i*M+j];
    }
    }
    Vector vec(M);
    for(int i=0; i<M; i++) vec(i) = Vin[i];


  gettimeofday(&start, NULL);
  for (int iter = 0; iter < ITERS; iter++) {
    stack.new_recording();
    adouble resa = matvec(mat, vec);
    resa.set_gradient(1.0);
    stack.continue_recording();
  }
    //stack.reverse();
    //stack.pause_recording();
    /*
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < M; j++) {
        mat(i,j) -= mat(i,j).get_gradient()*RATE;
      }
    }*/
  gettimeofday(&end, NULL);
  //gettimeofday(&end, NULL);
  printf("%0.6f res'=%f %f %f\n", tdiff(&start, &end), Mout[1], Mout[2], Mout[3]);

  }



  {
  struct timeval start, end;
  //gettimeofday(&start, NULL);

  double res2 = 0;
  {
  adept::Stack stack;

    aMatrix mat(N,M);
    for(int i=0; i<N; i++) {
    for(int j=0; j<M; j++) {
        mat(i, j) = Min[i*M+j];
    }
    }
    Vector vec(M);
    for(int i=0; i<M; i++) vec(i) = Vin[i];


  gettimeofday(&start, NULL);
  for (int iter = 0; iter < ITERS; iter++) {
    stack.new_recording();
    adouble resa = matvec(mat, vec);
    resa.set_gradient(1.0);
    stack.reverse();
    stack.pause_recording();
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < M; j++) {
        mat(i,j) -= mat(i,j).get_gradient()*RATE;
      }
    }
    stack.continue_recording();
  }
  gettimeofday(&end, NULL);
#if 0
    stack.new_recording();
     gettimeofday(&start, NULL);
  adouble resa = matvec(mat, vec);
  res2 = resa.value();

    resa.set_gradient(1.0);
    stack.reverse();
    gettimeofday(&end, NULL);
    stack.pause_recording();
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < M; j++) {
        Mout[i*M+j] = mat(i,j).get_gradient();
      }
    }
#endif
    //for(int i=0; i<M; i++) Vout[i] = vec(i).get_gradient();

    stack.pause_recording();
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < M; j++) {
        Mout[i*M+j] = mat(i,j).get_gradient();
      }
    }

  }
  //gettimeofday(&end, NULL);
  printf("%0.6f res'=%f %f %f\n", tdiff(&start, &end), Mout[1], Mout[2], Mout[3]);
  }
}
#endif

static void tapenade_sincos(double *Min, double *Mout, double *Vin, double *Vout) {

  {
  struct timeval start, end;
  gettimeofday(&start, NULL);

  double res = matvec_real(Min, Vin);

  gettimeofday(&end, NULL);
  printf("tapenade %0.6f res=%f\n", tdiff(&start, &end), res);
  }

  {
  struct timeval start, end;
  double tmp = Min[0];
  gettimeofday(&start, NULL);

  double sum = 0;
  for(int i=0; i<ITERS; i++) {
      Min[0] = tmp + i/100000000.;
        sum += matvec_real(Min, Vin);
  }

  gettimeofday(&end, NULL);
  printf("tapenade mv %0.6f res=%f\n", tdiff(&start, &end), sum);
  Min[0] = tmp;
  }

  {
  struct timeval start, end;
  gettimeofday(&start, NULL);
  double res2;

  for(int i=0; i<ITERS; i++) {
  for(int i=0; i<N*M; i++) { Mout[i] = 0; }
  //for(int i=0; i<M; i++) { Vout[i] = 0; }
    matvec_real_b(Min, Mout, Vin, 1.0);
  //res2 = __builtin_autodiff(matvec_real, Min, Mout, Vin, Vout);
  for(int i=0; i<N*M; i++) { Min[i] -= Mout[i] * RATE; }
  }

  gettimeofday(&end, NULL);
  printf("tapenade %0.6f res'=%f %f %f\n", tdiff(&start, &end), Mout[1], Mout[2], Mout[3]);
  }
}
static void enzyme_sincos(double *Min, double *Mout, double *Vin, double *Vout) {

  {
  struct timeval start, end;
  gettimeofday(&start, NULL);

  double res = matvec_real(Min, Vin);

  gettimeofday(&end, NULL);
  printf("%0.6f res=%f\n", tdiff(&start, &end), res);
  }

  {
  struct timeval start, end;
  double tmp = Min[0];
  gettimeofday(&start, NULL);

  double sum = 0;
  for(int i=0; i<ITERS; i++) {
      Min[0] = tmp + i/100000000.;
        sum += matvec_real(Min, Vin);
  }

  gettimeofday(&end, NULL);
  printf("mv %0.6f res=%f\n", tdiff(&start, &end), sum);
  Min[0] = tmp;
  }

  {
  struct timeval start, end;
  gettimeofday(&start, NULL);
  double res2;

  for(int i=0; i<ITERS; i++) {
  for(int i=0; i<N*M; i++) { Mout[i] = 0; }
  //for(int i=0; i<M; i++) { Vout[i] = 0; }
  res2 = __enzyme_autodiff<double>(matvec_real, Min, Mout, enzyme_const, Vin);
  //res2 = __builtin_autodiff(matvec_real, Min, Mout, Vin, Vout);
  for(int i=0; i<N*M; i++) { Min[i] -= Mout[i] * RATE; }
  }

  gettimeofday(&end, NULL);
  printf("%0.6f res'=%f %f %f\n", tdiff(&start, &end), Mout[1], Mout[2], Mout[3]);
  }
}

int main(int argc, char** argv) {

  double *Min = new double[N*M];
  double *Mout = new double[N*M];
  double *Vin = new double[M];
  double *Vout = new double[M];

  for(int i=0; i<N*M; i++) Min[i] = 3*i;
  for(int i=0; i<M; i++) Vin[i] = 1*i;

  memset(Mout, 0, sizeof(double)*N*M);
  memset(Vout, 0, sizeof(double)*M);
  adept_sincos(Min, Mout, Vin, Vout);

  memset(Mout, 0, sizeof(double)*N*M);
  memset(Vout, 0, sizeof(double)*M);
  tapenade_sincos(Min, Mout, Vin, Vout);

  memset(Mout, 0, sizeof(double)*N*M);
  memset(Vout, 0, sizeof(double)*M);
  enzyme_sincos(Min, Mout, Vin, Vout);
}

