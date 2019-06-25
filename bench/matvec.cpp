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

#include "adept.h"
using adept::adouble;

#define N 600
#define M 600
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

#if 1
#include <vector>

static
adouble matvec(aMatrix& mat, aVector& vec) {
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
    sum += out[i] * out[i];
  }
  //delete[] out;
  return sum;
}

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

static void adept_sincos(double *Min, double *Mout, double *Vin, double *Vout) {
  {
  struct timeval start, end;
  gettimeofday(&start, NULL);

  double res = matvec_real(Min, Vin);

  gettimeofday(&end, NULL);
  printf("%0.6f res=%f\n", tdiff(&start, &end), res);
  }

  {
  struct timeval start, end;
  gettimeofday(&start, NULL);

  adept::Stack stack;
  stack.new_recording();
    
    adouble mat[N*M];
    for(int i=0; i<N*M; i++) mat[i] = Min[i];
    adouble vec[M];
    for(int i=0; i<M; i++) vec[i] = Vin[i];

  adouble resa = matvec(mat, vec);
  double res = resa.value();

  gettimeofday(&end, NULL);
  printf("%0.6f res=%f\n", tdiff(&start, &end), res);
  }

  {
  struct timeval start, end;
  gettimeofday(&start, NULL);

  double res2 = 0;
  //sincos_and_gradient(Min, Mout, Vin, Vout);

  gettimeofday(&end, NULL);
  printf("%0.6f res'=%f %f %f\n", tdiff(&start, &end), Mout[1], Mout[2], Mout[3]);
  }
}
#endif

static void my_sincos(double *Min, double *Mout, double *Vin, double *Vout) {

  {
  struct timeval start, end;
  gettimeofday(&start, NULL);

  double res = matvec_real(Min, Vin);

  gettimeofday(&end, NULL);
  printf("%0.6f res=%f\n", tdiff(&start, &end), res);
  }

  {
  struct timeval start, end;
  gettimeofday(&start, NULL);

  double res = matvec_real(Min, Vin);

  gettimeofday(&end, NULL);
  printf("%0.6f res=%f\n", tdiff(&start, &end), res);
  }

  {
  struct timeval start, end;
  gettimeofday(&start, NULL);
  double res2;

  res2 = __builtin_autodiff(matvec_real, Min, Mout, Vin, Vout);

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
  my_sincos(Min, Mout, Vin, Vout);
}

