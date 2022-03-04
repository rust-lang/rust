#ifndef _fft_h_
#define _fft_h_

#include <adept_source.h>
#include <adept.h>
#include <adept_arrays.h>
using adept::adouble;

using adept::aVector;


/*
  A classy FFT and Inverse FFT C++ class library

  Author: Tim Molteno, tim@physics.otago.ac.nz

  Based on the article "A Simple and Efficient FFT Implementation in C++" by Volodymyr Myrnyy
  with just a simple Inverse FFT modification.

  Licensed under the GPL v3.
*/


#include <cmath>

inline void swap(double* a, double* b) {
  double temp=*a;
  *a = *b;
  *b = temp;
}

inline void swapad(adept::ActiveReference<double> a, adept::ActiveReference<double> b) {
  adouble temp=a;
  a = b;
  b = temp;
}


template<unsigned N, typename T=double>
class DanielsonLanczos
{
  DanielsonLanczos<N/2,T> next;
public:
  void apply(T* data, int iSign) {
    next.apply(data, iSign);
    next.apply(data+N, iSign);

    T wtemp = iSign*std::sin(M_PI/N);
    T wpi = -iSign*std::sin(2*M_PI/N);
    T wpr = -2.0*wtemp*wtemp;
    T wr = 1.0;
    T wi = 0.0;

    for (unsigned i=0; i<N; i+=2) {
      int iN = i+N;

      T tempr = data[iN]*wr - data[iN+1]*wi;
      T tempi = data[iN]*wi + data[iN+1]*wr;

      data[iN] = data[i]-tempr;
      data[iN+1] = data[i+1]-tempi;
      data[i] += tempr;
      data[i+1] += tempi;

      wtemp = wr;
      wr += wr*wpr - wi*wpi;
      wi += wi*wpr + wtemp*wpi;
    }
  }
};


template<typename T>
class DanielsonLanczos<1,T>
{
public:
  void apply(T* data, int iSign) { }
};


/*!\brief Create a templated FFT/Inverse FFT object

  How to use this FFT
  FFT<LOG_LENGTH, double> gfft;

  unsigned long n = 1<<LOG_LENGTH;
  double* data = new double [2*n];

  gfft.fft(data);
  gfft.ifft(data);
*/
template<unsigned P,typename T=double>
class FFT
{
  enum { N = 1<<P };
  DanielsonLanczos<N,T> recursion;

  // reverse-binary reindexing
  void scramble(T* data) {
    int j=1;
    for (int i=1; i<2*N; i+=2) {
      if (j>i) {
        swap(&data[j-1], &data[i-1]);
        swap(&data[j], &data[i]);
      }
      int m = N;
      while (m>=2 && j>m) {
        j -= m;
        m >>= 1;
      }
      j += m;
    }
  }

  void rescale(T* data)
  {
    /*  scale all results by 1/n */
    T scale = static_cast<T>(1)/N;
    for (unsigned i=0; i<2*N; i++) {
      data[i] *= scale;
    }
  }

public:
  /*!\brief Replaces data[1..2*N] by its discrete Fourier transform */
  void fft(T* data) {
    scramble(data);
    recursion.apply(data,1);
  }

  /*!\brief Replaces data[1..2*N] by its inverse discrete Fourier transform */
  void ifft(T* data) {
    scramble(data);
    recursion.apply(data,-1);
    rescale(data);
  }
};


template<unsigned N>
class DanielsonLanczos<N, adouble>
{
  DanielsonLanczos<N/2,adouble> next;
public:
  void apply(aVector data, int iSign) {
    next.apply(data, iSign);
    next.apply(data(adept::range(N,adept::end)), iSign);

    adouble wtemp = iSign*std::sin(M_PI/N);
    adouble wpi = -iSign*std::sin(2*M_PI/N);
    adouble wpr = -2.0*wtemp*wtemp;
    adouble wr = 1.0;
    adouble wi = 0.0;

    for (unsigned i=0; i<N; i+=2) {
      int iN = i+N;

      adouble tempr = data(iN)*wr - data(iN+1)*wi;
      adouble tempi = data(iN)*wi + data(iN+1)*wr;

      data(iN) = data(i)-tempr;
      data(iN+1) = data(i+1)-tempi;
      data(i) += tempr;
      data(i+1) += tempi;

      wtemp = wr;
      wr += wr*wpr - wi*wpi;
      wi += wi*wpr + wtemp*wpi;
    }
  }
};

template<>
class DanielsonLanczos<1,adouble>
{
public:
  void apply(aVector data, int iSign) { }
};

template<unsigned P>
class FFT<P,adouble>
{
  enum { N = 1<<P };
  DanielsonLanczos<N,adouble> recursion;

  // reverse-binary reindexing
  void scramble(aVector data) {
    int j=1;
    for (int i=1; i<2*N; i+=2) {
      if (j>i) {
        swapad(data(j-1), data(i-1));
        swapad(data(j), data(i));
      }
      int m = N;
      while (m>=2 && j>m) {
        j -= m;
        m >>= 1;
      }
      j += m;
    }
  }

  void rescale(aVector data)
  {
    /*  scale all results by 1/n */
    adouble scale = static_cast<adouble>(1)/N;
    for (unsigned i=0; i<2*N; i++) {
      data(i) *= scale;
    }
  }

public:
  /*!\brief Replaces data[1..2*N] by its discrete Fourier transform */
  void fft(aVector data) {
    scramble(data);
    recursion.apply(data,1);
  }

  /*!\brief Replaces data[1..2*N] by its inverse discrete Fourier transform */
  void ifft(aVector data) {
    scramble(data);
    recursion.apply(data,-1);
    rescale(data);
  }
};

#endif /* _fft_h_ */
