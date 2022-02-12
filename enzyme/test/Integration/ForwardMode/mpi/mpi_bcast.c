// RUN: %clang -std=c11 -O0 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -std=c11 -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -std=c11 -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -std=c11 -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -std=c11 -O0 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme --enzyme-inline=1 -S | %lli - 
// RUN: %clang -std=c11 -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme --enzyme-inline=1 -S | %lli - 
// RUN: %clang -std=c11 -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme --enzyme-inline=1 -S | %lli - 
// RUN: %clang -std=c11 -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme --enzyme-inline=1 -S | %lli - 

#include <stdio.h>
#include <math.h>
#include <assert.h>

#include "test_utils.h"

#include <mpi.h>

// XFAIL: *

double __enzyme_fwddiff(void*, ...);

double mpi_bcast_test(double b, int n, int rank, int numprocs) {
  #define n 1 
  
  //double a[n];
  //for (int i=0; i<n; i++) a[i] = b;
  double a = b;
  
  //memcpy(buf, a, sizeof(double)*n);
  MPI_Bcast(&a,n,MPI_DOUBLE,0,MPI_COMM_WORLD);

  //double sum = 0;

  //for(int i=0;i<n;i++)
  //    sum += pow(a[i], rank+1);

  //sum = a[0];
  
  printf("end ran %f %d/%d\n", a, rank, numprocs);
  fflush(0);
  return a;
}

MPI_Op op = MPI_SUM;

MPI_Op operation;

void my_sum_function(void* inputBuffer, void* outputBuffer, int* len, MPI_Datatype* datatype)
{
    int* input = (int*)inputBuffer;
    int* output = (int*)outputBuffer;

    for(int i = 0; i < *len; i++)
    {
        output[i] += input[i];
    }
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  double h=1e-6;
  if(argc<2) {
    printf("Not enough arguments. Missing problem size.");
    MPI_Finalize();
    return 0;
  }
  int numprocs;
  MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
  int N=10;
  
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MPI_Op_create(&my_sum_function, 1, &operation);

  float local_sum = 0;

// Reduce all of the local sums into the global sum
float global_sum;
MPI_Reduce(&local_sum, &global_sum, 1, MPI_FLOAT, operation, 0,
           MPI_COMM_WORLD);


  double res = __enzyme_fwddiff((void*)mpi_bcast_test, 10.0+rank, 1.0, N, rank, numprocs);
  printf("res=%f rank=%d\n", res, rank);
  fflush(0);
  MPI_Finalize();
}

