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

void __enzyme_fwddiff(void*, ...);

void mpi_reduce_test(double *b, double *global_sum, int n, int rank, int numprocs) {
    MPI_Reduce(b, global_sum, 1, MPI_DOUBLE, MPI_SUM, 0,
           MPI_COMM_WORLD);
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  double h=1e-6;
  int numprocs;
  MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
  int N=10;
  
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  double sum;
  double dsum = (rank == 0) ? 1.0 : 123456789;
  double b = 10.0+rank;
  double db = 0;
  __enzyme_fwddiff((void*)mpi_reduce_test, &b, &db, &sum, &dsum, N, rank, numprocs);
  printf("dsum=%f db=%f rank=%d\n", dsum, db, rank);
  fflush(0);
  MPI_Finalize();
}

