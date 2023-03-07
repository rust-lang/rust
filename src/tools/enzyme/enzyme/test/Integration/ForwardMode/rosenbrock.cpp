// RUN: %clang++ -std=c++11 -fno-exceptions -O0 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli -
// RUN: %clang++ -std=c++11 -fno-exceptions -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli -
// RUN: %clang++ -std=c++11 -fno-exceptions -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli -
// RUN: %clang++ -std=c++11 -fno-exceptions -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli -
// RUN: %clang++ -std=c++11 -fno-exceptions -O0 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli -
// RUN: %clang++ -std=c++11 -fno-exceptions -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli -
// RUN: %clang++ -std=c++11 -fno-exceptions -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli -
// RUN: %clang++ -std=c++11 -fno-exceptions -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli -

#include "test_utils.h"
#include <cmath>
#include <vector>

int enzyme_dup;
int enzyme_out;
int enzyme_const;

using std::vector;

template <typename T> T rosenbrock2(const vector<T> &control) {
  T result(0);
  for (std::size_t i = 0; i < control.size(); i += 2) {
    T c1 = (control[i + 1] - control[i] * control[i]);
    T c2 = 1.0 - control[i];
    result += 100 * c1 * c1 + c2 * c2;
  }
  return result;
};

extern double __enzyme_fwddiff(double (*)(const vector<double> &),
                               vector<double> &, vector<double> &);

////////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[]) {
  using S = double;
  vector<S> control{1.5, 2.5, 3.5, 4.5};
  vector<S> results{-149.000000, 50.000000, 10855.000000, -1550.000000};
  for (size_t i = 0; i < control.size(); i++) {
    vector<S> activity(4);
    activity[i] = 1.0;
    double dret = __enzyme_fwddiff(rosenbrock2<S>, control, activity);
    APPROX_EQ(results[i], dret, 1e-10)
  }

  return 0;
}
