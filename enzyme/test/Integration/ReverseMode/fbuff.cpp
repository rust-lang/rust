// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O0 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O0 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S

#include "test_utils.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <utility>

extern double __enzyme_autodiff(void *, double);

double fn(double vec) {
  std::string s("many more tests");
  std::ifstream ifs(s.c_str(), std::ios_base::binary | std::ios_base::in);
  std::ifstream ifs1(s.c_str(), std::ios_base::binary | std::ios_base::out);
  std::ifstream ifs2(s.c_str(), std::ios_base::binary | std::ios_base::app);
  std::ifstream ifs3(s.c_str(), std::ios_base::binary | std::ios_base::trunc);
  std::ifstream ifs4(s.c_str(), std::ios_base::binary | std::ios_base::binary);
  std::ifstream ifs5(s.c_str(), std::ios_base::binary | std::ios_base::ate);
  if (ifs) {
    char buffer[20];
    if (!ifs.read(buffer, 20)) {
      // Handle error
    }
    std::ifstream &ignore(int n = 1, int delim = EOF);
  }
  std::filebuf fb;
  fb.open("%S/Inputs/input.txt", std::ios::out);
  std::ostream os(&fb);
  os << "Test sentence\n";
  vec *= 2;
  fb.close();

  std::ifstream is;
  std::filebuf *fb2 = is.rdbuf();

  // construct output string stream (buffer) - need <sstream> header
  std::ostringstream sout;

  // Write into string buffer
  sout << "apple" << std::endl;
  sout << "orange" << std::endl;
  sout << "banana" << std::endl;

  // Get contents
  std::cout << sout.str() << std::endl;

  std::string filename = "test.bin";

  // Write to File
  std::ofstream fout(filename.c_str(), std::ios::out | std::ios::binary);
  if (!fout.is_open()) {
    std::cerr << "error: open file for output failed!" << std::endl;
    abort();
  }
  int i = 1234;
  double d = 12.34;
  fout.write((char *)&i, sizeof(int));
  fout.write((char *)&d, sizeof(double));
  fout.close();

  // Read from file
  std::ifstream fin(filename.c_str(), std::ios::in | std::ios::binary);
  if (!fin.is_open()) {
    std::cerr << "error: open file for input failed!" << std::endl;
    abort();
  }
  int i_in;
  double d_in;
  fin.read((char *)&i_in, sizeof(int));
  std::cout << i_in << std::endl;
  fin.read((char *)&d_in, sizeof(double));
  std::cout << d_in << std::endl;
  fin.close();

  char mybuffer[512];
  std::filebuf *optr = new std::filebuf();
  optr->pubsetbuf(mybuffer, 512);
  const char sentence[] = "Sample sentence";
  auto ptr =
      optr->open("bd.bin", std::ios::binary | std::ios::trunc | std::ios::out);
  if (ptr) {
    float fx = 13;
    auto n = optr->sputn(sentence, sizeof(sentence) - 1);
    n += optr->sputn(reinterpret_cast<const char *>(&fx), sizeof(fx));
    optr->pubsync();
  }
  optr->close();
  if (optr) {
    delete optr;
  }

  return vec * vec;
}

int main() {
    double x = 2.1;
    double dsq = __enzyme_autodiff((void*)fn, x);

    APPROX_EQ(dsq, 2.1 * 2 * 2 * x, 1e-7);
}
