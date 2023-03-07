#pragma once

#include "../../json.hpp"
#include "../mshared/defs.h"
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
float tdiff(struct timeval *start, struct timeval *end) {
  return (end->tv_sec-start->tv_sec) + 1e-6*(end->tv_usec-start->tv_usec);
}

using namespace std;
using json = nlohmann::json;

struct LSTMInput
{
    int l;
    int c;
    int b;
    std::vector<double> main_params;
    std::vector<double> extra_params;
    std::vector<double> state;
    std::vector<double> sequence;
};

struct LSTMOutput {
    double objective;
    std::vector<double> gradient;
};

extern "C" {
    void dlstm_objective(
        int l,
        int c,
        int b,
        double const* main_params,
        double* dmain_params,
        double const* extra_params,
        double* dextra_params,
        double* state,
        double const* sequence,
        double* loss,
        double* dloss
    );

    void lstm_objective_b(int l, int c, int b, const double *main_params, double *
        main_paramsb, const double *extra_params, double *extra_paramsb,
        double *state, const double *sequence, double *loss, double *lossb);

    void adept_dlstm_objective(
        int l,
        int c,
        int b,
        double const* main_params,
        double* dmain_params,
        double const* extra_params,
        double* dextra_params,
        double* state,
        double const* sequence,
        double* loss,
        double* dloss
    );
}

void read_lstm_instance(const string& fn,
    int* l, int* c, int* b,
    vector<double>& main_params,
    vector<double>& extra_params,
    vector<double>& state,
    vector<double>& sequence)
{
    FILE* fid = fopen(fn.c_str(), "r");

    if (!fid) {
        printf("could not open file: %s\n", fn.c_str());
        exit(1);
    }

    fscanf(fid, "%i %i %i", l, c, b);

    int l_ = *l, c_ = *c, b_ = *b;

    int main_sz = 2 * l_ * 4 * b_;
    int extra_sz = 3 * b_;
    int state_sz = 2 * l_ * b_;
    int seq_sz = c_ * b_;

    main_params.resize(main_sz);
    extra_params.resize(extra_sz);
    state.resize(state_sz);
    sequence.resize(seq_sz);

    for (int i = 0; i < main_sz; i++) {
        fscanf(fid, "%lf", &main_params[i]);
    }

    for (int i = 0; i < extra_sz; i++) {
        fscanf(fid, "%lf", &extra_params[i]);
    }

    for (int i = 0; i < state_sz; i++) {
        fscanf(fid, "%lf", &state[i]);
    }

    for (int i = 0; i < c_ * b_; i++) {
        fscanf(fid, "%lf", &sequence[i]);
    }

    /*char ch;
    fscanf(fid, "%c", &ch);
    fscanf(fid, "%c", &ch);

    for (int i = 0; i < c_; i++) {
        unsigned char ch;
        fscanf(fid, "%c", &ch);
        int cb = ch;
        for (int j = b_ - 1; j >= 0; j--) {
            int p = pow(2, j);
            if (cb >= p) {
                sequence[(i + 1) * b_ - j - 1] = 1;
                cb -= p;
            }
            else {
                sequence[(i + 1) * b_ - j - 1] = 0;
            }
        }
    }*/

    fclose(fid);
}

typedef void(*deriv_t)(
        int l,
        int c,
        int b,
        double const* main_params,
        double* dmain_params,
        double const* extra_params,
        double* dextra_params,
        double* state,
        double const* sequence,
        double* loss,
        double* dloss
    );

template<deriv_t deriv>
void calculate_jacobian(struct LSTMInput &input, struct LSTMOutput &result)
{
    for(int i=0; i<100; i++) {

        double* main_params_gradient_part = result.gradient.data();
        double* extra_params_gradient_part = result.gradient.data() + input.main_params.size();

        double loss = 0.0;      // stores fictive result
                                // (Tapenade doesn't calculate an original function in reverse mode)

        double lossb = 1.0;     // stores dY
                                // (equals to 1.0 for gradient calculation)
        deriv(
            input.l,
            input.c,
            input.b,
            input.main_params.data(),
            main_params_gradient_part,
            input.extra_params.data(),
            extra_params_gradient_part,
            input.state.data(),
            input.sequence.data(),
            &loss,
            &lossb
        );
    }
}

int main(const int argc, const char* argv[]) {
    printf("starting main\n");

    std::vector<std::string> paths = { "lstm_l2_c1024.txt", "lstm_l4_c1024.txt", "lstm_l2_c4096.txt", "lstm_l4_c4096.txt" };
    
    std::ofstream jsonfile("results.json", std::ofstream::trunc);
    json test_results;

    for (auto path : paths) {
        printf("starting path %s\n", path.c_str());
        json test_suite;
        test_suite["name"] = path;
    {

    struct LSTMInput input;

    // Read instance
    read_lstm_instance("data/" + path, &input.l, &input.c, &input.b, input.main_params, input.extra_params, input.state,
                       input.sequence);

    for(unsigned i=0; i<5; i++) {
      printf("%f ", input.state[i]);
    }
    printf("\n");

    int Jcols = 8 * input.l * input.b + 3 * input.b;
    struct LSTMOutput result = { 0, std::vector<double>(Jcols) };

    {
      struct timeval start, end;
      gettimeofday(&start, NULL);
      calculate_jacobian<lstm_objective_b>(input, result);
      gettimeofday(&end, NULL);
      printf("Tapenade combined %0.6f\n", tdiff(&start, &end));
      json tapenade;
      tapenade["name"] = "Tapenade combined";
      tapenade["runtime"] = tdiff(&start, &end);
      for (unsigned i = result.gradient.size() - 5;
           i < result.gradient.size(); i++) {
        printf("%f ", result.gradient[i]);
        tapenade["result"].push_back(result.gradient[i]);
      }
      test_suite["tools"].push_back(tapenade);
      printf("\n");
    }

    }

    {

    struct LSTMInput input = {};

    // Read instance
    read_lstm_instance("data/" + path, &input.l, &input.c, &input.b, input.main_params, input.extra_params, input.state,
                       input.sequence);

    std::vector<double> state = std::vector<double>(input.state.size());

    int Jcols = 8 * input.l * input.b + 3 * input.b;
    struct LSTMOutput result = { 0, std::vector<double>(Jcols) };

    {
      struct timeval start, end;
      gettimeofday(&start, NULL);
      calculate_jacobian<adept_dlstm_objective>(input, result);
      gettimeofday(&end, NULL);
      printf("Adept combined %0.6f\n", tdiff(&start, &end));
      json adept;
      adept["name"] = "Adept combined";
      adept["runtime"] = tdiff(&start, &end);
      for (unsigned i = result.gradient.size() - 5;
           i < result.gradient.size(); i++) {
        printf("%f ", result.gradient[i]);
        adept["result"].push_back(result.gradient[i]);
      }
      test_suite["tools"].push_back(adept);
      printf("\n");
    }

    }

    {

    struct LSTMInput input = {};

    // Read instance
    read_lstm_instance("data/" + path, &input.l, &input.c, &input.b, input.main_params, input.extra_params, input.state,
                       input.sequence);

    std::vector<double> state = std::vector<double>(input.state.size());

    int Jcols = 8 * input.l * input.b + 3 * input.b;
    struct LSTMOutput result = { 0, std::vector<double>(Jcols) };

    {
      struct timeval start, end;
      gettimeofday(&start, NULL);
      calculate_jacobian<dlstm_objective>(input, result);
      gettimeofday(&end, NULL);
      printf("Enzyme combined %0.6f\n", tdiff(&start, &end));
      json enzyme;
       enzyme["name"] = "Enzyme combined";
       enzyme["runtime"] = tdiff(&start, &end);
       for (unsigned i = result.gradient.size() - 5;
            i < result.gradient.size(); i++) {
         printf("%f ", result.gradient[i]);
         enzyme["result"].push_back(result.gradient[i]);
       }
       test_suite["tools"].push_back(enzyme);
       
       printf("\n");
    }

    }
    test_suite["llvm-version"] = __clang_version__;
    test_suite["mode"] = "ReverseMode";
    test_suite["batch-size"] = 1;
    test_results.push_back(test_suite);
   }
   jsonfile << std::setw(4) << test_results;
}
