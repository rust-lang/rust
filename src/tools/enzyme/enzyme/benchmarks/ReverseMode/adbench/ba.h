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

struct BAInput {
    int n = 0, m = 0, p = 0;
    std::vector<double> cams, X, w, feats;
    std::vector<int> obs;
};

// rows is nrows+1 vector containing
// indices to cols and vals.
// rows[i] ... rows[i+1]-1 are elements of i-th row
// i.e. cols[row[i]] is the column of the first
// element in the row. Similarly for values.
class BASparseMat
{
public:
    int n, m, p; // number of cams, points and observations
    int nrows, ncols;
    std::vector<int> rows;
    std::vector<int> cols;
    std::vector<double> vals;

    BASparseMat();
    BASparseMat(int n_, int m_, int p_);

    void insert_reproj_err_block(int obsIdx,
        int camIdx, int ptIdx, const double* const J);

    void insert_w_err_block(int wIdx, double w_d);

    void clear();
};

BASparseMat::BASparseMat() {}

BASparseMat::BASparseMat(int n_, int m_, int p_) : n(n_), m(m_), p(p_)
{
    nrows = 2 * p + p;
    ncols = BA_NCAMPARAMS * n + 3 * m + p;
    rows.reserve(nrows + 1);
    int nnonzero = (BA_NCAMPARAMS + 3 + 1) * 2 * p + p;
    cols.reserve(nnonzero);
    vals.reserve(nnonzero);
    rows.push_back(0);
}

void BASparseMat::insert_reproj_err_block(int obsIdx,
    int camIdx, int ptIdx, const double* const J)
{
    int n_new_cols = BA_NCAMPARAMS + 3 + 1;
    rows.push_back(rows.back() + n_new_cols);
    rows.push_back(rows.back() + n_new_cols);

    for (int i_row = 0; i_row < 2; i_row++)
    {
        for (int i = 0; i < BA_NCAMPARAMS; i++)
        {
            cols.push_back(BA_NCAMPARAMS * camIdx + i);
            vals.push_back(J[2 * i + i_row]);
        }
        int col_offset = BA_NCAMPARAMS * n;
        int val_offset = BA_NCAMPARAMS * 2;
        for (int i = 0; i < 3; i++)
        {
            cols.push_back(col_offset + 3 * ptIdx + i);
            vals.push_back(J[val_offset + 2 * i + i_row]);
        }
        col_offset += 3 * m;
        val_offset += 3 * 2;
        cols.push_back(col_offset + obsIdx);
        vals.push_back(J[val_offset + i_row]);
    }
}

void BASparseMat::insert_w_err_block(int wIdx, double w_d)
{
    rows.push_back(rows.back() + 1);
    cols.push_back(BA_NCAMPARAMS * n + 3 * m + wIdx);
    vals.push_back(w_d);
}

void BASparseMat::clear()
{
    rows.clear();
    cols.clear();
    vals.clear();
    rows.reserve(nrows + 1);
    int nnonzero = (BA_NCAMPARAMS + 3 + 1) * 2 * p + p;
    cols.reserve(nnonzero);
    vals.reserve(nnonzero);
    rows.push_back(0);
}

struct BAOutput {
    std::vector<double> reproj_err;
    std::vector<double> w_err;
    BASparseMat J;
};

extern "C" {
    void ba_objective(
        int n,
        int m,
        int p,
        double const* cams,
        double const* X,
        double const* w,
        int const* obs,
        double const* feats,
        double* reproj_err,
        double* w_err
    );

    void dcompute_reproj_error(
        double const* cam,
        double * dcam,
        double const* X,
        double * dX,
        double const* w,
        double * wb,
        double const* feat,
        double *err,
        double *derr
    );

    void dcompute_zach_weight_error(double const* w, double* dw, double* err, double* derr);

    void compute_reproj_error_b(
        double const* cam,
        double * dcam,
        double const* X,
        double * dX,
        double const* w,
        double * wb,
        double const* feat,
        double *err,
        double *derr
    );

    void compute_zach_weight_error_b(double const* w, double* dw, double* err, double* derr);

    void adept_compute_reproj_error(
        double const* cam,
        double * dcam,
        double const* X,
        double * dX,
        double const* w,
        double * wb,
        double const* feat,
        double *err,
        double *derr
    );

    void adept_compute_zach_weight_error(double const* w, double* dw, double* err, double* derr);
}

void read_ba_instance(const string& fn,
    int& n, int& m, int& p,
    vector<double>& cams,
    vector<double>& X,
    vector<double>& w,
    vector<int>& obs,
    vector<double>& feats)
{
    FILE* fid = fopen(fn.c_str(), "r");
    if (!fid) {
        printf("could not open file: %s\n", fn.c_str());
        exit(1);
    }
    std::cout << "read_ba_instance: opened " << fn << std::endl;

    fscanf(fid, "%i %i %i", &n, &m, &p);
    int nCamParams = 11;

    cams.resize(nCamParams * n);
    X.resize(3 * m);
    w.resize(p);
    obs.resize(2 * p);
    feats.resize(2 * p);

    for (int j = 0; j < nCamParams; j++)
        fscanf(fid, "%lf", &cams[j]);
    for (int i = 1; i < n; i++)
        memcpy(&cams[i * nCamParams], &cams[0], nCamParams * sizeof(double));

    for (int j = 0; j < 3; j++)
        fscanf(fid, "%lf", &X[j]);
    for (int i = 1; i < m; i++)
        memcpy(&X[i * 3], &X[0], 3 * sizeof(double));

    fscanf(fid, "%lf", &w[0]);
    for (int i = 1; i < p; i++)
        w[i] = w[0];

    int camIdx = 0;
    int ptIdx = 0;
    for (int i = 0; i < p; i++)
    {
        obs[i * 2 + 0] = (camIdx++ % n);
        obs[i * 2 + 1] = (ptIdx++ % m);
    }

    fscanf(fid, "%lf %lf", &feats[0], &feats[1]);
    for (int i = 1; i < p; i++)
    {
        feats[i * 2 + 0] = feats[0];
        feats[i * 2 + 1] = feats[1];
    }

    fclose(fid);
}

typedef void(*deriv_reproj_t)(double const*, double *, double const*, double*, double const*, double*, double const*, double *, double *);

template<deriv_reproj_t deriv_reproj>
void calculate_reproj_error_jacobian_part(struct BAInput &input, struct BAOutput &result, std::vector<double> &reproj_err_d, std::vector<double> &reproj_err_d_row)
{
    double errb[2];     // stores dY
                        // (i-th element equals to 1.0 for calculating i-th jacobian row)

    double err[2];      // stores fictive result
                        // (Tapenade doesn't calculate an original function in reverse mode)

    double* cam_gradient_part = reproj_err_d_row.data();
    double* x_gradient_part = reproj_err_d_row.data() + BA_NCAMPARAMS;
    double* weight_gradient_part = reproj_err_d_row.data() + BA_NCAMPARAMS + 3;

    for (int i = 0; i < input.p; i++)
    {
        int camIdx = input.obs[2 * i + 0];
        int ptIdx = input.obs[2 * i + 1];

        // calculate first row
        errb[0] = 1.0;
        errb[1] = 0.0;

        for(auto& a : reproj_err_d_row) a = 0.0;

        deriv_reproj(
            &input.cams[camIdx * BA_NCAMPARAMS],
            cam_gradient_part,
            &input.X[ptIdx * 3],
            x_gradient_part,
            &input.w[i],
            weight_gradient_part,
            &input.feats[i * 2],
            err,
            errb
        );

        // fill first row elements
        for (int j = 0; j < BA_NCAMPARAMS + 3 + 1; j++)
        {
            reproj_err_d[2 * j] = reproj_err_d_row[j];
        }

        for(auto& a : reproj_err_d_row) a = 0.0;
        // calculate second row
        errb[0] = 0.0;
        errb[1] = 1.0;
        deriv_reproj(
            &input.cams[camIdx * BA_NCAMPARAMS],
            cam_gradient_part,
            &input.X[ptIdx * 3],
            x_gradient_part,
            &input.w[i],
            weight_gradient_part,
            &input.feats[i * 2],
            err,
            errb
        );

        // fill second row elements
        for (int j = 0; j < BA_NCAMPARAMS + 3 + 1; j++)
        {
            reproj_err_d[2 * j + 1] = reproj_err_d_row[j];
        }

        result.J.insert_reproj_err_block(i, camIdx, ptIdx, reproj_err_d.data());
    }
}



typedef void(*deriv_weight_t)(double const* w, double* dw, double* err, double* derr);

template<deriv_weight_t deriv_weight>
void calculate_weight_error_jacobian_part(struct BAInput &input, struct BAOutput &result, std::vector<double> &reproj_err_d, std::vector<double> &reproj_err_d_row)
{
    for (int j = 0; j < input.p; j++)
    {
        // NOTE added set of 0 here
        double wb = 0.0;         // stores calculated derivative

        double err = 0.0;       // stores fictive result
                                // (Tapenade doesn't calculate an original function in reverse mode)

        double errb = 1.0;      // stores dY
                                // (equals to 1.0 for derivative calculation)

        deriv_weight(&input.w[j], &wb, &err, &errb);
        result.J.insert_w_err_block(j, wb);
    }
}


template<deriv_reproj_t deriv_reproj, deriv_weight_t deriv_weight>
void calculate_jacobian(struct BAInput &input, struct BAOutput &result)
{
    auto reproj_err_d = std::vector<double>(2 * (BA_NCAMPARAMS + 3 + 1));
    auto reproj_err_d_row = std::vector<double>(BA_NCAMPARAMS + 3 + 1);

    calculate_reproj_error_jacobian_part<deriv_reproj>(input, result, reproj_err_d, reproj_err_d_row);
    calculate_weight_error_jacobian_part<deriv_weight>(input, result, reproj_err_d, reproj_err_d_row);
}

int main(const int argc, const char* argv[]) {
    std::string path = "/mnt/Data/git/Enzyme/apps/ADBench/data/ba/ba1_n49_m7776_p31843.txt";

    std::vector<std::string> paths = {
        "ba10_n1197_m126327_p563734.txt",  "ba14_n356_m226730_p1255268.txt",   "ba18_n1936_m649673_p5213733.txt",    "ba2_n21_m11315_p36455.txt",    "ba6_n539_m65220_p277273.txt",  "test.txt",
        "ba11_n1723_m156502_p678718.txt",  "ba15_n1102_m780462_p4052340.txt",  "ba19_n4585_m1324582_p9125125.txt",   "ba3_n161_m48126_p182072.txt",  "ba7_n93_m61203_p287451.txt",
        "ba12_n253_m163691_p899155.txt",   "ba16_n1544_m942409_p4750193.txt",  "ba1_n49_m7776_p31843.txt",           "ba4_n372_m47423_p204472.txt",  "ba8_n88_m64298_p383937.txt",
        "ba13_n245_m198739_p1091386.txt",  "ba17_n1778_m993923_p5001946.txt",  "ba20_n13682_m4456117_p2987644.txt",  "ba5_n257_m65132_p225911.txt",  "ba9_n810_m88814_p393775.txt",
    };

    std::ofstream jsonfile("results.json", std::ofstream::trunc);
    json test_results;

    for (auto path : paths) {
      json test_suite;
      test_suite["name"] = path;
    {

    struct BAInput input;
    read_ba_instance("data/" + path, input.n, input.m, input.p, input.cams, input.X, input.w, input.obs, input.feats);

    struct BAOutput result = {
        std::vector<double>(2 * input.p),
        std::vector<double>(input.p),
        BASparseMat(input.n, input.m, input.p)
    };

    //BASparseMat(this->input.n, this->input.m, this->input.p)

    /*
    ba_objective(
        input.n,
        input.m,
        input.p,
        input.cams.data(),
        input.X.data(),
        input.w.data(),
        input.obs.data(),
        input.feats.data(),
        result.reproj_err.data(),
        result.w_err.data()
    );

    for(unsigned i=0; i<input.p; i++) {
        //printf("w_err[%d]=%f reproj_err[%d]=%f, reproj_err[%d]=%f\n", i, result.w_err[i], 2*i, result.reproj_err[2*i], 2*i+1, result.reproj_err[2*i+1]);
    }
    */

    {
      struct timeval start, end;
      gettimeofday(&start, NULL);
      calculate_jacobian<compute_reproj_error_b, compute_zach_weight_error_b>(input, result);
      gettimeofday(&end, NULL);
      printf("Tapenade combined %0.6f\n", tdiff(&start, &end));
      json tapenade;
      tapenade["name"] = "Tapenade combined";
      tapenade["runtime"] = tdiff(&start, &end);
      for(unsigned i=0; i<5; i++) {
        printf("%f ", result.J.vals[i]);
        tapenade["result"].push_back(result.J.vals[i]);
      }
      printf("\n");
      test_suite["tools"].push_back(tapenade);
    }

    }

    {

    struct BAInput input;
    read_ba_instance("data/" + path, input.n, input.m, input.p, input.cams, input.X, input.w, input.obs, input.feats);

    struct BAOutput result = {
        std::vector<double>(2 * input.p),
        std::vector<double>(input.p),
        BASparseMat(input.n, input.m, input.p)
    };

    //BASparseMat(this->input.n, this->input.m, this->input.p)

    /*
    ba_objective(
        input.n,
        input.m,
        input.p,
        input.cams.data(),
        input.X.data(),
        input.w.data(),
        input.obs.data(),
        input.feats.data(),
        result.reproj_err.data(),
        result.w_err.data()
    );

    for(unsigned i=0; i<input.p; i++) {
        //printf("w_err[%d]=%f reproj_err[%d]=%f, reproj_err[%d]=%f\n", i, result.w_err[i], 2*i, result.reproj_err[2*i], 2*i+1, result.reproj_err[2*i+1]);
    }
    */

    {
      struct timeval start, end;
      gettimeofday(&start, NULL);
      calculate_jacobian<adept_compute_reproj_error, adept_compute_zach_weight_error>(input, result);
      gettimeofday(&end, NULL);
      printf("Adept combined %0.6f\n", tdiff(&start, &end));
      json adept;
      adept["name"] = "Adept combined";
      adept["runtime"] = tdiff(&start, &end);
      for(unsigned i=0; i<5; i++) {
        printf("%f ", result.J.vals[i]);
        adept["result"].push_back(result.J.vals[i]);
      }
      printf("\n");
      test_suite["tools"].push_back(adept);
    }

    }

    {

    struct BAInput input;
    read_ba_instance("data/" + path, input.n, input.m, input.p, input.cams, input.X, input.w, input.obs, input.feats);

    struct BAOutput result = {
        std::vector<double>(2 * input.p),
        std::vector<double>(input.p),
        BASparseMat(input.n, input.m, input.p)
    };

    //BASparseMat(this->input.n, this->input.m, this->input.p)

    /*
    ba_objective(
        input.n,
        input.m,
        input.p,
        input.cams.data(),
        input.X.data(),
        input.w.data(),
        input.obs.data(),
        input.feats.data(),
        result.reproj_err.data(),
        result.w_err.data()
    );

    for(unsigned i=0; i<input.p; i++) {
        //printf("w_err[%d]=%f reproj_err[%d]=%f, reproj_err[%d]=%f\n", i, result.w_err[i], 2*i, result.reproj_err[2*i], 2*i+1, result.reproj_err[2*i+1]);
    }
    */

    {
      struct timeval start, end;
      gettimeofday(&start, NULL);
      calculate_jacobian<dcompute_reproj_error, dcompute_zach_weight_error>(input, result);
      gettimeofday(&end, NULL);
      printf("Enzyme combined %0.6f\n", tdiff(&start, &end));
      json enzyme;
      enzyme["name"] = "Enzyme combined";
      enzyme["runtime"] = tdiff(&start, &end);
      for(unsigned i=0; i<5; i++) {
        printf("%f ", result.J.vals[i]);
        enzyme["result"].push_back(result.J.vals[i]);
      }
      printf("\n");
      test_suite["tools"].push_back(enzyme);
    }

    }
    test_suite["llvm-version"] = __clang_version__;
    test_suite["mode"] = "ReverseMode";
    test_suite["batch-size"] = 1;
    test_results.push_back(test_suite);
   }
   jsonfile << std::setw(4) << test_results;
}
