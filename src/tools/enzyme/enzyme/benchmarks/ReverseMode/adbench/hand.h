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
#include <fstream>

float tdiff(struct timeval *start, struct timeval *end) {
  return (end->tv_sec-start->tv_sec) + 1e-6*(end->tv_usec-start->tv_usec);
}

using namespace std;
using json = nlohmann::json;

class HandModelLightMatrix
{
public:
    std::vector<std::string> bone_names;
    std::vector<int> parents; // assumimng that parent is earlier in the order of bones
    std::vector<LightMatrix<double>> base_relatives;
    std::vector<LightMatrix<double>> inverse_base_absolutes;
    LightMatrix<double> base_positions;
    LightMatrix<double> weights;
    std::vector<Triangle> triangles;
    bool is_mirrored;
};

class HandDataLightMatrix
{
public:
    HandModelLightMatrix model;
    std::vector<int> correspondences;
    LightMatrix<double> points;
};

struct HandInput
{
    std::vector<double> theta;
    HandDataLightMatrix data;
    std::vector<double> us;
};

struct HandOutput {
    std::vector<double> objective;
    int jacobian_ncols, jacobian_nrows;
    std::vector<double> jacobian;
};

struct HandParameters {
    bool is_complicated;
};

// Data for hand objective converted from input
struct HandObjectiveData
{
    int bone_count;
    const char** bone_names;
    const int* parents;             // assumimng that parent is earlier in the order of bones
    Matrix* base_relatives;
    Matrix* inverse_base_absolutes;
    Matrix base_positions;
    Matrix weights;
    const Triangle* triangles;
    int is_mirrored;
    int corresp_count;
    const int* correspondences;
    Matrix points;
};

extern "C" {
    void dhand_objective(
        double const* theta,
        double* dtheta,
        int bone_count,
        const char** bone_names,
        const int* parents,
        Matrix* base_relatives,
        Matrix* inverse_base_absolutes,
        Matrix* base_positions,
        Matrix* weights,
        const Triangle* triangles,
        int is_mirrored,
        int corresp_count,
        const int* correspondences,
        Matrix* points,
        double* err,
        double* derr
    );

    void dhand_objective_complicated(
        double const* theta,
        double* dtheta,
        double const* us,
        double* dus,
        int bone_count,
        const char** bone_names,
        const int* parents,
        Matrix* base_relatives,
        Matrix* inverse_base_absolutes,
        Matrix* base_positions,
        Matrix* weights,
        const Triangle* triangles,
        int is_mirrored,
        int corresp_count,
        const int* correspondences,
        Matrix* points,
        double* err,
        double* derr
    );

    void hand_objective_d(const double *theta, double *thetad, int
        bone_count, const char **bone_names, const int *parents, Matrix *
        base_relatives, Matrix *inverse_base_absolutes, Matrix *base_positions
        , Matrix *weights, const Triangle * /* TFIX */
        triangles, int is_mirrored, int corresp_count, const int *
        correspondences, Matrix *points, double *err, double *errd);

    void hand_objective_complicated_d(const double *theta, double *thetad,
        const double *us, double *usd, int bone_count, const char **
        bone_names, const int *parents, Matrix *base_relatives, Matrix *
        inverse_base_absolutes, Matrix *base_positions, /* TFIX */
        Matrix *weights, const Triangle *triangles, int
        is_mirrored, int corresp_count, const int *correspondences, Matrix *
        points, double *err, double *errd);

    void hand_objective_b(const double *theta, double *thetad, int
        bone_count, const char **bone_names, const int *parents, Matrix *
        base_relatives, Matrix *inverse_base_absolutes, Matrix *base_positions
        , Matrix *weights, const Triangle * /* TFIX */
        triangles, int is_mirrored, int corresp_count, const int *
        correspondences, Matrix *points, double *err, double *errd);
}

void read_hand_model(const string& path, HandModelLightMatrix* pmodel)
{
    const char DELIMITER = ':';
    auto& model = *pmodel;
    std::ifstream bones_in(path + "bones.txt");
    string s;
    while (bones_in.good())
    {
        getline(bones_in, s, DELIMITER);
        if (s.empty())
            continue;
        model.bone_names.push_back(s);
        getline(bones_in, s, DELIMITER);
        model.parents.push_back(std::stoi(s));
        double tmp[16];
        for (int i = 0; i < 16; i++)
        {
            getline(bones_in, s, DELIMITER);
            tmp[i] = std::stod(s);
        }
        model.base_relatives.emplace_back(4, 4);
        model.base_relatives.back().set(tmp);
        model.base_relatives.back().transpose_in_place();
        for (int i = 0; i < 15; i++)
        {
            getline(bones_in, s, DELIMITER);
            tmp[i] = std::stod(s);
        }
        getline(bones_in, s, '\n');
        tmp[15] = std::stod(s);
        model.inverse_base_absolutes.emplace_back(4, 4);
        model.inverse_base_absolutes.back().set(tmp);
        model.inverse_base_absolutes.back().transpose_in_place();
    }
    bones_in.close();
    int n_bones = (int)model.bone_names.size();

    std::ifstream vert_in(path + "vertices.txt");
    int n_vertices = 0;
    while (vert_in.good())
    {
        getline(vert_in, s);
        if (!s.empty())
            n_vertices++;
    }
    vert_in.close();

    model.base_positions.resize(4, n_vertices);
    model.base_positions.set_row(3, 1.);
    model.weights.resize(n_bones, n_vertices);
    model.weights.fill(0.);
    vert_in = std::ifstream(path + "vertices.txt");
    for (int i_vert = 0; i_vert < n_vertices; i_vert++)
    {
        for (int j = 0; j < 3; j++)
        {
            getline(vert_in, s, DELIMITER);
            model.base_positions(j, i_vert) = std::stod(s);
        }
        for (int j = 0; j < 3 + 2; j++)
        {
            getline(vert_in, s, DELIMITER); // skip
        }
        getline(vert_in, s, DELIMITER);
        int n = std::stoi(s);
        for (int j = 0; j < n; j++)
        {
            getline(vert_in, s, DELIMITER);
            int i_bone = std::stoi(s);
            if (j == n - 1)
                getline(vert_in, s, '\n');
            else
                getline(vert_in, s, DELIMITER);
            model.weights(i_bone, i_vert) = std::stod(s);
        }
    }
    vert_in.close();

    std::ifstream triangles_in(path + "triangles.txt");
    string ss[3];
    while (triangles_in.good())
    {
        getline(triangles_in, ss[0], DELIMITER);
        if (ss[0].empty())
            continue;

        getline(triangles_in, ss[1], DELIMITER);
        getline(triangles_in, ss[2], '\n');
        Triangle curr;
        for (int i = 0; i < 3; i++)
            curr.verts[i] = std::stoi(ss[i]);
        model.triangles.push_back(curr);
    }
    triangles_in.close();

    model.is_mirrored = false;
}

void read_hand_instance(const string& model_dir, const string& fn_in,
    vector<double>* theta, HandDataLightMatrix* data, vector<double>* us = nullptr)
{
    read_hand_model(model_dir, &data->model);
    std::ifstream in(fn_in);
    int n_pts, n_theta;
    in >> n_pts >> n_theta;
    data->correspondences.resize(n_pts);
    data->points.resize(3, n_pts);
    for (int i = 0; i < n_pts; i++)
    {
        in >> data->correspondences[i];
        for (int j = 0; j < 3; j++)
        {
            in >> data->points(j, i);
        }
    }
    if (us != nullptr)
    {
        us->resize(2 * n_pts);
        for (int i = 0; i < 2 * n_pts; i++)
        {
            in >> (*us)[i];
        }
    }
    theta->resize(n_theta);
    for (int i = 0; i < n_theta; i++)
    {
        in >> (*theta)[i];
    }
    in.close();
}

typedef void(*deriv_obj_t)(
    double const* theta,
    double* dtheta,
    int bone_count,
    const char** bone_names,
    const int* parents,
    Matrix* base_relatives,
    Matrix* inverse_base_absolutes,
    Matrix* base_positions,
    Matrix* weights,
    const Triangle* triangles,
    int is_mirrored,
    int corresp_count,
    const int* correspondences,
    Matrix* points,
    double* err,
    double* derr
);

typedef void(*deriv_obj_complicated_t)(
    double const* theta,
    double* dtheta,
    double const* us,
    double* dus,
    int bone_count,
    const char** bone_names,
    const int* parents,
    Matrix* base_relatives,
    Matrix* inverse_base_absolutes,
    Matrix* base_positions,
    Matrix* weights,
    const Triangle* triangles,
    int is_mirrored,
    int corresp_count,
    const int* correspondences,
    Matrix* points,
    double* err,
    double* derr
);


template<deriv_obj_t deriv_obj>
void calculate_jacobian_simple(HandObjectiveData* objective_input, struct HandInput &input, struct HandOutput &result, std::vector<double>& theta_d, std::vector<double>& us_d, std::vector<double>& us_jacobian_column)
{
    for (int i = 0; i < theta_d.size(); i++)
    {
        if (i > 0)
        {
            theta_d[i - 1] = 0.0;
        }

        theta_d[i] = 1.0;
        deriv_obj(
            input.theta.data(),
            theta_d.data(),
            objective_input->bone_count,
            objective_input->bone_names,
            objective_input->parents,
            objective_input->base_relatives,
            objective_input->inverse_base_absolutes,
            &objective_input->base_positions,
            &objective_input->weights,
            objective_input->triangles,
            objective_input->is_mirrored,
            objective_input->corresp_count,
            objective_input->correspondences,
            &objective_input->points,
            result.objective.data(),
            result.jacobian.data() + i * result.jacobian_nrows
        );
    }

    theta_d.back() = 0.0;
}

template<>
void calculate_jacobian_simple<dhand_objective>(HandObjectiveData* objective_input, struct HandInput &input, struct HandOutput &result, std::vector<double>& theta_d, std::vector<double>& us_d, std::vector<double>& us_jacobian_column)
{
    printf("manual reverse mode\n");

    std::vector<double> err(result.objective.size());
    for(int i=0; i<err.size(); i++) {
        err[i] = 1.0;
    }

    dhand_objective(
        input.theta.data(),
        result.jacobian.data(),
        objective_input->bone_count,
        objective_input->bone_names,
        objective_input->parents,
        objective_input->base_relatives,
        objective_input->inverse_base_absolutes,
        &objective_input->base_positions,
        &objective_input->weights,
        objective_input->triangles,
        objective_input->is_mirrored,
        objective_input->corresp_count,
        objective_input->correspondences,
        &objective_input->points,
        result.objective.data(),
        err.data()
    );
}

template<>
void calculate_jacobian_simple<hand_objective_d>(HandObjectiveData* objective_input, struct HandInput &input, struct HandOutput &result, std::vector<double>& theta_d, std::vector<double>& us_d, std::vector<double>& us_jacobian_column)
{
    printf("manual ad reverse mode\n");

    std::vector<double> err(result.objective.size());
    for(int i=0; i<err.size(); i++) {
        err[i] = 1.0;
    }

    hand_objective_b(
        input.theta.data(),
        result.jacobian.data(),
        objective_input->bone_count,
        objective_input->bone_names,
        objective_input->parents,
        objective_input->base_relatives,
        objective_input->inverse_base_absolutes,
        &objective_input->base_positions,
        &objective_input->weights,
        objective_input->triangles,
        objective_input->is_mirrored,
        objective_input->corresp_count,
        objective_input->correspondences,
        &objective_input->points,
        result.objective.data(),
        err.data()
    );
}

template<deriv_obj_complicated_t deriv_obj_complicated>
void calculate_jacobian_complicated(HandObjectiveData* objective_input, struct HandInput &input, struct HandOutput &result, std::vector<double>& theta_d, std::vector<double>& us_d, std::vector<double>& us_jacobian_column)
{
    int nrows = result.objective.size();
    int shift = 2 * nrows;

    // calculate theta jacobian part
    for (int i = 0; i < theta_d.size(); i++)
    {
        if (i > 0)
        {
            theta_d[i - 1] = 0.0;
        }

        theta_d[i] = 1.0;
        deriv_obj_complicated(
            input.theta.data(),
            theta_d.data(),
            input.us.data(),
            us_d.data(),
            objective_input->bone_count,
            objective_input->bone_names,
            objective_input->parents,
            objective_input->base_relatives,
            objective_input->inverse_base_absolutes,
            &objective_input->base_positions,
            &objective_input->weights,
            objective_input->triangles,
            objective_input->is_mirrored,
            objective_input->corresp_count,
            objective_input->correspondences,
            &objective_input->points,
            result.objective.data(),
            result.jacobian.data() + shift + i * nrows
        );
    }

    theta_d.back() = 0.0;

    // calculate us jacobian part
    for (int i = 0; i < us_d.size(); i++)
    {
        if (i > 0)
        {
            us_d[i - 1] = 0.0;
        }

        us_d[i] = 1.0;
        deriv_obj_complicated(
            input.theta.data(),
            theta_d.data(),
            input.us.data(),
            us_d.data(),
            objective_input->bone_count,
            objective_input->bone_names,
            objective_input->parents,
            objective_input->base_relatives,
            objective_input->inverse_base_absolutes,
            &objective_input->base_positions,
            &objective_input->weights,
            objective_input->triangles,
            objective_input->is_mirrored,
            objective_input->corresp_count,
            objective_input->correspondences,
            &objective_input->points,
            result.objective.data(),
            us_jacobian_column.data()
        );

        if (i % 2 == 0)
        {
            for (int j = 0; j < 3; j++)
            {
                result.jacobian[3 * (i / 2) + j] = us_jacobian_column[3 * (i / 2) + j];
            }
        }
        else
        {
            for (int j = 0; j < 3; j++)
            {
                result.jacobian[nrows + 3 * ((i - 1) / 2) + j] = us_jacobian_column[3 * ((i - 1) / 2) + j];
            }
        }
    }

    us_d.back() = 0.0;
}

template<deriv_obj_t deriv_obj, deriv_obj_complicated_t deriv_obj_complicated>
void calculate_jacobian(HandObjectiveData* objective_input, struct HandInput &input, struct HandOutput &result, bool complicated, std::vector<double>& theta_d, std::vector<double>& us_d, std::vector<double>& us_jacobian_column)
{
    if (complicated)
    {
        calculate_jacobian_complicated<deriv_obj_complicated>(objective_input, input, result, theta_d, us_d, us_jacobian_column);
    }
    else
    {
        calculate_jacobian_simple<deriv_obj>(objective_input, input, result, theta_d, us_d, us_jacobian_column);
    }
}

Matrix convert_to_matrix(const LightMatrix<double>& mat)
{
    return {
        mat.nrows_,
        mat.ncols_,
        mat.data_
    };
}

HandObjectiveData* convert_to_hand_objective_data(const HandInput& input)
{
    HandObjectiveData* result = new HandObjectiveData;

    result->correspondences = input.data.correspondences.data();
    result->corresp_count = input.data.correspondences.size();
    result->points = convert_to_matrix(input.data.points);

    const HandModelLightMatrix& imd = input.data.model;
    result->bone_count = imd.bone_names.size();
    result->parents = imd.parents.data();
    result->base_positions = convert_to_matrix(imd.base_positions);
    result->weights = convert_to_matrix(imd.weights);
    result->triangles = imd.triangles.data();
    result->is_mirrored = imd.is_mirrored ? 1 : 0;

    result->bone_names = new const char* [result->bone_count];
    result->base_relatives = new Matrix[result->bone_count];
    result->inverse_base_absolutes = new Matrix[result->bone_count];

    for (int i = 0; i < result->bone_count; i++)
    {
        result->bone_names[i] = imd.bone_names[i].data();
        result->base_relatives[i] = convert_to_matrix(imd.base_relatives[i]);
        result->inverse_base_absolutes[i] = convert_to_matrix(imd.inverse_base_absolutes[i]);
    }

    return result;
}

int main(const int argc, const char* argv[]) {
    printf("starting main\n");

    std::vector<std::string> paths = { "simple_small/hand1_t26_c100.txt" };

    const HandParameters params = { false }; // true or false

    std::ofstream jsonfile("results.json", std::ofstream::trunc);
    json test_results;

    for (auto path : paths) {
        printf("starting path %s\n", path.c_str());
        json test_suite;
        test_suite["name"] = path;
    // {

    // struct HandInput input;

    // const auto model_dir = filepath_to_dirname("data/" + path) + "model/";
    // // Read instance
    // if (params.is_complicated) {
    //     read_hand_instance(model_dir, "data/" + path, &input.theta, &input.data, &input.us);
    // }
    // else {
    //     read_hand_instance(model_dir, "data/" + path, &input.theta, &input.data);
    // }

    // //assert( (input.us.size() > 0) == params.is_complicated );

    // auto objective_input = convert_to_hand_objective_data(input);

    // int err_size = 3 * input.data.correspondences.size();
    // int ncols = input.theta.size();
    // if (params.is_complicated)
    // {
    //     ncols += 2;
    // }

    // struct HandOutput result = {
    //     std::vector<double>(err_size),
    //     ncols,
    //     err_size,
    //     std::vector<double>(err_size * ncols)
    // };

    // auto theta_d = std::vector<double>(input.theta.size());
    // auto us_d = std::vector<double>(input.us.size());
    // auto us_jacobian_column = std::vector<double>(err_size);

    // {
    //   struct timeval start, end;
    //   gettimeofday(&start, NULL);
    //   calculate_jacobian<hand_objective_d, hand_objective_complicated_d>(objective_input, input, result, params.is_complicated, theta_d, us_d, us_jacobian_column);
    //   gettimeofday(&end, NULL);
    //   json tapenade;
    //   tapenade["name"] = "Tapenade combined";
    //   tapenade["runtime"] = tdiff(&start, &end);
    //   for (unsigned i = 0; i < 5; i++) {
    //     printf("%f ", result.jacobian[i]);
    //     tapenade["result"].push_back(result.jacobian[i]);
    //   }
    //   test_suite["tools"].push_back(tapenade);
    //   printf("\n");
    // }

    // }

    {

    struct HandInput input;

    const auto model_dir = filepath_to_dirname("data/" + path) + "model/";
    // Read instance
    if (params.is_complicated) {
        read_hand_instance(model_dir, "data/" + path, &input.theta, &input.data, &input.us);
    }
    else {
        read_hand_instance(model_dir, "data/" + path, &input.theta, &input.data);
    }

    //assert( (input.us.size() > 0) == params.is_complicated );

    auto objective_input = convert_to_hand_objective_data(input);

    int err_size = 3 * input.data.correspondences.size();
    int ncols = input.theta.size();
    if (params.is_complicated)
    {
        ncols += 2;
    }

    struct HandOutput result = {
        std::vector<double>(err_size),
        ncols,
        err_size,
        std::vector<double>(err_size * ncols)
    };

    auto theta_d = std::vector<double>(input.theta.size());
    auto us_d = std::vector<double>(input.us.size());
    auto us_jacobian_column = std::vector<double>(err_size);

    {
      struct timeval start, end;
      gettimeofday(&start, NULL);
      calculate_jacobian<dhand_objective, dhand_objective_complicated>(objective_input, input, result, params.is_complicated, theta_d, us_d, us_jacobian_column);
      gettimeofday(&end, NULL);
      printf("Enzyme combined %0.6f\n", tdiff(&start, &end));
      json enzyme;
      enzyme["name"] = "Enzyme combined";
      enzyme["runtime"] = tdiff(&start, &end);
      for (unsigned i = 0; i < 5; i++) {
        printf("%f ", result.jacobian[i]);
        enzyme["result"].push_back(result.jacobian[i]);
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
