// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>
#include <math.h>

#include "../mshared/defs.h"

// theta: 26 [global rotation, global translation, finger parameters (4*5)]
// bone_count, bone_names, parents, base_relatives, inverse_base_absolutes,
// base_positions, weights, triangles, is_mirrored, corresp_count, correspondencies: data measurements and hand model
// err: 3*number_of_correspondences
void hand_objective(
    double const* theta,
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
    double* err
);



// theta: 26 [global rotation, global translation, finger parameters (4*5)]
// us: 2*number_of_correspondences
// bone_count, bone_names, parents, base_relatives, inverse_base_absolutes,
// base_positions, weights, triangles, is_mirrored, corresp_count, correspondencies: data measurements and hand model
// err: 3*number_of_correspondences
void hand_objective_complicated(
    double const* theta,
    double const* us,
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
    double* err
);

#ifdef __cplusplus
}
#endif
