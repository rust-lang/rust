// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#pragma once

typedef struct
{
    double gamma;
    int m;
} Wishart;

typedef struct
{
    int verts[3];
} Triangle;

#ifndef NBDirsMax
#define NBDirsMax 1650
#endif

#define NBDirsMaxReproj_BV 2

#ifndef PI
#define PI 3.14159265359
#endif

#define BA_NCAMPARAMS 11
#define BA_ROT_IDX 0
#define BA_C_IDX 3
#define BA_F_IDX 6
#define BA_X0_IDX 7
#define BA_RAD_IDX 9

//# Flexion, Abduction, Twist = 'xzy'
#define HAND_XYZ_TO_ROTATIONAL_PARAMETERIZATION {0, 2, 1}

#ifdef __cplusplus
#include <vector>
#include <dirent.h>
#include <string>
#include <string.h>

inline bool ends_with(std::string const & value, std::string const & ending)
{
    if (ending.size() > value.size()) return false;
    return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

void getTests(std::vector<std::string> &tests, const char *name="data", std::string indent="") {
    DIR *dir;
    struct dirent *entry;

    if (!(dir = opendir(name)))
        return;

    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type == DT_DIR) {
            char path[1024];
            if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
                continue;
            snprintf(path, sizeof(path), "%s/%s", name, entry->d_name);
            //printf("%*s[%s]\n", indent, "", entry->d_name);
            getTests(tests, path, indent + entry->d_name + "/");
        } else if (ends_with(std::string(entry->d_name),".txt")){
            tests.push_back(indent + entry->d_name);
        }
    }
    closedir(dir);
}

#endif
