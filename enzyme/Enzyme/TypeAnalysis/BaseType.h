//===- BaseType.h - Category of type used in Type Analysis    ------------===//
//
//                             Enzyme Project
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// If using this code in an academic setting, please cite the following:
// @misc{enzymeGithub,
//  author = {William S. Moses and Valentin Churavy},
//  title = {Enzyme: High Performance Automatic Differentiation of LLVM},
//  year = {2020},
//  howpublished = {\url{https://github.com/wsmoses/Enzyme}},
//  note = {commit xxxxxxx}
// }
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of an enum representing the potential
// types used in Type Analysis
//
//===----------------------------------------------------------------------===//
#ifndef ENZYME_TYPE_ANALYSIS_BASE_TYPE_H
#define ENZYME_TYPE_ANALYSIS_BASE_TYPE_H 1

#include <string>
#include "llvm/Support/ErrorHandling.h"

enum class BaseType {
  // integral type
  Integer,
  // floating point
  Float,
  // pointer
  Pointer,
  // can be anything of users choosing [usually result of a constant]
  Anything,
  // insufficient information
  Unknown
};

static inline std::string to_string(BaseType t) {
  switch (t) {
  case BaseType::Integer:
    return "Integer";
  case BaseType::Float:
    return "Float";
  case BaseType::Pointer:
    return "Pointer";
  case BaseType::Anything:
    return "Anything";
  case BaseType::Unknown:
    return "Unknown";
  }
  llvm_unreachable("unknown inttype");
}

static inline BaseType parseBaseType(std::string str) {
  if (str == "Integer")
    return BaseType::Integer;
  if (str == "Float")
    return BaseType::Float;
  if (str == "Pointer")
    return BaseType::Pointer;
  if (str == "Anything")
    return BaseType::Anything;
  if (str == "Unknown")
    return BaseType::Unknown;
  llvm_unreachable("Unknown BaseType string");
}
#endif