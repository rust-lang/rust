//===- BaseType.h - Category of type used in Type Analysis    ------------===//
//
//                             Enzyme Project
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// If using this code in an academic setting, please cite the following:
// @incollection{enzymeNeurips,
// title = {Instead of Rewriting Foreign Code for Machine Learning,
//          Automatically Synthesize Fast Gradients},
// author = {Moses, William S. and Churavy, Valentin},
// booktitle = {Advances in Neural Information Processing Systems 33},
// year = {2020},
// note = {To appear in},
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

#include "llvm/Support/ErrorHandling.h"
#include <string>

/// Categories of potential types
enum class BaseType {
  // integral type which doesn't represent a pointer
  Integer,
  // floating point
  Float,
  // pointer
  Pointer,
  // can be anything of users choosing [usually result of a constant such as 0]
  Anything,
  // insufficient information
  Unknown
};

/// Convert Basetype to string
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

/// Convert string to BaseType
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