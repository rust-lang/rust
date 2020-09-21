/*
 * BaseType.h - Underlying enum denoting type of a value
 *
 * Copyright (C) 2020 William S. Moses (enzyme@wsmoses.com) - All Rights
 * Reserved
 *
 * For commercial use of this code please contact the author(s) above.
 */
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