//===- AutoDiffTypeInterface.cpp - Type interface for auto diff -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the type interfaces necessary to implement scalable
// automatic differentiation across an unbounded number of MLIR IR constructs.
//
//===----------------------------------------------------------------------===//

#include "AutoDiffTypeInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"

using namespace mlir;

#include "MLIR/Interfaces/AutoDiffTypeInterface.cpp.inc"
