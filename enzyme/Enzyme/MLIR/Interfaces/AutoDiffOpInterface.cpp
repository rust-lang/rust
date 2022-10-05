//===- AutoDiffOpInterface.cpp - Op interface for auto differentiation ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces necessary to implement scalable automatic
// differentiation across an unbounded number of MLIR IR constructs.
//
//===----------------------------------------------------------------------===//

#include "AutoDiffOpInterface.h"
#include "GradientUtils.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"

using namespace mlir;
using namespace mlir::enzyme;

#include "MLIR/Interfaces/AutoDiffOpInterface.cpp.inc"
