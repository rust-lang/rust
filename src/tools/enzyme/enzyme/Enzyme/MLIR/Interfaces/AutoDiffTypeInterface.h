//===- AutoDiffTypeInterface.h - Type interface for auto diff ----* C++ -*-===//
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

#ifndef ENZYME_MLIR_INTERFACES_AUTODIFFTYPEINTERFACE_H
#define ENZYME_MLIR_INTERFACES_AUTODIFFTYPEINTERFACE_H

#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

namespace mlir {
class OpBuilder;
}

#include "MLIR/Interfaces/AutoDiffTypeInterface.h.inc"

#endif // ENZYME_MLIR_INTERFACES_AUTODIFFTYPEINTERFACE_H
