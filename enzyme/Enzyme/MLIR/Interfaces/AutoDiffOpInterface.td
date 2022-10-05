//===-- AutoDiffOpInterface.td - Op Interface for auto diff-*- tablegen -*-===//
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

#ifndef ENZYME_MLIR_INTERFACES_AUTODIFFOPINTERFACES
#define ENZYME_MLIR_INTERFACES_AUTODIFFOPINTERFACES

include "mlir/IR/OpBase.td"

def AutoDiffOpInterface : OpInterface<"AutoDiffOpInterface"> {
  let description = [{
    A differetiable MLIR operation that is able to emit forward and reverse
    mode adjoints for itself.
  }];
  let cppNamespace = "::mlir::enzyme";

  let methods = [
    InterfaceMethod<
    /*desc=*/[{
      Emits a forward-mode adjoint of the given function. All IR manipulation
      must go through the supplied arguments: `builder` is preset to insert new
      IR in the correct location and should be used to construct any new IR;
      `gutils` provides the mapping between main and derivative computation for
      values and operations and should be updated to keep track of newly created
      IR entities as well as of deleted entities. Returns success if generation
      was successful.
    }],
    /*retTy=*/"::mlir::LogicalResult",
    /*methodName=*/"createForwardModeAdjoint",
    /*args=*/(ins "::mlir::OpBuilder &":$builder, "::mlir::enzyme::MGradientUtils *":$gutils)
    >
  ];
}

#endif // ENZYME_MLIR_INTERFACES_AUTODIFFOPINTERFACES
