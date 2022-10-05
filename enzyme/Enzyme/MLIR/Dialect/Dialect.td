//===- Enzyme.td - Enzyme dialect --------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_DIALECT
#define ENZYME_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Enzyme dialect definition.
//===----------------------------------------------------------------------===//

def Enzyme_Dialect : Dialect {
  let name = "enzyme";
  let description = [{}];
  let cppNamespace = "::mlir::enzyme";
  let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Base Enzyme operation definition.
//===----------------------------------------------------------------------===//

class Enzyme_Op<string mnemonic, list<Trait> traits = []>
    : Op<Enzyme_Dialect, mnemonic, traits>;

#endif // ENZYME_DIALECT
