//===- CoreDialectsAutoDiffImplementation.h - Impl registrations -* C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains context registration facilities for external model
// implementations of the automatic differentiation interface for upstream MLIR
// dialects.
//
//===----------------------------------------------------------------------===//

namespace mlir {
class DialectRegistry;

namespace enzyme {
void registerArithDialectAutoDiffInterface(DialectRegistry &registry);
void registerBuiltinDialectAutoDiffInterface(DialectRegistry &registry);
void registerSCFDialectAutoDiffInterface(DialectRegistry &registry);
} // namespace enzyme
} // namespace mlir
