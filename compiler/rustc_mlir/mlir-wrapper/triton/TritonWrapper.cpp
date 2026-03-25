/*
 * Copyright (c) 2026 Teenygrad.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <string>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include "../MLIRWrapper.h"
#include "TritonCompiler.h"
#include "TritonWrapper.h"

using namespace mlir;
using namespace mlir::triton;

extern "C" ::MlirTritonCompiler mlirTritonCompilerCreate(MlirContext context,
                                                         const char *target,
                                                         const char *options) {
  if (!context.ptr || !target) {
    return ::MlirTritonCompiler{nullptr};
  }

  auto *ctx = unwrap(context);
  auto *handle = new TritonCompiler(ctx, target, options);
  return ::MlirTritonCompiler{handle};
}

extern "C" bool mlirTritonCompilerCompile(::MlirTritonCompiler compiler,
                                          MlirModule module) {
  auto *handle = unwrap(compiler);
  const auto *op = reinterpret_cast<const Operation *>(module.ptr);
  ModuleOp moduleOp = llvm::cast<ModuleOp>(const_cast<Operation *>(op));
  return succeeded(handle->compile(moduleOp));
}

extern "C" const char *
mlirTritonCompilerGetLLIR(::MlirTritonCompiler compiler) {
  auto *handle = unwrap(compiler);
  return handle->getLLIR();
}

extern "C" const char *
mlirTritonCompilerGetTTIR(::MlirTritonCompiler compiler) {
  auto *handle = unwrap(compiler);
  return handle->getTTIR();
}

extern "C" const char *
mlirTritonCompilerGetTTGIR(::MlirTritonCompiler compiler) {
  auto *handle = unwrap(compiler);
  return handle->getTTGIR();
}

extern "C" const char *
mlirTritonCompilerGetLLVMIR(::MlirTritonCompiler compiler) {
  auto *handle = unwrap(compiler);
  return handle->getLLVMIR();
}

extern "C" const char *mlirTritonCompilerGetASM(::MlirTritonCompiler compiler) {
  auto *handle = unwrap(compiler);
  return handle->getASM();
}

extern "C" const char *mlirTritonCompilerGetBIN(::MlirTritonCompiler compiler) {
  auto *handle = unwrap(compiler);
  return handle->getBIN();
}

extern "C" void mlirTritonCompilerFree(::MlirTritonCompiler compiler) {
  auto *handle = unwrap(compiler);
  delete handle;
}
