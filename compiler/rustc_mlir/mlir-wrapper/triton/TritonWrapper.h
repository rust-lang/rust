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

#ifndef TRITON_WRAPPER_H
#define TRITON_WRAPPER_H

#include "mlir/CAPI/Wrap.h"
#include <mlir-c/IR.h>

#include "TritonCompiler.h"

using namespace mlir::triton;

extern "C" {

typedef struct MlirTritonCompiler {
  void *ptr;
} MlirTritonCompiler;

DEFINE_C_API_PTR_METHODS(MlirTritonCompiler, TritonCompiler)

/// Creates a Triton compiler handle for the given MLIR context and target
/// (e.g. "cuda").
MlirTritonCompiler mlirTritonCompilerCreate(MlirContext context,
                                            const char *target,
                                            const char *options);

/// Runs the Triton compilation pipeline on `module`. On success, the module is
/// transformed in-place and the textual IR is stored as the handle output.
bool mlirTritonCompilerCompile(MlirTritonCompiler compiler, MlirModule module);

/// Returns the output string owned by `compiler`.
/// The returned pointer remains valid until the next successful compile on the
/// same handle or until `mlirTritonCompilerFree` is called.
const char *mlirTritonCompilerGetLLIR(MlirTritonCompiler compiler);
const char *mlirTritonCompilerGetTTIR(MlirTritonCompiler compiler);
const char *mlirTritonCompilerGetTTGIR(MlirTritonCompiler compiler);
const char *mlirTritonCompilerGetLLVMIR(MlirTritonCompiler compiler);
const char *mlirTritonCompilerGetASM(MlirTritonCompiler compiler);
const char *mlirTritonCompilerGetBIN(MlirTritonCompiler compiler);

/// Frees the compiler handle.
void mlirTritonCompilerFree(MlirTritonCompiler compiler);
}

#endif /* TRITON_WRAPPER_H */
