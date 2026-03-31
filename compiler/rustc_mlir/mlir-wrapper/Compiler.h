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

#ifndef COMPILER_H
#define COMPILER_H

#include <stdint.h>
#include <string>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"

// CudaCompileOptions and its FFI-safe helper types (OptionalI32, OptionalBool,
// Dim3) are defined in CudaBackend.h and pulled in here so that CompileOptions
// can reference CudaCompileOptions in its union.
#include "triton/backend/CudaBackend.h"

namespace mlir {
namespace triton {

// ---------------------------------------------------------------------------
// Target backend discriminator
// The fixed underlying type (uint32_t) guarantees a stable 4-byte width on
// every platform, matching a Rust #[repr(u32)] enum.
// ---------------------------------------------------------------------------

enum TargetBackend : uint32_t {
  TargetBackend_Cuda = 0,
  TargetBackend_Rocm = 1,
  TargetBackend_Spirv = 2,
};

// ---------------------------------------------------------------------------
// Tagged union of all backend option structs
//
// Equivalent to a Rust #[repr(C)] enum with associated data:
//
//   #[repr(C)]
//   pub struct CompileOptions {
//       pub backend: TargetBackend,
//       pub data: CompileOptionsData,   // #[repr(C)] union
//   }
// ---------------------------------------------------------------------------

/// Holds the active backend's options; only the member selected by
/// CompileOptions::backend may be accessed.
union CompileOptionsData {
  CudaCompileOptions cuda;
  // RocmCompileOptions  rocm;   // reserved for future use
  // SpirvCompileOptions spirv;  // reserved for future use
};

/// Complete compile options passed across the C/Rust FFI boundary.
struct CompileOptions {
  TargetBackend backend; ///< Selects the active union member.
  CompileOptionsData data;
};

// ---------------------------------------------------------------------------
// Abstract compiler interface
// ---------------------------------------------------------------------------

class Compiler {
public:
  Compiler(MLIRContext *context, std::string target, CompileOptions options)
      : context(context), target(target), options(options) {}

  virtual ~Compiler() = default;

  virtual LogicalResult compile(ModuleOp mlir_module) = 0;

protected:
  std::string target;
  CompileOptions options;
  MLIRContext *context;
};

} // namespace triton
} // namespace mlir

#endif /* COMPILER_H */
