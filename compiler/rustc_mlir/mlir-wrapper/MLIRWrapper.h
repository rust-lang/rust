//===-- MLIRWrapper.h - C bindings for MLIR types ---------------*- C++ -*-===//
//
// Provides C-compatible bindings for MLIR types that can be used from Rust
// via FFI. This follows the pattern from LLVM's C API bindings.
//
// These bindings are designed to work alongside melior, providing access to
// MLIR functionality that melior doesn't expose directly.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_WRAPPER_H
#define MLIR_WRAPPER_H

#include "mlir/CAPI/Wrap.h"
#include "mlir/IR/Types.h"
#include <mlir-c/IR.h>

using namespace mlir;

DEFINE_C_API_PTR_METHODS(MlirContext, MLIRContext)

DEFINE_C_API_METHODS(MlirType, Type)

#ifdef __cplusplus
extern "C" {
#endif

void mlirTritonLoadDialects(MlirContext context);

MlirType mlirCreateTritonPointerType(MlirType pointee, int address_space);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // MLIR_WRAPPER_H
