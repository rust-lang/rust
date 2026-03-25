//===-- MLIRWrapper.cpp - C bindings for MLIR types -----------------------===//
//
// Implementation of C-compatible bindings for MLIR types.
//
//===----------------------------------------------------------------------===//

#include "MLIRWrapper.h"

#include "mlir/IR/DialectRegistry.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir;

extern "C" void mlirLoadTritonDialect(MlirContext ctx) {
  MLIRContext *context = unwrap(ctx);
  DialectRegistry registry;

  registry.insert<triton::TritonDialect>();
  context->appendDialectRegistry(registry);

  context->loadDialect<triton::TritonDialect>();
}

extern "C" MlirType mlirCreateTritonPointerType(MlirType pointee,
                                                int address_space) {
  auto type = unwrap(pointee);

  auto pointer_type = triton::getPointerType(type, address_space);

  return wrap(pointer_type);
}
