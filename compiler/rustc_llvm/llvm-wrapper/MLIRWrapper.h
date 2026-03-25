#ifndef INCLUDED_RUSTC_LLVM_MLIRWRAPPER_H
#define INCLUDED_RUSTC_LLVM_MLIRWRAPPER_H

#include "llvm/Support/CBindingWrapping.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

using namespace mlir;

enum class MLIRRustResult { Success, Failure };

typedef struct MLIROpaqueContext *MLIRContextRef;
typedef struct OpaqueOpBuilder *OpBuilderRef;
typedef struct OpaqueModule *ModuleRef;

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(MLIRContext, MLIRContextRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(OpBuilder, OpBuilderRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(mlir::OwningOpRef<ModuleOp>, ModuleRef)

extern "C" MLIRContextRef MLIRRustContextCreate();

extern "C" MLIRRustResult MLIRRustInitTriton(MLIRContextRef ctx);

extern "C" OpBuilderRef MLIRRustModuleBuilderCreate(MLIRContextRef ctx);

extern "C" ModuleRef MLIRRustModuleCreate(OpBuilderRef builder_ref);

#endif // INCLUDED_RUSTC_LLVM_MLIRWRAPPER_H
