//===- BuiltinAutoDiffOpInterfaceImpl.cpp - Interface external model ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the external model implementation of the automatic
// differentiation type interfaces for the upstream MLIR builtin dialect.
//
//===----------------------------------------------------------------------===//

#include "Implementations/CoreDialectsAutoDiffImplementations.h"
#include "Interfaces/AutoDiffTypeInterface.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;
using namespace mlir::enzyme;

namespace {
class FloatTypeInterface
    : public AutoDiffTypeInterface::ExternalModel<FloatTypeInterface,
                                                  FloatType> {
public:
  Value createNullValue(Type self, OpBuilder &builder, Location loc) const {
    auto fltType = self.cast<FloatType>();
    return builder.create<arith::ConstantFloatOp>(
        loc, APFloat(fltType.getFloatSemantics(), 0), fltType);
  }

  Type getShadowType(Type self, unsigned width) const {
    assert(width == 1 && "unsupported width != 1");
    return self;
  }
};
} // namespace

void mlir::enzyme::registerBuiltinDialectAutoDiffInterface(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, BuiltinDialect *) {
    BFloat16Type::attachInterface<FloatTypeInterface>(*context);
    Float16Type::attachInterface<FloatTypeInterface>(*context);
    Float32Type::attachInterface<FloatTypeInterface>(*context);
    Float64Type::attachInterface<FloatTypeInterface>(*context);
  });
}
