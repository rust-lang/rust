//===- ArithAutoDiffOpInterfaceImpl.cpp - Interface external model --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the external model implementation of the automatic
// differentiation op interfaces for the upstream MLIR arithmetic dialect.
//
//===----------------------------------------------------------------------===//

#include "Implementations/CoreDialectsAutoDiffImplementations.h"
#include "Interfaces/AutoDiffOpInterface.h"
#include "Interfaces/GradientUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::enzyme;

namespace {
struct MulFOpInterface
    : public AutoDiffOpInterface::ExternalModel<MulFOpInterface,
                                                arith::MulFOp> {
  LogicalResult createForwardModeAdjoint(Operation *op, OpBuilder &builder,
                                         MGradientUtils *gutils) const {
    // Derivative of r = a * b -> dr = a * db + da * b
    auto mulOp = cast<arith::MulFOp>(op);
    if (!gutils->isConstantValue(mulOp)) {
      mlir::Value res = nullptr;
      for (int i = 0; i < 2; i++) {
        if (!gutils->isConstantValue(mulOp.getOperand(i))) {
          Value tmp = builder.create<arith::MulFOp>(
              mulOp.getLoc(),
              gutils->invertPointerM(mulOp.getOperand(i), builder),
              gutils->getNewFromOriginal(mulOp.getOperand(1 - i)));
          if (res == nullptr)
            res = tmp;
          else
            res = builder.create<arith::AddFOp>(mulOp.getLoc(), res, tmp);
        }
      }
      gutils->setDiffe(mulOp, res, builder);
    }
    gutils->eraseIfUnused(op);
    return success();
  }
};

struct AddFOpInterface
    : public AutoDiffOpInterface::ExternalModel<AddFOpInterface,
                                                arith::AddFOp> {
  LogicalResult createForwardModeAdjoint(Operation *op, OpBuilder &builder,
                                         MGradientUtils *gutils) const {
    // Derivative of r = a + b -> dr = da + db
    auto addOp = cast<arith::AddFOp>(op);
    if (!gutils->isConstantValue(addOp)) {
      mlir::Value res = nullptr;
      for (int i = 0; i < 2; i++) {
        if (!gutils->isConstantValue(addOp.getOperand(i))) {
          Value tmp = gutils->invertPointerM(addOp.getOperand(i), builder);
          if (res == nullptr)
            res = tmp;
          else
            res = builder.create<arith::AddFOp>(addOp.getLoc(), res, tmp);
        }
      }
      gutils->setDiffe(addOp, res, builder);
    }
    gutils->eraseIfUnused(op);
    return success();
  }
};
} // namespace

void mlir::enzyme::registerArithDialectAutoDiffInterface(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, arith::ArithDialect *) {
    arith::AddFOp::attachInterface<AddFOpInterface>(*context);
    arith::MulFOp::attachInterface<MulFOpInterface>(*context);
  });
}
