//===- LLVMAutoDiffOpInterfaceImpl.cpp - Interface external model  --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the external model implementation of the automatic
// differentiation op interfaces for the upstream LLVM dialect.
//
//===----------------------------------------------------------------------===//

#include "Implementations/CoreDialectsAutoDiffImplementations.h"
#include "Interfaces/AutoDiffOpInterface.h"
#include "Interfaces/AutoDiffTypeInterface.h"
#include "Interfaces/GradientUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::enzyme;

namespace {
struct LoadOpInterface
    : public AutoDiffOpInterface::ExternalModel<LoadOpInterface, LLVM::LoadOp> {
  LogicalResult createForwardModeTangent(Operation *op, OpBuilder &builder,
                                         MGradientUtils *gutils) const {
    auto loadOp = cast<LLVM::LoadOp>(op);
    if (!gutils->isConstantValue(loadOp)) {
      mlir::Value res = builder.create<LLVM::LoadOp>(
          loadOp.getLoc(), gutils->invertPointerM(loadOp.getAddr(), builder));
      gutils->setDiffe(loadOp, res, builder);
    }
    gutils->eraseIfUnused(op);
    return success();
  }
};

struct StoreOpInterface
    : public AutoDiffOpInterface::ExternalModel<StoreOpInterface,
                                                LLVM::StoreOp> {
  LogicalResult createForwardModeTangent(Operation *op, OpBuilder &builder,
                                         MGradientUtils *gutils) const {
    auto storeOp = cast<LLVM::StoreOp>(op);
    if (!gutils->isConstantValue(storeOp.getAddr())) {
      builder.create<LLVM::StoreOp>(
          storeOp.getLoc(), gutils->invertPointerM(storeOp.getValue(), builder),
          gutils->invertPointerM(storeOp.getAddr(), builder));
    }
    gutils->eraseIfUnused(op);
    return success();
  }
};

struct AllocaOpInterface
    : public AutoDiffOpInterface::ExternalModel<AllocaOpInterface,
                                                LLVM::AllocaOp> {
  LogicalResult createForwardModeTangent(Operation *op, OpBuilder &builder,
                                         MGradientUtils *gutils) const {
    auto allocOp = cast<LLVM::AllocaOp>(op);
    if (!gutils->isConstantValue(allocOp)) {
      Operation *nop = gutils->cloneWithNewOperands(builder, op);
      gutils->setDiffe(allocOp, nop->getResult(0), builder);
    }
    gutils->eraseIfUnused(op);
    return success();
  }
};

class PointerTypeInterface
    : public AutoDiffTypeInterface::ExternalModel<PointerTypeInterface,
                                                  LLVM::LLVMPointerType> {
public:
  Value createNullValue(Type self, OpBuilder &builder, Location loc) const {
    return builder.create<LLVM::NullOp>(loc, self);
  }

  Type getShadowType(Type self, unsigned width) const {
    assert(width == 1 && "unsupported width != 1");
    return self;
  }
};
} // namespace

void mlir::enzyme::registerLLVMDialectAutoDiffInterface(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, LLVM::LLVMDialect *) {
    LLVM::LoadOp::attachInterface<LoadOpInterface>(*context);
    LLVM::StoreOp::attachInterface<StoreOpInterface>(*context);
    LLVM::AllocaOp::attachInterface<AllocaOpInterface>(*context);
    LLVM::LLVMPointerType::attachInterface<PointerTypeInterface>(*context);
  });
}
