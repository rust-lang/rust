//===- MemRefAutoDiffOpInterfaceImpl.cpp - Interface external model -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the external model implementation of the automatic
// differentiation op interfaces for the upstream MLIR memref dialect.
//
//===----------------------------------------------------------------------===//

#include "Implementations/CoreDialectsAutoDiffImplementations.h"
#include "Interfaces/AutoDiffOpInterface.h"
#include "Interfaces/AutoDiffTypeInterface.h"
#include "Interfaces/GradientUtils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::enzyme;

namespace {
struct LoadOpInterface
    : public AutoDiffOpInterface::ExternalModel<LoadOpInterface,
                                                memref::LoadOp> {
  LogicalResult createForwardModeTangent(Operation *op, OpBuilder &builder,
                                         MGradientUtils *gutils) const {
    auto loadOp = cast<memref::LoadOp>(op);
    if (!gutils->isConstantValue(loadOp)) {
      SmallVector<Value> inds;
      for (auto ind : loadOp.getIndices())
        inds.push_back(gutils->getNewFromOriginal(ind));
      mlir::Value res = builder.create<memref::LoadOp>(
          loadOp.getLoc(), gutils->invertPointerM(loadOp.getMemref(), builder),
          inds);
      gutils->setDiffe(loadOp, res, builder);
    }
    gutils->eraseIfUnused(op);
    return success();
  }
};

struct StoreOpInterface
    : public AutoDiffOpInterface::ExternalModel<StoreOpInterface,
                                                memref::StoreOp> {
  LogicalResult createForwardModeTangent(Operation *op, OpBuilder &builder,
                                         MGradientUtils *gutils) const {
    auto storeOp = cast<memref::StoreOp>(op);
    if (!gutils->isConstantValue(storeOp.getMemref())) {
      SmallVector<Value> inds;
      for (auto ind : storeOp.getIndices())
        inds.push_back(gutils->getNewFromOriginal(ind));
      builder.create<memref::StoreOp>(
          storeOp.getLoc(), gutils->invertPointerM(storeOp.getValue(), builder),
          gutils->invertPointerM(storeOp.getMemref(), builder), inds);
    }
    gutils->eraseIfUnused(op);
    return success();
  }
};

struct AllocOpInterface
    : public AutoDiffOpInterface::ExternalModel<AllocOpInterface,
                                                memref::AllocOp> {
  LogicalResult createForwardModeTangent(Operation *op, OpBuilder &builder,
                                         MGradientUtils *gutils) const {
    auto allocOp = cast<memref::AllocOp>(op);
    if (!gutils->isConstantValue(allocOp)) {
      Operation *nop = gutils->cloneWithNewOperands(builder, op);
      gutils->setDiffe(allocOp, nop->getResult(0), builder);
    }
    gutils->eraseIfUnused(op);
    return success();
  }
};

class MemRefTypeInterface
    : public AutoDiffTypeInterface::ExternalModel<MemRefTypeInterface,
                                                  MemRefType> {
public:
  Value createNullValue(Type self, OpBuilder &builder, Location loc) const {
    llvm_unreachable("Cannot create null of memref (todo polygeist null)");
  }

  Type getShadowType(Type self, unsigned width) const {
    assert(width == 1 && "unsupported width != 1");
    return self;
  }
};
} // namespace

void mlir::enzyme::registerMemRefDialectAutoDiffInterface(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, memref::MemRefDialect *) {
    memref::LoadOp::attachInterface<LoadOpInterface>(*context);
    memref::StoreOp::attachInterface<StoreOpInterface>(*context);
    memref::AllocOp::attachInterface<AllocOpInterface>(*context);
    MemRefType::attachInterface<MemRefTypeInterface>(*context);
  });
}
