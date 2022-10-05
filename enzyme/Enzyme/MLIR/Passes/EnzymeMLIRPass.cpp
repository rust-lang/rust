//===- EnzymeMLIRPass.cpp - Replace calls with their derivatives ------------ //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to lower gpu kernels in NVVM/gpu dialects into
// a generic parallel for representation
//===----------------------------------------------------------------------===//

#include "../../EnzymeLogic.h"
#include "Dialect/Ops.h"
#include "Interfaces/GradientUtils.h"
#include "PassDetails.h"
#include "Passes/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

#define DEBUG_TYPE "enzyme"

using namespace mlir;
using namespace mlir::enzyme;
using namespace enzyme;

namespace {
struct DifferentiatePass : public DifferentiatePassBase<DifferentiatePass> {
  MEnzymeLogic Logic;

  void runOnOperation() override;

  template <typename T>
  void HandleAutoDiff(SymbolTableCollection &symbolTable, T CI) {
    std::vector<DIFFE_TYPE> constants;
    SmallVector<mlir::Value, 2> args;

    size_t truei = 0;
    auto activityAttr = CI.getActivity();

    for (unsigned i = 0; i < CI.getInputs().size(); ++i) {
      mlir::Value res = CI.getInputs()[i];

      auto mop = activityAttr[truei];
      auto iattr = cast<mlir::enzyme::ActivityAttr>(mop);
      DIFFE_TYPE ty = (DIFFE_TYPE)(iattr.getValue());

      constants.push_back(ty);
      args.push_back(res);
      if (ty == DIFFE_TYPE::DUP_ARG || ty == DIFFE_TYPE::DUP_NONEED) {
        ++i;
        res = CI.getInputs()[i];
        args.push_back(res);
      }

      truei++;
    }

    auto *symbolOp = symbolTable.lookupNearestSymbolFrom(CI, CI.getFnAttr());
    auto fn = cast<FunctionOpInterface>(symbolOp);

    DIFFE_TYPE retType =
        fn.getNumResults() == 0 ? DIFFE_TYPE::CONSTANT : DIFFE_TYPE::DUP_ARG;

    MTypeAnalysis TA;
    auto type_args = TA.getAnalyzedTypeInfo(fn);
    auto mode = DerivativeMode::ForwardMode;
    bool freeMemory = true;
    size_t width = 1;

    std::vector<bool> volatile_args;
    for (auto &a : fn.getBody().getArguments()) {
      volatile_args.push_back(!(mode == DerivativeMode::ReverseModeCombined));
    }

    FunctionOpInterface newFunc = Logic.CreateForwardDiff(
        fn, retType, constants, TA,
        /*should return*/ false, mode, freeMemory, width,
        /*addedType*/ nullptr, type_args, volatile_args,
        /*augmented*/ nullptr);

    OpBuilder builder(CI);
    auto dCI = builder.create<func::CallOp>(CI.getLoc(), newFunc.getName(),
                                            newFunc.getResultTypes(), args);
    CI.replaceAllUsesWith(dCI);
    CI->erase();
  }

  void lowerEnzymeCalls(SymbolTableCollection &symbolTable,
                        FunctionOpInterface op) {
    SmallVector<Operation *> toLower;
    op->walk([&](enzyme::ForwardDiffOp dop) {
      auto *symbolOp =
          symbolTable.lookupNearestSymbolFrom(dop, dop.getFnAttr());
      auto callableOp = cast<FunctionOpInterface>(symbolOp);

      lowerEnzymeCalls(symbolTable, callableOp);
      toLower.push_back(dop);
    });

    for (auto T : toLower) {
      if (auto F = dyn_cast<enzyme::ForwardDiffOp>(T)) {
        HandleAutoDiff(symbolTable, F);
      } else {
        llvm_unreachable("Illegal type");
      }
    }
  }
};

} // end anonymous namespace

namespace mlir {
namespace enzyme {
std::unique_ptr<Pass> createDifferentiatePass() {
  new DifferentiatePass();
  return std::make_unique<DifferentiatePass>();
}
} // namespace enzyme
} // namespace mlir

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

void DifferentiatePass::runOnOperation() {
  SymbolTableCollection symbolTable;
  symbolTable.getSymbolTable(getOperation());
  ConversionPatternRewriter B(getOperation()->getContext());
  getOperation()->walk(
      [&](FunctionOpInterface op) { lowerEnzymeCalls(symbolTable, op); });
}
