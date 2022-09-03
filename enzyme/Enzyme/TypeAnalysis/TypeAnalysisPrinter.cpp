//===- ActivityAnalysisPrinter.cpp - Printer utility pass for Type Analysis
//----===//
//
//                             Enzyme Project
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// If using this code in an academic setting, please cite the following:
// @incollection{enzymeNeurips,
// title = {Instead of Rewriting Foreign Code for Machine Learning,
//          Automatically Synthesize Fast Gradients},
// author = {Moses, William S. and Churavy, Valentin},
// booktitle = {Advances in Neural Information Processing Systems 33},
// year = {2020},
// note = {To appear in},
// }
//
//===----------------------------------------------------------------------===//
//
// This file contains a utility LLVM pass for printing derived Activity Analysis
// results of a given function.
//
//===----------------------------------------------------------------------===//
#include <llvm/Config/llvm-config.h>

#include "SCEV/ScalarEvolution.h"
#include "SCEV/ScalarEvolutionExpander.h"

#include "llvm/ADT/SmallVector.h"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Metadata.h"

#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Scalar.h"

#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/ScalarEvolution.h"

#include "llvm/Support/CommandLine.h"

#include "../FunctionUtils.h"
#include "../Utils.h"
#include "TypeAnalysis.h"

using namespace llvm;
#ifdef DEBUG_TYPE
#undef DEBUG_TYPE
#endif
#define DEBUG_TYPE "type-analysis-results"

/// Function ActivityAnalysis will be starting its run from
llvm::cl::opt<std::string>
    FunctionToAnalyze("type-analysis-func", cl::init(""), cl::Hidden,
                      cl::desc("Which function to analyze/print"));

namespace {

class TypeAnalysisPrinter final : public FunctionPass {
public:
  static char ID;
  TypeAnalysisPrinter() : FunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetLibraryInfoWrapperPass>();
  }

  bool runOnFunction(Function &F) override {
    if (F.getName() != FunctionToAnalyze)
      return /*changed*/ false;

    FnTypeInfo type_args(&F);
    for (auto &a : type_args.Function->args()) {
      TypeTree dt;
      if (a.getType()->isFPOrFPVectorTy()) {
        dt = ConcreteType(a.getType()->getScalarType());
      } else if (a.getType()->isPointerTy()) {
        auto et = cast<PointerType>(a.getType())->getPointerElementType();
        if (et->isFPOrFPVectorTy()) {
          dt = TypeTree(ConcreteType(et->getScalarType())).Only(-1);
        } else if (et->isPointerTy()) {
          dt = TypeTree(ConcreteType(BaseType::Pointer)).Only(-1);
        }
        dt.insert({}, BaseType::Pointer);
      } else if (a.getType()->isIntOrIntVectorTy()) {
        dt = ConcreteType(BaseType::Integer);
      }
      type_args.Arguments.insert(
          std::pair<Argument *, TypeTree>(&a, dt.Only(-1)));
      // TODO note that here we do NOT propagate constants in type info (and
      // should consider whether we should)
      type_args.KnownValues.insert(
          std::pair<Argument *, std::set<int64_t>>(&a, {}));
    }

    TypeTree dt;
    if (F.getReturnType()->isFPOrFPVectorTy()) {
      dt = ConcreteType(F.getReturnType()->getScalarType());
    } else if (F.getReturnType()->isPointerTy()) {
      auto et = cast<PointerType>(F.getReturnType())->getPointerElementType();
      if (et->isFPOrFPVectorTy()) {
        dt = TypeTree(ConcreteType(et->getScalarType())).Only(-1);
      } else if (et->isPointerTy()) {
        dt = TypeTree(ConcreteType(BaseType::Pointer)).Only(-1);
      }
    } else if (F.getReturnType()->isIntOrIntVectorTy()) {
      dt = ConcreteType(BaseType::Integer);
    }
    type_args.Return = dt.Only(-1);
    PreProcessCache PPC;
    TypeAnalysis TA(PPC.FAM);
    TA.analyzeFunction(type_args);
    for (Function &f : *F.getParent()) {

      for (auto &analysis : TA.analyzedFunctions) {
        if (analysis.first.Function != &f)
          continue;
        auto &ta = *analysis.second;
        llvm::outs() << f.getName() << " - " << analysis.first.Return.str()
                     << " |";

        for (auto &a : f.args()) {
          llvm::outs() << analysis.first.Arguments.find(&a)->second.str() << ":"
                       << to_string(analysis.first.KnownValues.find(&a)->second)
                       << " ";
        }
        llvm::outs() << "\n";

        for (auto &a : f.args()) {
          llvm::outs() << a << ": " << ta.getAnalysis(&a).str() << "\n";
        }
        for (auto &BB : f) {
          llvm::outs() << BB.getName() << "\n";
          for (auto &I : BB) {
            llvm::outs() << I << ": " << ta.getAnalysis(&I).str() << "\n";
          }
        }
      }
    }
    return /*changed*/ false;
  }
};

} // namespace

char TypeAnalysisPrinter::ID = 0;

static RegisterPass<TypeAnalysisPrinter> X("print-type-analysis",
                                           "Print Type Analysis Results");
