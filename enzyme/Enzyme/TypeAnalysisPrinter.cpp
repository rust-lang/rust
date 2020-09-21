/*
 * TypeAnalysisPrinter.cpp - Lower Enzyme calls
 *
 * Copyright (C) 2020 William S. Moses (enzyme@wsmoses.com) - All Rights
 * Reserved
 *
 * For commercial use of this code please contact the author(s) above.
 */

#include <llvm/Config/llvm-config.h>

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

#include "TypeAnalysis.h"
#include "Utils.h"

using namespace llvm;
#ifdef DEBUG_TYPE
#undef DEBUG_TYPE
#endif
#define DEBUG_TYPE "type-analysis-results"

llvm::cl::opt<std::string> ta_fn("type-analysis-func", cl::init(""), cl::Hidden,
                                 cl::desc("Which function to analyze/print"));

namespace {

class TypeAnalysisPrinter : public FunctionPass {
public:
  static char ID;
  TypeAnalysisPrinter() : FunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetLibraryInfoWrapperPass>();
    AU.addRequired<AAResultsWrapperPass>();
    AU.addRequired<GlobalsAAWrapperPass>();
    AU.addRequired<BasicAAWrapperPass>();
  }

  bool runOnFunction(Function &F) override {
    if (F.getName() != ta_fn)
      return /*changed*/ false;
#if LLVM_VERSION_MAJOR >= 10
    auto &TLI = getAnalysis<TargetLibraryInfoWrapperPass>().getTLI(F);
#else
    auto &TLI = getAnalysis<TargetLibraryInfoWrapperPass>().getTLI();
#endif

    // auto &AA = getAnalysis<AAResultsWrapperPass>().getAAResults();
    auto &G_AA = getAnalysis<GlobalsAAWrapperPass>().getResult();

    NewFnTypeInfo type_args(&F);
    for (auto &a : type_args.function->args()) {
      ValueData dt;
      if (a.getType()->isFPOrFPVectorTy()) {
        dt = DataType(a.getType()->getScalarType());
      } else if (a.getType()->isPointerTy()) {
        auto et = cast<PointerType>(a.getType())->getElementType();
        if (et->isFPOrFPVectorTy()) {
          dt = ValueData(DataType(et->getScalarType())).Only({-1});
        } else if (et->isPointerTy()) {
          dt = ValueData(DataType(IntType::Pointer)).Only({-1});
        }
      }
      type_args.first.insert(std::pair<Argument *, ValueData>(&a, dt.Only(-1)));
      // TODO note that here we do NOT propagate constants in type info (and
      // should consider whether we should)
      type_args.knownValues.insert(
          std::pair<Argument *, std::set<int64_t>>(&a, {}));
    }

    TypeAnalysis TA;
    TA.analyzeFunction(type_args);
    for (Function &f : *F.getParent()) {

      for (auto &analysis : TA.analyzedFunctions) {
        if (analysis.first.function != &f)
          continue;
        auto &ta = analysis.second;
        llvm::outs() << f.getName() << " - " << analysis.first.second.str()
                     << " |";

        for (auto &a : f.args()) {
          llvm::outs() << analysis.first.first.find(&a)->second.str() << ":"
                       << to_string(analysis.first.knownValues.find(&a)->second)
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
