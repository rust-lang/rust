//===- Enzyme.cpp - Automatic Differentiation Transformation Pass  -------===//
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
// This file contains Enzyme, a transformation pass that takes replaces calls
// to function calls to *__enzyme_autodiff* with a call to the derivative of
// the function passed as the first argument.
//
//===----------------------------------------------------------------------===//
#include "SCEV/ScalarEvolutionExpander.h"

#include <llvm/Config/llvm-config.h>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
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
#include "llvm/Analysis/TargetLibraryInfo.h"

#include "ActivityAnalysis.h"
#include "EnzymeLogic.h"
#include "GradientUtils.h"
#include "Utils.h"

using namespace llvm;
#ifdef DEBUG_TYPE
#undef DEBUG_TYPE
#endif
#define DEBUG_TYPE "lower-enzyme-intrinsic"

llvm::cl::opt<bool>
    EnzymePostOpt("enzmye-postopt", cl::init(false), cl::Hidden,
                  cl::desc("Run enzymepostprocessing optimizations"));

/// Return whether successful
bool HandleAutoDiff(CallInst *CI, TargetLibraryInfo &TLI, AAResults &AA,
                    bool PostOpt) {

  Value *fn = CI->getArgOperand(0);

  while (auto ci = dyn_cast<CastInst>(fn)) {
    fn = ci->getOperand(0);
  }
  while (auto ci = dyn_cast<BlockAddress>(fn)) {
    fn = ci->getFunction();
  }
  while (auto ci = dyn_cast<ConstantExpr>(fn)) {
    fn = ci->getOperand(0);
  }
  if (!isa<Function>(fn)) {
    EmitFailure("NoFunctionToDifferentiate", CI->getDebugLoc(), CI,
                "failed to find fn to differentiate", *CI, " - found - ", *fn);
    return false;
  }
  auto FT = cast<Function>(fn)->getFunctionType();
  assert(fn);

  if (EnzymePrint)
    llvm::errs() << "prefn:\n" << *fn << "\n";

  std::vector<DIFFE_TYPE> constants;
  SmallVector<Value *, 2> args;

  unsigned truei = 0;
  IRBuilder<> Builder(CI);

  bool AtomicAdd =
      llvm::Triple(CI->getParent()->getParent()->getParent()->getTargetTriple())
              .getArch() == Triple::nvptx ||
      llvm::Triple(CI->getParent()->getParent()->getParent()->getTargetTriple())
              .getArch() == Triple::nvptx64;

  for (unsigned i = 1; i < CI->getNumArgOperands(); ++i) {
    Value *res = CI->getArgOperand(i);

    assert(truei < FT->getNumParams());
    auto PTy = FT->getParamType(truei);
    DIFFE_TYPE ty = DIFFE_TYPE::CONSTANT;

    if (auto av = dyn_cast<MetadataAsValue>(res)) {
      auto MS = cast<MDString>(av->getMetadata())->getString();
      if (MS == "enzyme_dup") {
        ty = DIFFE_TYPE::DUP_ARG;
      } else if (MS == "enzyme_dupnoneed") {
        ty = DIFFE_TYPE::DUP_NONEED;
      } else if (MS == "enzyme_out") {
        ty = DIFFE_TYPE::OUT_DIFF;
      } else if (MS == "enzyme_const") {
        ty = DIFFE_TYPE::CONSTANT;
      } else {
        EmitFailure("IllegalDiffeType", CI->getDebugLoc(), CI,
                    "illegal enzyme metadata classification ", *CI);
        return false;
      }
      ++i;
      res = CI->getArgOperand(i);
    } else if (isa<LoadInst>(res) &&
               isa<GlobalVariable>(cast<LoadInst>(res)->getOperand(0))) {
      auto gv = cast<GlobalVariable>(cast<LoadInst>(res)->getOperand(0));
      auto MS = gv->getName();
      if (MS == "enzyme_dup") {
        ty = DIFFE_TYPE::DUP_ARG;
        ++i;
        res = CI->getArgOperand(i);
      } else if (MS == "enzyme_dupnoneed") {
        ty = DIFFE_TYPE::DUP_NONEED;
        ++i;
        res = CI->getArgOperand(i);
      } else if (MS == "enzyme_out") {
        ty = DIFFE_TYPE::OUT_DIFF;
        ++i;
        res = CI->getArgOperand(i);
      } else if (MS == "enzyme_const") {
        ty = DIFFE_TYPE::CONSTANT;
        ++i;
        res = CI->getArgOperand(i);
      } else {
        ty = whatType(PTy);
      }
    } else if (isa<LoadInst>(res) &&
               isa<ConstantExpr>(cast<LoadInst>(res)->getOperand(0)) &&
               cast<ConstantExpr>(cast<LoadInst>(res)->getOperand(0))
                   ->isCast() &&
               isa<GlobalVariable>(
                   cast<ConstantExpr>(cast<LoadInst>(res)->getOperand(0))
                       ->getOperand(0))) {
      auto gv = cast<GlobalVariable>(
          cast<ConstantExpr>(cast<LoadInst>(res)->getOperand(0))
              ->getOperand(0));
      auto MS = gv->getName();
      if (MS == "enzyme_dup") {
        ty = DIFFE_TYPE::DUP_ARG;
        ++i;
        res = CI->getArgOperand(i);
      } else if (MS == "enzyme_dupnoneed") {
        ty = DIFFE_TYPE::DUP_NONEED;
        ++i;
        res = CI->getArgOperand(i);
      } else if (MS == "enzyme_out") {
        llvm::errs() << "saw metadata for diffe_out\n";
        ty = DIFFE_TYPE::OUT_DIFF;
        ++i;
        res = CI->getArgOperand(i);
      } else if (MS == "enzyme_const") {
        ty = DIFFE_TYPE::CONSTANT;
        ++i;
        res = CI->getArgOperand(i);
      } else {
        ty = whatType(PTy);
      }
    } else
      ty = whatType(PTy);

    constants.push_back(ty);

    assert(truei < FT->getNumParams());
    if (PTy != res->getType()) {
      if (auto ptr = dyn_cast<PointerType>(res->getType())) {
        if (auto PT = dyn_cast<PointerType>(PTy)) {
          if (ptr->getAddressSpace() != PT->getAddressSpace()) {
            res = Builder.CreateAddrSpaceCast(
                res,
                PointerType::get(ptr->getElementType(), PT->getAddressSpace()));
            assert(res);
            assert(PTy);
            assert(FT);
            llvm::errs() << "Warning cast(1) __enzyme_autodiff argument " << i
                         << " " << *res << "|" << *res->getType()
                         << " to argument " << truei << " " << *PTy << "\n"
                         << "orig: " << *FT << "\n";
          }
        }
      }
      if (!res->getType()->canLosslesslyBitCastTo(PTy)) {
        auto loc = CI->getDebugLoc();
        if (auto arg = dyn_cast<Instruction>(res)) {
          loc = arg->getDebugLoc();
        }
        EmitFailure("IllegalArgCast", loc, CI,
                    "Cannot cast __enzyme_autodiff primal argument ", i,
                    ", found ", *res, ", type ", *res->getType(), " - to arg ",
                    truei, " ", *PTy);
        return false;
      }
      res = Builder.CreateBitCast(res, PTy);
    }

    args.push_back(res);
    if (ty == DIFFE_TYPE::DUP_ARG || ty == DIFFE_TYPE::DUP_NONEED) {
      ++i;

      Value *res = CI->getArgOperand(i);
      if (PTy != res->getType()) {
        if (auto ptr = dyn_cast<PointerType>(res->getType())) {
          if (auto PT = dyn_cast<PointerType>(PTy)) {
            if (ptr->getAddressSpace() != PT->getAddressSpace()) {
              res = Builder.CreateAddrSpaceCast(
                  res, PointerType::get(ptr->getElementType(),
                                        PT->getAddressSpace()));
              assert(res);
              assert(PTy);
              assert(FT);
              llvm::errs() << "Warning cast(2) __enzyme_autodiff argument " << i
                           << " " << *res << "|" << *res->getType()
                           << " to argument " << truei << " " << *PTy << "\n"
                           << "orig: " << *FT << "\n";
            }
          }
        }
        if (!res->getType()->canLosslesslyBitCastTo(PTy)) {
          assert(res);
          assert(res->getType());
          assert(PTy);
          assert(FT);
          auto loc = CI->getDebugLoc();
          if (auto arg = dyn_cast<Instruction>(res)) {
            loc = arg->getDebugLoc();
          }
          EmitFailure("IllegalArgCast", loc, CI,
                      "Cannot cast __enzyme_autodiff shadow argument", i,
                      ", found ", *res, ", type ", *res->getType(),
                      " - to arg ", truei, " ", *PTy);
          return false;
        }
        res = Builder.CreateBitCast(res, PTy);
      }
      args.push_back(res);
    }

    ++truei;
  }

  bool differentialReturn =
      cast<Function>(fn)->getReturnType()->isFPOrFPVectorTy();

  DIFFE_TYPE retType = whatType(cast<Function>(fn)->getReturnType());

  std::map<Argument *, bool> volatile_args;
  FnTypeInfo type_args(cast<Function>(fn));
  for (auto &a : type_args.Function->args()) {
    volatile_args[&a] = false;
    TypeTree dt;
    if (a.getType()->isFPOrFPVectorTy()) {
      dt = ConcreteType(a.getType()->getScalarType());
    } else if (a.getType()->isPointerTy()) {
      auto et = cast<PointerType>(a.getType())->getElementType();
      if (et->isFPOrFPVectorTy()) {
        dt = TypeTree(ConcreteType(et->getScalarType())).Only(-1);
      } else if (et->isPointerTy()) {
        dt = TypeTree(ConcreteType(BaseType::Pointer)).Only(-1);
      }
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

  TypeAnalysis TA(TLI);
  type_args = TA.analyzeFunction(type_args).getAnalyzedTypeInfo();

  auto newFunc = CreatePrimalAndGradient(
      cast<Function>(fn), retType, constants, TLI, TA, AA,
      /*should return*/ false, /*dretPtr*/ false, /*topLevel*/ true,
      /*addedType*/ nullptr, type_args, volatile_args,
      /*index mapping*/ nullptr, AtomicAdd, PostOpt);

  if (!newFunc)
    return false;

  if (differentialReturn)
    args.push_back(ConstantFP::get(cast<Function>(fn)->getReturnType(), 1.0));
  assert(newFunc);

  if (EnzymePrint)
    llvm::errs() << "postfn:\n" << *newFunc << "\n";
  Builder.setFastMathFlags(getFast());

  CallInst *diffret = cast<CallInst>(Builder.CreateCall(newFunc, args));
  diffret->setCallingConv(CI->getCallingConv());
  diffret->setDebugLoc(CI->getDebugLoc());
  if (!diffret->getType()->isEmptyTy() && !diffret->getType()->isVoidTy()) {
    unsigned idxs[] = {0};
    auto diffreti = Builder.CreateExtractValue(diffret, idxs);
    CI->replaceAllUsesWith(diffreti);
  } else {
    CI->replaceAllUsesWith(UndefValue::get(CI->getType()));
  }
  CI->eraseFromParent();
  return true;
}

static bool lowerEnzymeCalls(Function &F, TargetLibraryInfo &TLI, AAResults &AA,
                             bool PostOpt, bool &successful) {

  bool Changed = false;

reset:
  for (BasicBlock &BB : F) {

    for (auto BI = BB.rbegin(), BE = BB.rend(); BI != BE; ++BI) {
      CallInst *CI = dyn_cast<CallInst>(&*BI);
      if (!CI)
        continue;

      Function *Fn = CI->getCalledFunction();

#if LLVM_VERSION_MAJOR >= 11
      if (auto castinst = dyn_cast<ConstantExpr>(CI->getCalledOperand())) {
#else
      if (auto castinst = dyn_cast<ConstantExpr>(CI->getCalledValue())) {
#endif
        if (castinst->isCast())
          if (auto fn = dyn_cast<Function>(castinst->getOperand(0)))
            Fn = fn;
      }

      if (Fn && (Fn->getName() == "__enzyme_autodiff" ||
                 Fn->getName().startswith("__enzyme_autodiff") ||
                 Fn->getName().contains("__enzyme_autodiff"))) {
        successful &= HandleAutoDiff(CI, TLI, AA, PostOpt);
        Changed = true;
        if (successful)
          goto reset;
      }
    }
  }

  return Changed;
}

namespace {

class Enzyme : public ModulePass {
public:
  bool PostOpt;
  static char ID;
  Enzyme(bool PostOpt = false) : ModulePass(ID), PostOpt(PostOpt) {
    PostOpt |= EnzymePostOpt;
    // initializeLowerAutodiffIntrinsicPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetLibraryInfoWrapperPass>();
    AU.addRequired<AAResultsWrapperPass>();
    AU.addRequired<GlobalsAAWrapperPass>();
    AU.addRequired<BasicAAWrapperPass>();

    // AU.addRequiredID(LCSSAID);

    // LoopInfo is required to ensure that all loops have preheaders
    // AU.addRequired<LoopInfoWrapperPass>();

    // AU.addRequiredID(llvm::LoopSimplifyID);//<LoopSimplifyWrapperPass>();
  }

  bool runOnModule(Module &M) override {
    // auto &AA = getAnalysis<AAResultsWrapperPass>().getAAResults();
    auto &G_AA = getAnalysis<GlobalsAAWrapperPass>().getResult();

    // llvm::errs() << "G_AA: " << &G_AA << "\n";
    // AAResults AA(TLI);
    // AA.addAAResult(B_AA);
    // AA.addAAResult(G_AA);

    /*
    auto &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    auto &SE = getAnalysis<ScalarEvolutionWrapperPass>().getSE();
    auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    */
    bool changed = false;
    for (Function &F : M) {
      if (F.empty())
        continue;

#if LLVM_VERSION_MAJOR >= 10
      auto &TLI = getAnalysis<TargetLibraryInfoWrapperPass>().getTLI(F);
#else
      auto &TLI = getAnalysis<TargetLibraryInfoWrapperPass>().getTLI();
#endif

      AAResults AA(TLI);
      // auto &B_AA = getAnalysis<BasicAAWrapperPass>().getResult();
      // AA.addAAResult(B_AA);
      AA.addAAResult(G_AA);

      // auto &AA = getAnalysis<AAResultsWrapperPass>().getAAResults();
      // auto &LI = getAnalysis<LoopInfoWrapperPass>(F).getLoopInfo();

      bool successful = true;
      changed |= lowerEnzymeCalls(F, TLI, AA, PostOpt, successful);

      if (!successful) {
        M.getContext().diagnose(
            (EnzymeFailure("FailedToDifferentiate", F.getSubprogram(),
                           &*F.getEntryBlock().begin())
             << "EnzymeFailure when replacing __enzyme_autodiff calls in "
             << F.getName()));
      }
    }

    std::vector<CallInst *> toErase;
    for (Function &F : M) {
      if (F.empty())
        continue;

      for (BasicBlock &BB : F) {
        for (Instruction &I : BB) {
          if (auto CI = dyn_cast<CallInst>(&I)) {
            if (auto called = CI->getCalledFunction()) {
              if (called->getName() == "__enzyme_float" ||
                  called->getName() == "__enzyme_double" ||
                  called->getName() == "__enzyme_integer" ||
                  called->getName() == "__enzyme_pointer") {
                toErase.push_back(CI);
              }
            }
          }
        }
      }
    }
    for (auto I : toErase) {
      I->eraseFromParent();
    }
    return changed;
  }
};

} // namespace

char Enzyme::ID = 0;

static RegisterPass<Enzyme> X("enzyme", "Enzyme Pass");

ModulePass *createEnzymePass(bool PostOpt) { return new Enzyme(PostOpt); }