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
#include "SCEV/ScalarEvolution.h"
#include "SCEV/ScalarEvolutionExpander.h"

#include <llvm/Config/llvm-config.h>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SetVector.h"
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

#include "llvm/Transforms/Utils/Cloning.h"
#if LLVM_VERSION_MAJOR >= 11
#include "llvm/Analysis/InlineAdvisor.h"
#endif
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/InlineCost.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetLibraryInfo.h"

#include "ActivityAnalysis.h"
#include "EnzymeLogic.h"
#include "GradientUtils.h"
#include "Utils.h"

#include "CApi.h"
using namespace llvm;
#ifdef DEBUG_TYPE
#undef DEBUG_TYPE
#endif
#define DEBUG_TYPE "lower-enzyme-intrinsic"

llvm::cl::opt<bool>
    EnzymePostOpt("enzmye-postopt", cl::init(false), cl::Hidden,
                  cl::desc("Run enzymepostprocessing optimizations"));

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

  /// Return whether successful
  template <typename T>
  bool HandleAutoDiff(T *CI, TargetLibraryInfo &TLI, AAResults &AA,
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
                  "failed to find fn to differentiate", *CI, " - found - ",
                  *fn);
      return false;
    }
    if (cast<Function>(fn)->empty()) {
      EmitFailure("EmptyFunctionToDifferentiate", CI->getDebugLoc(), CI,
                  "failed to find fn to differentiate", *CI, " - found - ",
                  *fn);
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
        llvm::Triple(
            CI->getParent()->getParent()->getParent()->getTargetTriple())
                .getArch() == Triple::nvptx ||
        llvm::Triple(
            CI->getParent()->getParent()->getParent()->getTargetTriple())
                .getArch() == Triple::nvptx64;

    for (unsigned i = 1; i < CI->getNumArgOperands(); ++i) {
      Value *res = CI->getArgOperand(i);

      if (truei >= FT->getNumParams()) {
        EmitFailure("TooManyArgs", CI->getDebugLoc(), CI,
                    "Had too many arguments to __enzyme_autodiff", *CI,
                    " - extra arg - ", *res);
        return false;
      }
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
      } else if ((isa<LoadInst>(res) || isa<CastInst>(res)) &&
                 isa<GlobalVariable>(cast<Instruction>(res)->getOperand(0))) {
        GlobalVariable *gv =
            cast<GlobalVariable>(cast<Instruction>(res)->getOperand(0));
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
      } else if (isa<GlobalVariable>(res)) {
        auto gv = cast<GlobalVariable>(res);
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
      } else if (isa<ConstantExpr>(res) && cast<ConstantExpr>(res)->isCast() &&
                 isa<GlobalVariable>(cast<ConstantExpr>(res)->getOperand(0))) {
        auto gv = cast<GlobalVariable>(cast<ConstantExpr>(res)->getOperand(0));
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
      } else if (isa<CastInst>(res) && cast<CastInst>(res) &&
                 isa<AllocaInst>(cast<CastInst>(res)->getOperand(0))) {
        auto gv = cast<AllocaInst>(cast<CastInst>(res)->getOperand(0));
        auto MS = gv->getName();
        if (MS.startswith("enzyme_dup")) {
          ty = DIFFE_TYPE::DUP_ARG;
          ++i;
          res = CI->getArgOperand(i);
        } else if (MS.startswith("enzyme_dupnoneed")) {
          ty = DIFFE_TYPE::DUP_NONEED;
          ++i;
          res = CI->getArgOperand(i);
        } else if (MS.startswith("enzyme_out")) {
          ty = DIFFE_TYPE::OUT_DIFF;
          ++i;
          res = CI->getArgOperand(i);
        } else if (MS.startswith("enzyme_const")) {
          ty = DIFFE_TYPE::CONSTANT;
          ++i;
          res = CI->getArgOperand(i);
        } else {
          ty = whatType(PTy);
        }
      } else if (isa<AllocaInst>(res)) {
        auto gv = cast<AllocaInst>(res);
        auto MS = gv->getName();
        if (MS.startswith("enzyme_dup")) {
          ty = DIFFE_TYPE::DUP_ARG;
          ++i;
          res = CI->getArgOperand(i);
        } else if (MS.startswith("enzyme_dupnoneed")) {
          ty = DIFFE_TYPE::DUP_NONEED;
          ++i;
          res = CI->getArgOperand(i);
        } else if (MS.startswith("enzyme_out")) {
          ty = DIFFE_TYPE::OUT_DIFF;
          ++i;
          res = CI->getArgOperand(i);
        } else if (MS.startswith("enzyme_const")) {
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
                  res, PointerType::get(ptr->getElementType(),
                                        PT->getAddressSpace()));
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
                      ", found ", *res, ", type ", *res->getType(),
                      " - to arg ", truei, " ", *PTy);
          return false;
        }
        res = Builder.CreateBitCast(res, PTy);
      }

      args.push_back(res);
      if (ty == DIFFE_TYPE::DUP_ARG || ty == DIFFE_TYPE::DUP_NONEED) {
        ++i;

        if (i >= CI->getNumArgOperands()) {
          EmitFailure("MissingArgShadow", CI->getDebugLoc(), CI,
                      "__enzyme_autodiff missing argument shadow at index ", i,
                      ", need shadow of type ", *PTy,
                      " to shadow primal argument ", *args.back(), " at call ",
                      *CI);
          return false;
        }
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
                llvm::errs() << "Warning cast(2) __enzyme_autodiff argument "
                             << i << " " << *res << "|" << *res->getType()
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

    if (args.size() != newFunc->getFunctionType()->getNumParams()) {
      llvm::errs() << *CI << "\n";
      llvm::errs() << *newFunc << "\n";
      for (auto arg : args) {
        llvm::errs() << " + " << *arg << "\n";
      }
      EmitFailure("TooFewArguments", CI->getDebugLoc(), CI,
                  "Too few arguments passed to __enzyme_autodiff");
      return false;
    }
    assert(args.size() == newFunc->getFunctionType()->getNumParams());
    CallInst *diffret = cast<CallInst>(Builder.CreateCall(newFunc, args));
    diffret->setCallingConv(CI->getCallingConv());
    diffret->setDebugLoc(CI->getDebugLoc());

    if (!diffret->getType()->isEmptyTy() && !diffret->getType()->isVoidTy()) {
      unsigned idxs[] = {0};
      auto diffreti = Builder.CreateExtractValue(diffret, idxs);
      if (diffreti->getType() == CI->getType()) {
        CI->replaceAllUsesWith(diffreti);
      } else if (diffret->getType() == CI->getType()) {
        CI->replaceAllUsesWith(diffret);
      } else {
        EmitFailure("IllegalReturnCast", CI->getDebugLoc(), CI,
                    "Cannot cast return type of gradient ",
                    *diffreti->getType(), *diffreti, ", to desired type ",
                    *CI->getType());
        return false;
      }
    } else {
      CI->replaceAllUsesWith(UndefValue::get(CI->getType()));
    }
    CI->eraseFromParent();

    if (PostOpt) {
#if LLVM_VERSION_MAJOR >= 11
      auto Params = llvm::getInlineParams();

      llvm::SetVector<CallInst *> Q;
      Q.insert(diffret);
      TargetLibraryInfoWrapperPass TLIWP(
          Triple(newFunc->getParent()->getTargetTriple()));
      while (Q.size()) {
        auto cur = *Q.begin();
        Function *outerFunc = cur->getParent()->getParent();
        llvm::OptimizationRemarkEmitter ORE(outerFunc);
        Q.erase(Q.begin());
        if (auto F = cur->getCalledFunction()) {
          if (!F->empty()) {
            // Garbage collect AC's created
            SmallVector<AssumptionCache *, 2> ACAlloc;
            auto getAC = [&](Function &F) -> llvm::AssumptionCache & {
              auto AC = new AssumptionCache(F);
              ACAlloc.push_back(AC);
              return *AC;
            };
            auto GetTLI =
                [&](llvm::Function &F) -> const llvm::TargetLibraryInfo & {
              return TLIWP.getTLI(F);
            };

            auto GetInlineCost = [&](CallBase &CB) {
              TargetTransformInfo TTI(F->getParent()->getDataLayout());
              auto cst = llvm::getInlineCost(CB, Params, TTI, getAC, GetTLI);
              return cst;
            };
            if (llvm::shouldInline(*cur, GetInlineCost, ORE)) {
              InlineFunctionInfo IFI;
              InlineResult IR =
#if LLVM_VERSION_MAJOR >= 11
                  InlineFunction(*cur, IFI);
#else
                  InlineFunction(cur, IFI);
#endif
              if (IR.isSuccess()) {
                for (auto U : outerFunc->users()) {
                  if (auto CI = dyn_cast<CallInst>(U)) {
                    if (CI->getCalledFunction() == outerFunc) {
                      Q.insert(CI);
                    }
                  }
                }
              }
            }
            for (auto AC : ACAlloc) {
              delete AC;
            }
          }
        }
      }
#endif
    }
    return true;
  }

  bool lowerEnzymeCalls(Function &F, bool PostOpt, bool &successful,
                        std::set<Function *> &done) {
    if (done.count(&F))
      return false;
    done.insert(&F);

    if (F.empty())
      return false;

#if LLVM_VERSION_MAJOR >= 10
    auto &TLI = getAnalysis<TargetLibraryInfoWrapperPass>().getTLI(F);
#else
    auto &TLI = getAnalysis<TargetLibraryInfoWrapperPass>().getTLI();
#endif

    AAResults AA(TLI);
    // auto &B_AA = getAnalysis<BasicAAWrapperPass>().getResult();
    // AA.addAAResult(B_AA);

    auto &G_AA = getAnalysis<GlobalsAAWrapperPass>().getResult();
    AA.addAAResult(G_AA);

    bool Changed = false;

    std::set<CallInst *> toLower;
    std::set<InvokeInst *> toLowerI;
    std::set<CallInst *> InactiveCalls;
  retry:;
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        CallInst *CI = dyn_cast<CallInst>(&I);
        if (!CI)
          continue;

        Function *Fn = CI->getCalledFunction();

#if LLVM_VERSION_MAJOR >= 11
        if (auto castinst = dyn_cast<ConstantExpr>(CI->getCalledOperand()))
#else
        if (auto castinst = dyn_cast<ConstantExpr>(CI->getCalledValue()))
#endif
        {
          if (castinst->isCast())
            if (auto fn = dyn_cast<Function>(castinst->getOperand(0)))
              Fn = fn;
        }

        if (Fn && Fn->getName() == "__enzyme_float") {
          Fn->addFnAttr(Attribute::ReadNone);
          CI->addAttribute(AttributeList::FunctionIndex, Attribute::ReadNone);
          for (size_t i = 0; i < CI->getNumArgOperands(); ++i) {
            if (CI->getArgOperand(i)->getType()->isPointerTy()) {
              CI->addParamAttr(i, Attribute::ReadNone);
              CI->addParamAttr(i, Attribute::NoCapture);
            }
          }
        }
        if (Fn && Fn->getName() == "__enzyme_integer") {
          Fn->addFnAttr(Attribute::ReadNone);
          CI->addAttribute(AttributeList::FunctionIndex, Attribute::ReadNone);
          for (size_t i = 0; i < CI->getNumArgOperands(); ++i) {
            if (CI->getArgOperand(i)->getType()->isPointerTy()) {
              CI->addParamAttr(i, Attribute::ReadNone);
              CI->addParamAttr(i, Attribute::NoCapture);
            }
          }
        }
        if (Fn && Fn->getName() == "__enzyme_double") {
          Fn->addFnAttr(Attribute::ReadNone);
          CI->addAttribute(AttributeList::FunctionIndex, Attribute::ReadNone);
          for (size_t i = 0; i < CI->getNumArgOperands(); ++i) {
            if (CI->getArgOperand(i)->getType()->isPointerTy()) {
              CI->addParamAttr(i, Attribute::ReadNone);
              CI->addParamAttr(i, Attribute::NoCapture);
            }
          }
        }
        if (Fn && Fn->getName() == "__enzyme_pointer") {
          Fn->addFnAttr(Attribute::ReadNone);
          CI->addAttribute(AttributeList::FunctionIndex, Attribute::ReadNone);
          for (size_t i = 0; i < CI->getNumArgOperands(); ++i) {
            if (CI->getArgOperand(i)->getType()->isPointerTy()) {
              CI->addParamAttr(i, Attribute::ReadNone);
              CI->addParamAttr(i, Attribute::NoCapture);
            }
          }
        }
        if (Fn && Fn->getName().contains("__enzyme_call_inactive")) {
          InactiveCalls.insert(CI);
        }
        if (Fn && (Fn->getName() == "__fd_sincos_1" ||
                   Fn->getName() == "__fd_cos_1" ||
                   Fn->getName() == "__mth_i_ipowi")) {
          Fn->addFnAttr(Attribute::ReadNone);
          CI->addAttribute(AttributeList::FunctionIndex, Attribute::ReadNone);
        }
        if (Fn && (Fn->getName() == "f90io_fmtw_end" ||
                   Fn->getName() == "f90io_unf_end")) {
          Fn->addFnAttr(Attribute::InaccessibleMemOnly);
          CI->addAttribute(AttributeList::FunctionIndex,
                           Attribute::InaccessibleMemOnly);
        }
        if (Fn && (Fn->getName() == "f90io_open2003a")) {
          Fn->addFnAttr(Attribute::InaccessibleMemOrArgMemOnly);
          CI->addAttribute(AttributeList::FunctionIndex,
                           Attribute::InaccessibleMemOrArgMemOnly);
          for (size_t i : {0, 1, 2, 3, 4, 5, 6, 7, /*8, */ 9, 10, 11, 12, 13}) {
            if (i < CI->getNumArgOperands() &&
                CI->getArgOperand(i)->getType()->isPointerTy()) {
              CI->addParamAttr(i, Attribute::ReadOnly);
            }
          }
          // todo more
          for (size_t i : {0, 1}) {
            if (i < CI->getNumArgOperands() &&
                CI->getArgOperand(i)->getType()->isPointerTy()) {
              CI->addParamAttr(i, Attribute::NoCapture);
            }
          }
        }
        if (Fn && (Fn->getName() == "f90io_fmtw_inita")) {
          Fn->addFnAttr(Attribute::InaccessibleMemOrArgMemOnly);
          CI->addAttribute(AttributeList::FunctionIndex,
                           Attribute::InaccessibleMemOrArgMemOnly);
          // todo more
          for (size_t i : {0, 2}) {
            if (i < CI->getNumArgOperands() &&
                CI->getArgOperand(i)->getType()->isPointerTy()) {
              CI->addParamAttr(i, Attribute::ReadOnly);
            }
          }

          // todo more
          for (size_t i : {0, 2}) {
            if (i < CI->getNumArgOperands() &&
                CI->getArgOperand(i)->getType()->isPointerTy()) {
              CI->addParamAttr(i, Attribute::NoCapture);
            }
          }
        }

        if (Fn && (Fn->getName() == "f90io_unf_init")) {
          Fn->addFnAttr(Attribute::InaccessibleMemOrArgMemOnly);
          CI->addAttribute(AttributeList::FunctionIndex,
                           Attribute::InaccessibleMemOrArgMemOnly);
          // todo more
          for (size_t i : {0, 1, 2, 3}) {
            if (i < CI->getNumArgOperands() &&
                CI->getArgOperand(i)->getType()->isPointerTy()) {
              CI->addParamAttr(i, Attribute::ReadOnly);
            }
          }

          // todo more
          for (size_t i : {0, 1, 2, 3}) {
            if (i < CI->getNumArgOperands() &&
                CI->getArgOperand(i)->getType()->isPointerTy()) {
              CI->addParamAttr(i, Attribute::NoCapture);
            }
          }
        }

        if (Fn && (Fn->getName() == "f90io_src_info03a")) {
          Fn->addFnAttr(Attribute::InaccessibleMemOrArgMemOnly);
          CI->addAttribute(AttributeList::FunctionIndex,
                           Attribute::InaccessibleMemOrArgMemOnly);
          // todo more
          for (size_t i : {0, 1}) {
            if (i < CI->getNumArgOperands() &&
                CI->getArgOperand(i)->getType()->isPointerTy()) {
              CI->addParamAttr(i, Attribute::ReadOnly);
            }
          }

          // todo more
          for (size_t i : {0}) {
            if (i < CI->getNumArgOperands() &&
                CI->getArgOperand(i)->getType()->isPointerTy()) {
              CI->addParamAttr(i, Attribute::NoCapture);
            }
          }
        }
        if (Fn && (Fn->getName() == "f90io_sc_d_fmt_write" ||
                   Fn->getName() == "f90io_sc_i_fmt_write" ||
                   Fn->getName() == "ftnio_fmt_write64" ||
                   Fn->getName() == "f90io_fmt_write64_aa" ||
                   Fn->getName() == "f90io_fmt_writea" ||
                   Fn->getName() == "f90io_unf_writea" ||
                   Fn->getName() == "f90_pausea")) {
          Fn->addFnAttr(Attribute::InaccessibleMemOrArgMemOnly);
          CI->addAttribute(AttributeList::FunctionIndex,
                           Attribute::InaccessibleMemOrArgMemOnly);
          for (size_t i = 0; i < CI->getNumArgOperands(); ++i) {
            if (CI->getArgOperand(i)->getType()->isPointerTy()) {
              CI->addParamAttr(i, Attribute::ReadOnly);
              CI->addParamAttr(i, Attribute::NoCapture);
            }
          }
        }

        if (Fn && (Fn->getName() == "__enzyme_autodiff" ||
                   Fn->getName() == "enzyme_autodiff_" ||
                   Fn->getName().startswith("__enzyme_autodiff") ||
                   Fn->getName().contains("__enzyme_autodiff"))) {
          toLower.insert(CI);

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
          if (auto si = dyn_cast<SelectInst>(fn)) {
            BasicBlock *post = BB.splitBasicBlock(CI);
            BasicBlock *sel1 = BasicBlock::Create(BB.getContext(), "sel1", &F);
            BasicBlock *sel2 = BasicBlock::Create(BB.getContext(), "sel2", &F);
            BB.getTerminator()->eraseFromParent();
            IRBuilder<> PB(&BB);
            PB.CreateCondBr(si->getCondition(), sel1, sel2);
            IRBuilder<> S1(sel1);
            auto B1 = S1.CreateBr(post);
            CallInst *cloned = cast<CallInst>(CI->clone());
            cloned->insertBefore(B1);
            cloned->setOperand(0, si->getTrueValue());
            IRBuilder<> S2(sel2);
            auto B2 = S2.CreateBr(post);
            CI->moveBefore(B2);
            CI->setOperand(0, si->getFalseValue());
            if (CI->getNumUses() != 0) {
              IRBuilder<> P(post->getFirstNonPHI());
              auto merge = P.CreatePHI(CI->getType(), 2);
              merge->addIncoming(cloned, sel1);
              merge->addIncoming(CI, sel2);
              CI->replaceAllUsesWith(merge);
            }
            goto retry;
          }
          if (auto dc = dyn_cast<Function>(fn))
            Changed |=
                lowerEnzymeCalls(*dc, /*PostOpt*/ true, successful, done);
        }
      }
    }

    for (auto CI : InactiveCalls) {
      IRBuilder<> B(CI);
      Value *fn = CI->getArgOperand(0);
      SmallVector<Value *, 4> Args;
      SmallVector<Type *, 4> ArgTypes;
      for (size_t i = 1; i < CI->getNumArgOperands(); ++i) {
        Args.push_back(CI->getArgOperand(i));
        ArgTypes.push_back(CI->getArgOperand(i)->getType());
      }
      auto FT = FunctionType::get(CI->getType(), ArgTypes, /*varargs*/ false);
      if (fn->getType() != FT) {
        fn = B.CreatePointerCast(fn, PointerType::getUnqual(FT));
      }
      auto Rep = B.CreateCall(FT, fn, Args);
      Rep->addAttribute(AttributeList::FunctionIndex,
                        Attribute::get(Rep->getContext(), "enzyme_inactive"));
      CI->replaceAllUsesWith(Rep);
      CI->eraseFromParent();
      Changed = true;
    }
    for (auto CI : toLower) {
      successful &= HandleAutoDiff(CI, TLI, AA, PostOpt);
      Changed = true;
      if (!successful)
        break;
    }
    for (auto CI : toLowerI) {
      successful &= HandleAutoDiff(CI, TLI, AA, PostOpt);
      Changed = true;
      if (!successful)
        break;
    }

    if (Changed) {
      // TODO consider enabling when attributor does not delete
      // dead internal functions, which invalidates Enzyme's cache
      // code left here to re-enable upon Attributor patch

#if 0
      AnalysisGetter AG;
      SetVector<Function *> Functions;
      for (Function &F2 : *F.getParent()) {
        Functions.insert(&F2);
      }
    
      CallGraphUpdater CGUpdater;
      BumpPtrAllocator Allocator;
      InformationCache InfoCache(*F.getParent(), AG, Allocator, /* CGSCC */ nullptr);

      DenseSet< const char * > Allowed = {
        &AAMemoryBehavior::ID,
        &AAMemoryLocation::ID,
        
        &AANoUnwind::ID,
        &AANoSync::ID,
        &AANoRecurse::ID,
        &AAWillReturn::ID,
        &AANoReturn::ID,
        &AANonNull::ID,
        &AANoAlias::ID,
        &AADereferenceable::ID,
        &AAAlign::ID,
        &AANoCapture::ID,

        &AAReturnedValues::ID,
        &AANoFree::ID,
        &AANoUndef::ID,
        //&AAValueSimplify::ID,
        //&AAReachability::ID,
        //&AAValueConstantRange::ID,
        //&AAUndefinedBehavior::ID,
        //&AAPotentialValues::ID,
      };
      Attributor(Functions, InfoCache, CGUpdater, &Allowed).run();
#endif
    }

    return Changed;
  }

  bool runOnModule(Module &M) override {
    // auto &AA = getAnalysis<AAResultsWrapperPass>().getAAResults();

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
      std::vector<Instruction *> toErase;
      for (BasicBlock &BB : F) {
        for (Instruction &I : BB) {
          if (auto CI = dyn_cast<CallInst>(&I)) {
            Function *F = CI->getCalledFunction();
#if LLVM_VERSION_MAJOR >= 11
            if (auto castinst = dyn_cast<ConstantExpr>(CI->getCalledOperand()))
#else
            if (auto castinst = dyn_cast<ConstantExpr>(CI->getCalledValue()))
#endif
            {
              if (castinst->isCast())
                if (auto fn = dyn_cast<Function>(castinst->getOperand(0))) {
                  F = fn;
                }
            }
            if (F && F->getName() == "f90_mzero8") {
              toErase.push_back(CI);
              IRBuilder<> B(CI);

              SmallVector<Value *, 4> args;
              args.push_back(CI->getArgOperand(0));
              args.push_back(
                  ConstantInt::get(Type::getInt8Ty(M.getContext()), 0));
              args.push_back(B.CreateMul(
                  CI->getArgOperand(1),
                  ConstantInt::get(CI->getArgOperand(1)->getType(), 8)));
#if LLVM_VERSION_MAJOR <= 6
              args.push_back(
                  ConstantInt::get(Type::getInt32Ty(M.getContext()), 1U));
#endif
              args.push_back(ConstantInt::getFalse(M.getContext()));

              Type *tys[] = {args[0]->getType(), args[2]->getType()};
              auto memsetIntr =
                  Intrinsic::getDeclaration(&M, Intrinsic::memset, tys);
              B.CreateCall(memsetIntr, args);
            }
          }
        }
      }
      for (Instruction *I : toErase) {
        I->eraseFromParent();
      }
    }
    std::set<Function *> done;
    for (Function &F : M) {
      if (F.empty())
        continue;

      // auto &AA = getAnalysis<AAResultsWrapperPass>().getAAResults();
      // auto &LI = getAnalysis<LoopInfoWrapperPass>(F).getLoopInfo();

      bool successful = true;
      changed |= lowerEnzymeCalls(F, PostOpt, successful, done);

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
            Function *F = CI->getCalledFunction();
#if LLVM_VERSION_MAJOR >= 11
            if (auto castinst = dyn_cast<ConstantExpr>(CI->getCalledOperand()))
#else
            if (auto castinst = dyn_cast<ConstantExpr>(CI->getCalledValue()))
#endif
            {
              if (castinst->isCast())
                if (auto fn = dyn_cast<Function>(castinst->getOperand(0))) {
                  F = fn;
                }
            }
            if (F) {
              if (F->getName() == "__enzyme_float" ||
                  F->getName() == "__enzyme_double" ||
                  F->getName() == "__enzyme_integer" ||
                  F->getName() == "__enzyme_pointer") {
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

#include <llvm-c/Core.h>
#include <llvm-c/Types.h>

#include "llvm/IR/LegacyPassManager.h"

extern "C" void AddEnzymePass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createEnzymePass(/*PostOpt*/ false));
}
