//===- FunctionUtils.cpp - Implementation of function utilities -----------===//
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
// This file defines utilities on LLVM Functions that are used as part of the AD
// process.
//
//===----------------------------------------------------------------------===//
#include "FunctionUtils.h"

#include "EnzymeLogic.h"
#include "GradientUtils.h"
#include "LibraryFuncs.h"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/LazyValueInfo.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/MemoryDependenceAnalysis.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"

#include "llvm/Analysis/TypeBasedAliasAnalysis.h"

#if LLVM_VERSION_MAJOR > 6
#include "llvm/Analysis/PhiValues.h"
#endif
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScopedNoAliasAA.h"
#include "llvm/Analysis/TargetTransformInfo.h"

#include "llvm/Transforms/IPO/FunctionAttrs.h"

#if LLVM_VERSION_MAJOR > 6
#include "llvm/Transforms/Utils.h"
#endif

#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"

#if LLVM_VERSION_MAJOR > 6
#include "llvm/Transforms/Scalar/InstSimplifyPass.h"
#endif

#include "llvm/Transforms/Scalar/MemCpyOptimizer.h"

#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar/CorrelatedValuePropagation.h"
#include "llvm/Transforms/Scalar/DCE.h"
#include "llvm/Transforms/Scalar/DeadStoreElimination.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Scalar/IndVarSimplify.h"
#include "llvm/Transforms/Scalar/LoopIdiomRecognize.h"
#include "llvm/Transforms/Scalar/SROA.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"
#include "llvm/Transforms/Utils/LCSSA.h"

#include "llvm/Transforms/IPO/FunctionAttrs.h"
#include "llvm/Transforms/Scalar/LoopDeletion.h"
#include "llvm/Transforms/Scalar/LoopRotation.h"

#define DEBUG_TYPE "enzyme"
using namespace llvm;

static cl::opt<bool>
    EnzymePreopt("enzyme-preopt", cl::init(true), cl::Hidden,
                 cl::desc("Run enzyme preprocessing optimizations"));

static cl::opt<bool> EnzymeInline("enzyme-inline", cl::init(false), cl::Hidden,
                                  cl::desc("Force inlining of autodiff"));

static cl::opt<int>
    EnzymeInlineCount("enzyme-inline-count", cl::init(10000), cl::Hidden,
                      cl::desc("Limit of number of functions to inline"));

// Locally run mem2reg on F, if ASsumptionCache AC is given it will
// be updated
static bool PromoteMemoryToRegister(Function &F, DominatorTree &DT,
                                    AssumptionCache *AC = nullptr) {
  std::vector<AllocaInst *> Allocas;
  BasicBlock &BB = F.getEntryBlock(); // Get the entry node for the function
  bool Changed = false;

  while (true) {
    Allocas.clear();

    // Find allocas that are safe to promote, by looking at all instructions in
    // the entry node
    for (BasicBlock::iterator I = BB.begin(), E = --BB.end(); I != E; ++I)
      if (AllocaInst *AI = dyn_cast<AllocaInst>(I)) // Is it an alloca?
        if (isAllocaPromotable(AI))
          Allocas.push_back(AI);

    if (Allocas.empty())
      break;

    PromoteMemToReg(Allocas, DT, AC);
    Changed = true;
  }
  return Changed;
}

/// Return whether this function eventually calls itself
static bool IsFunctionRecursive(Function *F) {
  enum RecurType {
    MaybeRecursive = 1,
    NotRecursive = 2,
    DefinitelyRecursive = 3,
  };

  static std::map<const Function *, RecurType> Results;

  // If we haven't seen this function before, look at all callers
  // and mark this as potentially recursive. If we see this function
  // still as marked as MaybeRecursive, we will definitionally have
  // found an eventual caller of the original function. If not,
  // the function does not eventually call itself (in a static way)
  if (Results.find(F) == Results.end()) {
    Results[F] = MaybeRecursive; // staging
    for (auto &BB : *F) {
      for (auto &I : BB) {
        if (auto call = dyn_cast<CallInst>(&I)) {
          if (call->getCalledFunction() == nullptr)
            continue;
          if (call->getCalledFunction()->empty())
            continue;
          IsFunctionRecursive(call->getCalledFunction());
        }
        if (auto call = dyn_cast<InvokeInst>(&I)) {
          if (call->getCalledFunction() == nullptr)
            continue;
          if (call->getCalledFunction()->empty())
            continue;
          IsFunctionRecursive(call->getCalledFunction());
        }
      }
    }
    if (Results[F] == MaybeRecursive) {
      Results[F] = NotRecursive; // not recursive
    }
  } else if (Results[F] == MaybeRecursive) {
    Results[F] = DefinitelyRecursive; // definitely recursive
  }
  assert(Results[F] != MaybeRecursive);
  return Results[F] == DefinitelyRecursive;
}

static inline bool OnlyUsedInOMP(AllocaInst *AI) {
  bool ompUse = false;
  for (auto U : AI->users()) {
    if (isa<StoreInst>(U))
      continue;
    if (auto CI = dyn_cast<CallInst>(U)) {
      if (auto F = CI->getCalledFunction()) {
        if (F->getName() == "__kmpc_for_static_init_4") {
          ompUse = true;
        }
      }
    }
  }

  if (!ompUse)
    return false;
  return true;
}
/// Convert necessary stack allocations into mallocs for use in the reverse
/// pass. Specifically if we're not topLevel all allocations must be upgraded
/// Even if topLevel any allocations that aren't in the entry block (and
/// therefore may not be reachable in the reverse pass) must be upgraded.
static inline void UpgradeAllocasToMallocs(Function *NewF, bool topLevel) {
  std::vector<AllocaInst *> ToConvert;

  for (auto &BB : *NewF) {
    for (auto &I : BB) {
      if (auto AI = dyn_cast<AllocaInst>(&I)) {
        bool UsableEverywhere = AI->getParent() == &NewF->getEntryBlock();
        // TODO use is_value_needed_in_reverse (requiring GradientUtils)
        if (OnlyUsedInOMP(AI))
          continue;
        if (!UsableEverywhere || !topLevel) {
          ToConvert.push_back(AI);
        }
      }
    }
  }

  for (auto AI : ToConvert) {
    std::string nam = AI->getName().str();
    AI->setName("");

    // Ensure we insert the malloc after the allocas
    Instruction *insertBefore = AI;
    while (isa<AllocaInst>(insertBefore->getNextNode())) {
      insertBefore = insertBefore->getNextNode();
      assert(insertBefore);
    }

    auto i64 = Type::getInt64Ty(NewF->getContext());
    auto rep = CallInst::CreateMalloc(
        insertBefore, i64, AI->getAllocatedType(),
        ConstantInt::get(
            i64, NewF->getParent()->getDataLayout().getTypeAllocSizeInBits(
                     AI->getAllocatedType()) /
                     8),
        IRBuilder<>(insertBefore).CreateZExtOrTrunc(AI->getArraySize(), i64),
        nullptr, nam);
    assert(rep->getType() == AI->getType());
    AI->replaceAllUsesWith(rep);
    AI->eraseFromParent();
  }
}

/// Perform recursive inlinining on NewF up to the given limit
static void ForceRecursiveInlining(Function *NewF, size_t Limit) {
  for (size_t count = 0; count < Limit; count++) {
    for (auto &BB : *NewF) {
      for (auto &I : BB) {
        if (auto CI = dyn_cast<CallInst>(&I)) {
          if (CI->getCalledFunction() == nullptr)
            continue;
          if (CI->getCalledFunction()->empty())
            continue;
          if (CI->getCalledFunction()->hasFnAttribute(
                  Attribute::ReturnsTwice) ||
              CI->getCalledFunction()->hasFnAttribute(Attribute::NoInline))
            continue;
          if (IsFunctionRecursive(CI->getCalledFunction())) {
            LLVM_DEBUG(llvm::dbgs()
                       << "not inlining recursive "
                       << CI->getCalledFunction()->getName() << "\n");
            continue;
          }
          InlineFunctionInfo IFI;
#if LLVM_VERSION_MAJOR >= 11
          InlineFunction(*CI, IFI);
#else
          InlineFunction(CI, IFI);
#endif
          goto outermostContinue;
        }
      }
    }

    // No functions were inlined, break
    break;

  outermostContinue:;
  }
}

Function *preprocessForClone(Function *F, AAResults &AA, TargetLibraryInfo &TLI,
                             bool topLevel) {
  static std::map<std::pair<Function *, bool>, Function *> cache;

  // If we've already processed this, return the previous version
  // and derive aliasing information
  if (cache.find(std::make_pair(F, topLevel)) != cache.end()) {
    Function *NewF = cache[std::make_pair(F, topLevel)];
    AssumptionCache *AC = new AssumptionCache(*NewF);
    DominatorTree *DT = new DominatorTree(*NewF);
    LoopInfo *LI = new LoopInfo(*DT);
#if LLVM_VERSION_MAJOR > 6
    PhiValues *PV = new PhiValues(*NewF);
#endif
    auto BAA = new BasicAAResult(NewF->getParent()->getDataLayout(),
#if LLVM_VERSION_MAJOR > 6
                                 *NewF,
#endif
                                 TLI, *AC, DT, LI
#if LLVM_VERSION_MAJOR > 6
                                 ,
                                 PV
#endif
    );
    AA.addAAResult(*BAA);
    AA.addAAResult(*(new TypeBasedAAResult()));
    return NewF;
  }

  Function *NewF =
      Function::Create(F->getFunctionType(), F->getLinkage(),
                       "preprocess_" + F->getName(), F->getParent());

  ValueToValueMapTy VMap;
  for (auto i = F->arg_begin(), j = NewF->arg_begin(); i != F->arg_end();) {
    VMap[i] = j;
    j->setName(i->getName());
    ++i;
    ++j;
  }

  SmallVector<ReturnInst *, 4> Returns;
  CloneFunctionInto(NewF, F, VMap, F->getSubprogram() != nullptr, Returns, "",
                    nullptr);
  NewF->setAttributes(F->getAttributes());

  if (EnzymePreopt) {
    if (EnzymeInline) {
      ForceRecursiveInlining(NewF, /*Limit*/ EnzymeInlineCount);
    }
  }

  {
    std::vector<Instruction *> FreesToErase;
    for (auto &BB : *NewF) {
      for (auto &I : BB) {

        if (auto CI = dyn_cast<CallInst>(&I)) {

          Function *called = CI->getCalledFunction();
#if LLVM_VERSION_MAJOR >= 11
          if (auto castinst = dyn_cast<ConstantExpr>(CI->getCalledOperand()))
#else
          if (auto castinst = dyn_cast<ConstantExpr>(CI->getCalledValue()))
#endif
          {
            if (castinst->isCast()) {
              if (auto fn = dyn_cast<Function>(castinst->getOperand(0))) {
                if (isDeallocationFunction(*fn, TLI)) {
                  called = fn;
                }
              }
            }
          }

          if (called && isDeallocationFunction(*called, TLI)) {
            FreesToErase.push_back(CI);
          }
        }
      }
    }
    // TODO we should ensure these are kept to avoid accidentially creating
    // a memory leak
    for (auto Free : FreesToErase) {
      Free->eraseFromParent();
    }
  }

  if (EnzymePreopt) {
    if (EnzymeInline) {
      {
        DominatorTree DT(*NewF);
        PromoteMemoryToRegister(*NewF, DT);
      }

      {
        FunctionAnalysisManager AM;
        AM.registerPass([] { return AAManager(); });
        AM.registerPass([] { return ScalarEvolutionAnalysis(); });
        AM.registerPass([] { return AssumptionAnalysis(); });
        AM.registerPass([] { return TargetLibraryAnalysis(); });
        AM.registerPass([] { return TargetIRAnalysis(); });
        AM.registerPass([] { return MemorySSAAnalysis(); });
        AM.registerPass([] { return DominatorTreeAnalysis(); });
        AM.registerPass([] { return MemoryDependenceAnalysis(); });
        AM.registerPass([] { return LoopAnalysis(); });
        AM.registerPass([] { return OptimizationRemarkEmitterAnalysis(); });
#if LLVM_VERSION_MAJOR > 6
        AM.registerPass([] { return PhiValuesAnalysis(); });
#endif
        AM.registerPass([] { return LazyValueAnalysis(); });
#if LLVM_VERSION_MAJOR > 10
        AM.registerPass([] { return PassInstrumentationAnalysis(); });
#endif
#if LLVM_VERSION_MAJOR <= 7
        GVN().run(*NewF, AM);
        SROA().run(*NewF, AM);
#endif
      }
    }

    {
      DominatorTree DT(*NewF);
      PromoteMemoryToRegister(*NewF, DT);
    }

    {
      FunctionAnalysisManager AM;
      AM.registerPass([] { return AAManager(); });
      AM.registerPass([] { return ScalarEvolutionAnalysis(); });
      AM.registerPass([] { return AssumptionAnalysis(); });
      AM.registerPass([] { return TargetLibraryAnalysis(); });
      AM.registerPass([] { return TargetIRAnalysis(); });
      AM.registerPass([] { return MemorySSAAnalysis(); });
      AM.registerPass([] { return DominatorTreeAnalysis(); });
      AM.registerPass([] { return MemoryDependenceAnalysis(); });
      AM.registerPass([] { return LoopAnalysis(); });
      AM.registerPass([] { return OptimizationRemarkEmitterAnalysis(); });
#if LLVM_VERSION_MAJOR > 6
      AM.registerPass([] { return PhiValuesAnalysis(); });
#endif
#if LLVM_VERSION_MAJOR >= 8
      AM.registerPass([] { return PassInstrumentationAnalysis(); });
#endif
      AM.registerPass([] { return LazyValueAnalysis(); });
      SROA().run(*NewF, AM);

#if LLVM_VERSION_MAJOR >= 12
      SimplifyCFGOptions scfgo;
#else
      SimplifyCFGOptions scfgo(
          /*unsigned BonusThreshold=*/1, /*bool ForwardSwitchCond=*/false,
          /*bool SwitchToLookup=*/false, /*bool CanonicalLoops=*/true,
          /*bool SinkCommon=*/true, /*AssumptionCache *AssumpCache=*/nullptr);
#endif
      SimplifyCFGPass(scfgo).run(*NewF, AM);
    }
  }

  // Run LoopSimplifyPass to ensure preheaders exist on all loops
  {
    FunctionAnalysisManager AM;
    AM.registerPass([] { return LoopAnalysis(); });
    AM.registerPass([] { return DominatorTreeAnalysis(); });
    AM.registerPass([] { return ScalarEvolutionAnalysis(); });
    AM.registerPass([] { return AssumptionAnalysis(); });
#if LLVM_VERSION_MAJOR >= 8
    AM.registerPass([] { return PassInstrumentationAnalysis(); });
#endif
#if LLVM_VERSION_MAJOR >= 11
    AM.registerPass([] { return MemorySSAAnalysis(); });
#endif
    LoopSimplifyPass().run(*NewF, AM);
  }

  // For subfunction calls upgrade stack allocations to mallocs
  // to ensure availability in the reverse pass
  // TODO we should ensure these are kept to avoid accidentially creating
  // a memory leak
  UpgradeAllocasToMallocs(NewF, topLevel);

  {
    // Alias analysis is necessary to ensure can query whether we can move a
    // forward pass function
    AssumptionCache *AC = new AssumptionCache(*NewF);
    DominatorTree *DT = new DominatorTree(*NewF);
    LoopInfo *LI = new LoopInfo(*DT);
#if LLVM_VERSION_MAJOR > 6
    PhiValues *PV = new PhiValues(*NewF);
#endif
    auto BAA = new BasicAAResult(NewF->getParent()->getDataLayout(),
#if LLVM_VERSION_MAJOR > 6
                                 *NewF,
#endif
                                 TLI, *AC, DT, LI
#if LLVM_VERSION_MAJOR > 6
                                 ,
                                 PV
#endif
    );
    AA.addAAResult(*BAA);
    AA.addAAResult(*(new TypeBasedAAResult()));
  }

  if (EnzymePrint)
    llvm::errs() << "after simplification :\n" << *NewF << "\n";

  if (llvm::verifyFunction(*NewF, &llvm::errs())) {
    llvm::errs() << *NewF << "\n";
    report_fatal_error("function failed verification (1)");
  }
  cache[std::make_pair(F, topLevel)] = NewF;
  return NewF;
}

Function *CloneFunctionWithReturns(
    bool topLevel, Function *&F, AAResults &AA, TargetLibraryInfo &TLI,
    ValueToValueMapTy &ptrInputs, const std::vector<DIFFE_TYPE> &constant_args,
    SmallPtrSetImpl<Value *> &constants, SmallPtrSetImpl<Value *> &nonconstant,
    SmallPtrSetImpl<Value *> &returnvals, ReturnType returnValue, Twine name,
    ValueToValueMapTy *VMapO, bool diffeReturnArg, llvm::Type *additionalArg) {
  assert(!F->empty());
  F = preprocessForClone(F, AA, TLI, topLevel);
  std::vector<Type *> RetTypes;
  if (returnValue == ReturnType::ArgsWithReturn ||
      returnValue == ReturnType::ArgsWithTwoReturns)
    RetTypes.push_back(F->getReturnType());
  if (returnValue == ReturnType::ArgsWithTwoReturns)
    RetTypes.push_back(F->getReturnType());
  std::vector<Type *> ArgTypes;

  ValueToValueMapTy VMap;

  // The user might be deleting arguments to the function by specifying them in
  // the VMap.  If so, we need to not add the arguments to the arg ty vector
  unsigned argno = 0;
  for (const Argument &I : F->args()) {
    ArgTypes.push_back(I.getType());
    if (constant_args[argno] == DIFFE_TYPE::DUP_ARG ||
        constant_args[argno] == DIFFE_TYPE::DUP_NONEED) {
      ArgTypes.push_back(I.getType());
    } else if (constant_args[argno] == DIFFE_TYPE::OUT_DIFF) {
      RetTypes.push_back(I.getType());
    }
    ++argno;
  }

  for (BasicBlock &BB : *F) {
    for (Instruction &I : BB) {
      if (auto ri = dyn_cast<ReturnInst>(&I)) {
        if (auto rv = ri->getReturnValue()) {
          returnvals.insert(rv);
        }
      }
    }
  }

  if (diffeReturnArg) {
    assert(!F->getReturnType()->isVoidTy());
    ArgTypes.push_back(F->getReturnType());
  }
  if (additionalArg) {
    ArgTypes.push_back(additionalArg);
  }
  Type *RetType = StructType::get(F->getContext(), RetTypes);
  if (returnValue == ReturnType::TapeAndTwoReturns ||
      returnValue == ReturnType::TapeAndReturn ||
      returnValue == ReturnType::Tape) {
    RetTypes.clear();
    RetTypes.push_back(Type::getInt8PtrTy(F->getContext()));
    if (returnValue == ReturnType::TapeAndTwoReturns) {
      RetTypes.push_back(F->getReturnType());
      RetTypes.push_back(F->getReturnType());
    } else if (returnValue == ReturnType::TapeAndReturn) {
      RetTypes.push_back(F->getReturnType());
    }
    RetType = StructType::get(F->getContext(), RetTypes);
  }

  bool noReturn = RetTypes.size() == 0;
  if (noReturn)
    RetType = Type::getVoidTy(RetType->getContext());

  // Create a new function type...
  FunctionType *FTy =
      FunctionType::get(RetType, ArgTypes, F->getFunctionType()->isVarArg());

  // Create the new function...
  Function *NewF = Function::Create(FTy, F->getLinkage(), name, F->getParent());
  if (diffeReturnArg) {
    auto I = NewF->arg_end();
    I--;
    if (additionalArg)
      I--;
    I->setName("differeturn");
  }
  if (additionalArg) {
    auto I = NewF->arg_end();
    I--;
    I->setName("tapeArg");
  }

  {
    unsigned ii = 0;
    for (auto i = F->arg_begin(), j = NewF->arg_begin(); i != F->arg_end();) {
      VMap[i] = j;
      ++j;
      ++i;
      if (constant_args[ii] == DIFFE_TYPE::DUP_ARG ||
          constant_args[ii] == DIFFE_TYPE::DUP_NONEED) {
        ++j;
      }
      ++ii;
    }
  }

  // Loop over the arguments, copying the names of the mapped arguments over...
  Function::arg_iterator DestI = NewF->arg_begin();

  for (const Argument &I : F->args())
    if (VMap.count(&I) == 0) {     // Is this argument preserved?
      DestI->setName(I.getName()); // Copy the name over...
      VMap[&I] = &*DestI++;        // Add mapping to VMap
    }
  SmallVector<ReturnInst *, 4> Returns;
  CloneFunctionInto(NewF, F, VMap, F->getSubprogram() != nullptr, Returns, "",
                    nullptr);
  if (VMapO)
    VMapO->insert(VMap.begin(), VMap.end());

  bool hasPtrInput = false;
  unsigned ii = 0, jj = 0;
  for (auto i = F->arg_begin(), j = NewF->arg_begin(); i != F->arg_end();) {
    if (constant_args[ii] == DIFFE_TYPE::CONSTANT) {
      constants.insert(i);
      if (printconst)
        llvm::errs() << "in new function " << NewF->getName()
                     << " constant arg " << *j << "\n";
    } else {
      nonconstant.insert(i);
      if (printconst)
        llvm::errs() << "in new function " << NewF->getName()
                     << " nonconstant arg " << *j << "\n";
    }

    if (constant_args[ii] == DIFFE_TYPE::DUP_ARG ||
        constant_args[ii] == DIFFE_TYPE::DUP_NONEED) {
      hasPtrInput = true;
      ptrInputs[i] = (j + 1);
      if (F->hasParamAttribute(ii, Attribute::NoCapture)) {
        NewF->addParamAttr(jj + 1, Attribute::NoCapture);
      }

      j->setName(i->getName());
      ++j;
      j->setName(i->getName() + "'");
      nonconstant.insert(j);
      ++j;
      jj += 2;

      ++i;

    } else {
      j->setName(i->getName());
      ++j;
      ++jj;
      ++i;
    }
    ++ii;
  }

  if (hasPtrInput) {
    if (NewF->hasFnAttribute(Attribute::ReadNone)) {
      NewF->removeFnAttr(Attribute::ReadNone);
    }
    if (NewF->hasFnAttribute(Attribute::ReadOnly)) {
      NewF->removeFnAttr(Attribute::ReadOnly);
    }
  }
  NewF->setLinkage(Function::LinkageTypes::InternalLinkage);
  assert(NewF->hasLocalLinkage());

  return NewF;
}

void optimizeIntermediate(GradientUtils *gutils, bool topLevel, Function *F) {
  {
    DominatorTree DT(*F);
    PromoteMemoryToRegister(*F, DT);
  }

  FunctionAnalysisManager AM;
  AM.registerPass([] { return AAManager(); });
  AM.registerPass([] { return ScalarEvolutionAnalysis(); });
  AM.registerPass([] { return AssumptionAnalysis(); });
  AM.registerPass([] { return TargetLibraryAnalysis(); });
  AM.registerPass([] { return TargetIRAnalysis(); });
  AM.registerPass([] { return MemorySSAAnalysis(); });
  AM.registerPass([] { return DominatorTreeAnalysis(); });
  AM.registerPass([] { return MemoryDependenceAnalysis(); });
  AM.registerPass([] { return LoopAnalysis(); });
  AM.registerPass([] { return OptimizationRemarkEmitterAnalysis(); });
#if LLVM_VERSION_MAJOR > 6
  AM.registerPass([] { return PhiValuesAnalysis(); });
#endif
  AM.registerPass([] { return LazyValueAnalysis(); });
  LoopAnalysisManager LAM;
  AM.registerPass([&] { return LoopAnalysisManagerFunctionProxy(LAM); });
  LAM.registerPass([&] { return FunctionAnalysisManagerLoopProxy(AM); });

#if LLVM_VERSION_MAJOR <= 7
  GVN().run(*F, AM);
  SROA().run(*F, AM);
  EarlyCSEPass(/*memoryssa*/ true).run(*F, AM);
#endif

  DCEPass().run(*F, AM);
}
