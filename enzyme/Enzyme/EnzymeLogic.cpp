//===- EnzymeLogic.cpp - Implementation of forward and reverse pass generation//
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
// This file defines two functions CreatePrimalAndGradient and
// CreateAugmentedPrimal. CreatePrimalAndGradient takes a function, known
// TypeResults of the calling context, known activity analysis of the
// arguments. It creates a corresponding gradient
// function, computing the primal as well if requested.
// CreateAugmentedPrimal takes similar arguments and creates an augmented
// primal pass.
//
//===----------------------------------------------------------------------===//
#include "AdjointGenerator.h"

#include "SCEV/ScalarEvolution.h"
#include "SCEV/ScalarEvolutionExpander.h"

#include "llvm/Analysis/DependenceAnalysis.h"
#include <deque>

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueTracking.h"

#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/GlobalsModRef.h"

#include "llvm/Support/AMDGPUMetadata.h"

#include "FunctionUtils.h"
#include "GradientUtils.h"
#include "InstructionBatcher.h"
#include "LibraryFuncs.h"
#include "Utils.h"

#if LLVM_VERSION_MAJOR >= 14
#define addAttribute addAttributeAtIndex
#define removeAttribute removeAttributeAtIndex
#endif

using namespace llvm;

extern "C" {
llvm::cl::opt<bool>
    EnzymePrint("enzyme-print", cl::init(false), cl::Hidden,
                cl::desc("Print before and after fns for autodiff"));

llvm::cl::opt<bool>
    EnzymePrintUnnecessary("enzyme-print-unnecessary", cl::init(false),
                           cl::Hidden,
                           cl::desc("Print unnecessary values in function"));

cl::opt<bool> looseTypeAnalysis("enzyme-loose-types", cl::init(false),
                                cl::Hidden,
                                cl::desc("Allow looser use of types"));

cl::opt<bool> nonmarkedglobals_inactiveloads(
    "enzyme_nonmarkedglobals_inactiveloads", cl::init(true), cl::Hidden,
    cl::desc("Consider loads of nonmarked globals to be inactive"));

cl::opt<bool> EnzymeJuliaAddrLoad(
    "enzyme-julia-addr-load", cl::init(false), cl::Hidden,
    cl::desc("Mark all loads resulting in an addr(13)* to be legal to redo"));
}

struct CacheAnalysis {

  const ValueMap<const CallInst *, SmallPtrSet<const CallInst *, 1>>
      &allocationsWithGuaranteedFree;
  const ValueMap<Value *, GradientUtils::Rematerializer>
      &rematerializableAllocations;
  TypeResults &TR;
  AAResults &AA;
  Function *oldFunc;
  ScalarEvolution &SE;
  LoopInfo &OrigLI;
  DominatorTree &OrigDT;
  TargetLibraryInfo &TLI;
  const SmallPtrSetImpl<const Instruction *> &unnecessaryInstructions;
  const std::map<Argument *, bool> &uncacheable_args;
  DerivativeMode mode;
  std::map<Value *, bool> seen;
  bool omp;
  CacheAnalysis(
      const ValueMap<const CallInst *, SmallPtrSet<const CallInst *, 1>>
          &allocationsWithGuaranteedFree,
      const ValueMap<Value *, GradientUtils::Rematerializer>
          &rematerializableAllocations,
      TypeResults &TR, AAResults &AA, Function *oldFunc, ScalarEvolution &SE,
      LoopInfo &OrigLI, DominatorTree &OrigDT, TargetLibraryInfo &TLI,
      const SmallPtrSetImpl<const Instruction *> &unnecessaryInstructions,
      const std::map<Argument *, bool> &uncacheable_args, DerivativeMode mode,
      bool omp)
      : allocationsWithGuaranteedFree(allocationsWithGuaranteedFree),
        rematerializableAllocations(rematerializableAllocations), TR(TR),
        AA(AA), oldFunc(oldFunc), SE(SE), OrigLI(OrigLI), OrigDT(OrigDT),
        TLI(TLI), unnecessaryInstructions(unnecessaryInstructions),
        uncacheable_args(uncacheable_args), mode(mode), omp(omp) {}

  bool is_value_mustcache_from_origin(Value *obj) {
    if (seen.find(obj) != seen.end())
      return seen[obj];

    bool mustcache = false;

    // If the pointer operand is from an argument to the function, we need to
    // check if the argument
    //   received from the caller is uncacheable.
    if (rematerializableAllocations.count(obj)) {
      return false;
    } else if (isa<UndefValue>(obj) || isa<ConstantPointerNull>(obj)) {
      return false;
    } else if (auto arg = dyn_cast<Argument>(obj)) {
      auto found = uncacheable_args.find(arg);
      if (found == uncacheable_args.end()) {
        llvm::errs() << "uncacheable_args:\n";
        for (auto &pair : uncacheable_args) {
          llvm::errs() << " + " << *pair.first << ": " << pair.second
                       << " of func " << pair.first->getParent()->getName()
                       << "\n";
        }
        llvm::errs() << "could not find " << *arg << " of func "
                     << arg->getParent()->getName() << " in args_map\n";
      }
      assert(found != uncacheable_args.end());
      if (found->second) {
        mustcache = true;
        // EmitWarning("UncacheableOrigin", arg->getDebugLoc(), oldFunc,
        // arg->getParent()->getEntryBlock(),
        //            "origin arg may need caching ", *arg);
      }
    } else if (auto pn = dyn_cast<PHINode>(obj)) {
      seen[pn] = false;
      for (auto &val : pn->incoming_values()) {
        if (is_value_mustcache_from_origin(val)) {
          mustcache = true;
          EmitWarning("UncacheableOrigin", pn->getDebugLoc(), oldFunc,
                      pn->getParent(), "origin pn may need caching ", *pn);
          break;
        }
      }
    } else if (auto ci = dyn_cast<CastInst>(obj)) {
      mustcache = is_value_mustcache_from_origin(ci->getOperand(0));
      if (mustcache) {
        EmitWarning("UncacheableOrigin", ci->getDebugLoc(), oldFunc,
                    ci->getParent(), "origin ci may need caching ", *ci);
      }
    } else if (auto gep = dyn_cast<GetElementPtrInst>(obj)) {
      mustcache = is_value_mustcache_from_origin(gep->getPointerOperand());
      if (mustcache) {
        EmitWarning("UncacheableOrigin", gep->getDebugLoc(), oldFunc,
                    gep->getParent(), "origin gep may need caching ", *gep);
      }
    } else {

      // Pointer operands originating from call instructions that are not
      // malloc/free are conservatively considered uncacheable.
      if (auto obj_op = dyn_cast<CallInst>(obj)) {
        // If this is a known allocation which is not captured or returned,
        // a caller function cannot overwrite this (since it cannot access).
        // Since we don't currently perform this check, we can instead check
        // if the pointer has a guaranteed free (which is a weaker form of
        // the required property).
        if (allocationsWithGuaranteedFree.find(obj_op) !=
            allocationsWithGuaranteedFree.end()) {

        } else {
          // OP is a non malloc/free call so we need to cache
          mustcache = true;
          EmitWarning("UncacheableOrigin", obj_op->getDebugLoc(), oldFunc,
                      obj_op->getParent(), "origin call may need caching ",
                      *obj_op);
        }
      } else if (isa<AllocaInst>(obj)) {
        // No change to modref if alloca since the memory only exists in
        // this function.
      } else if (auto GV = dyn_cast<GlobalVariable>(obj)) {
        // In the absense of more fine-grained global info, assume object is
        // written to in a subseqent call unless this is known to be constant
        if (!GV->isConstant()) {
          mustcache = true;
        }
      } else {
        // In absence of more information, assume that the underlying object for
        // pointer operand is uncacheable in caller.
        mustcache = true;
        if (isa<Instruction>(obj))
          EmitWarning("UncacheableOrigin",
                      cast<Instruction>(obj)->getDebugLoc(), oldFunc,
                      cast<Instruction>(obj)->getParent(),
                      "unknown origin may need caching ", *obj);
      }
    }

    return seen[obj] = mustcache;
  }

  bool is_load_uncacheable(Instruction &li) {
    assert(li.getParent()->getParent() == oldFunc);

    auto Arch = llvm::Triple(oldFunc->getParent()->getTargetTriple()).getArch();
    if (Arch == Triple::amdgcn &&
        cast<PointerType>(li.getOperand(0)->getType())->getAddressSpace() ==
            4) {
      return false;
    }

    if (EnzymeJuliaAddrLoad)
      if (auto PT = dyn_cast<PointerType>(li.getType()))
        if (PT->getAddressSpace() == 13)
          return false;

    // Only use invariant load data if either, we are not using Julia
    // or we are in combined mode. The reason for this is that Julia
    // incorrectly has invariant load info for a function, which specifies
    // the load value won't change over the course of a function, but
    // may change from a caller.
    bool checkFunction = true;
#if LLVM_VERSION_MAJOR >= 10
    if (li.hasMetadata(LLVMContext::MD_invariant_load))
#else
    if (li.getMetadata(LLVMContext::MD_invariant_load))
#endif
    {
      if (!EnzymeJuliaAddrLoad || mode == DerivativeMode::ReverseModeCombined)
        return false;
      else
        checkFunction = false;
    }

    // Find the underlying object for the pointer operand of the load
    // instruction.
    auto obj =
#if LLVM_VERSION_MAJOR >= 12
        getUnderlyingObject(li.getOperand(0), 100);
#else
        GetUnderlyingObject(li.getOperand(0),
                            oldFunc->getParent()->getDataLayout(), 100);
#endif

    // Openmp bound and local thread id are unchanging
    // definitionally cacheable.
    if (omp)
      if (auto arg = dyn_cast<Argument>(obj)) {
        if (arg->getArgNo() < 2) {
          return false;
        }
      }

    // Any load from a rematerializable allocation is definitionally
    // reloadable. Notably we don't need to perform the allFollowers
    // of check as the loop scope caching should allow us to ignore
    // such stores.
    if (rematerializableAllocations.count(obj))
      return false;

    // If not running combined, check if pointer operand is overwritten
    // by a subsequent call (i.e. not this function).
    bool can_modref = false;
    if (mode != DerivativeMode::ReverseModeCombined)
      can_modref = is_value_mustcache_from_origin(obj);

    if (!can_modref && checkFunction) {
      allFollowersOf(&li, [&](Instruction *inst2) {
        if (!inst2->mayWriteToMemory())
          return false;

        if (unnecessaryInstructions.count(inst2)) {
          return false;
        }
        if (auto CI = dyn_cast<CallInst>(inst2)) {
          if (auto F = CI->getCalledFunction()) {
            if (F->getName() == "__kmpc_for_static_fini") {
              return false;
            }
          }
        }

        if (!overwritesToMemoryReadBy(AA, SE, OrigLI, OrigDT, &li, inst2)) {
          return false;
        }

        if (auto II = dyn_cast<IntrinsicInst>(inst2)) {
          if (II->getIntrinsicID() == Intrinsic::nvvm_barrier0 ||
              II->getIntrinsicID() == Intrinsic::amdgcn_s_barrier) {
            allUnsyncdPredecessorsOf(
                II,
                [&](Instruction *mid) {
                  if (!mid->mayWriteToMemory())
                    return false;

                  if (unnecessaryInstructions.count(mid)) {
                    return false;
                  }

                  if (!writesToMemoryReadBy(AA, &li, mid)) {
                    return false;
                  }

                  can_modref = true;
                  EmitWarning("Uncacheable", li.getDebugLoc(), oldFunc,
                              li.getParent(), "Load may need caching ", li,
                              " due to ", *mid, " via ", *II);
                  return true;
                },
                [&]() {
                  // if gone past entry
                  if (mode != DerivativeMode::ReverseModeCombined) {
                    EmitWarning("Uncacheable", li.getDebugLoc(), oldFunc,
                                li.getParent(), "Load may need caching ", li,
                                " due to entry via ", *II);
                    can_modref = true;
                  }
                });
            if (can_modref)
              return true;
            else
              return false;
          }
        }
        can_modref = true;
        EmitWarning("Uncacheable", li.getDebugLoc(), oldFunc, li.getParent(),
                    "Load may need caching ", li, " due to ", *inst2);
        // Early exit
        return true;
      });
    } else {

      EmitWarning("Uncacheable", li.getDebugLoc(), oldFunc, li.getParent(),
                  "Load may need caching ", li, " due to origin ", *obj);
    }

    return can_modref;
  }

  // Computes a map of LoadInst -> boolean for a function indicating whether
  // that load is "uncacheable".
  //   A load is considered "uncacheable" if the data at the loaded memory
  //   location can be modified after the load instruction.
  std::map<Instruction *, bool> compute_uncacheable_load_map() {
    std::map<Instruction *, bool> can_modref_map;
    for (inst_iterator I = inst_begin(*oldFunc), E = inst_end(*oldFunc); I != E;
         ++I) {
      Instruction *inst = &*I;
      // For each load instruction, determine if it is uncacheable.
      if (auto op = dyn_cast<LoadInst>(inst)) {
        can_modref_map[inst] = is_load_uncacheable(*op);
      }
      if (auto II = dyn_cast<IntrinsicInst>(inst)) {
        switch (II->getIntrinsicID()) {
        case Intrinsic::nvvm_ldu_global_i:
        case Intrinsic::nvvm_ldu_global_p:
        case Intrinsic::nvvm_ldu_global_f:
        case Intrinsic::nvvm_ldg_global_i:
        case Intrinsic::nvvm_ldg_global_p:
        case Intrinsic::nvvm_ldg_global_f:
          can_modref_map[inst] = false;
          break;
        case Intrinsic::masked_load:
          can_modref_map[inst] = is_load_uncacheable(*II);
          break;
        default:
          break;
        }
      }
    }
    return can_modref_map;
  }

  std::map<Argument *, bool>
  compute_uncacheable_args_for_one_callsite(CallInst *callsite_op) {
    Function *Fn = getFunctionFromCall(callsite_op);

    if (!Fn)
      return {};

    if (isMemFreeLibMFunction(Fn->getName())) {
      return {};
    }

    if (isCertainPrintMallocOrFree(Fn) || isAllocationFunction(*Fn, TLI)) {
      return {};
    }

    StringRef funcName = Fn->getName();

    if (funcName.startswith("MPI_") || funcName.startswith("enzyme_wrapmpi$$"))
      return {};

    if (funcName == "__kmpc_for_static_init_4" ||
        funcName == "__kmpc_for_static_init_4u" ||
        funcName == "__kmpc_for_static_init_8" ||
        funcName == "__kmpc_for_static_init_8u") {
      return {};
    }

    SmallVector<Value *, 4> args;
    SmallVector<bool, 4> args_safe;

    // First, we need to propagate the uncacheable status from the parent
    // function to the callee.
    //   because memory location x modified after parent returns => x modified
    //   after callee returns.
#if LLVM_VERSION_MAJOR >= 14
    for (unsigned i = 0; i < callsite_op->arg_size(); ++i)
#else
    for (unsigned i = 0; i < callsite_op->getNumArgOperands(); ++i)
#endif
    {
      args.push_back(callsite_op->getArgOperand(i));

// If the UnderlyingObject is from one of this function's arguments, then we
// need to propagate the volatility.
#if LLVM_VERSION_MAJOR >= 12
      Value *obj = getUnderlyingObject(callsite_op->getArgOperand(i), 100);
#else
      Value *obj = GetUnderlyingObject(
          callsite_op->getArgOperand(i),
          callsite_op->getParent()->getModule()->getDataLayout(), 100);
#endif

      bool init_safe = !is_value_mustcache_from_origin(obj);
      if (!init_safe) {
        auto CD = TR.query(obj)[{-1}];
        if (CD == BaseType::Integer || CD.isFloat())
          init_safe = true;
      }
      if (!init_safe && !isa<UndefValue>(obj) && !isa<ConstantInt>(obj) &&
          !isa<Function>(obj)) {
        EmitWarning("UncacheableOrigin", callsite_op->getDebugLoc(), oldFunc,
                    callsite_op->getParent(), "Callsite ", *callsite_op,
                    " arg ", i, " ", *callsite_op->getArgOperand(i),
                    " uncacheable from origin ", *obj);
      }
      args_safe.push_back(init_safe);
    }

    // Second, we check for memory modifications that can occur in the
    // continuation of the
    //   callee inside the parent function.
    allFollowersOf(callsite_op, [&](Instruction *inst2) {
      // Don't consider modref from malloc/free as a need to cache
      if (auto obj_op = dyn_cast<CallInst>(inst2)) {
        Function *called = getFunctionFromCall(obj_op);
        if (called && isCertainPrintMallocOrFree(called)) {
          return false;
        }
        if (called && isMemFreeLibMFunction(called->getName())) {
          return false;
        }
        if (called && called->getName() == "__kmpc_for_static_fini") {
          return false;
        }

#if LLVM_VERSION_MAJOR >= 11
        if (auto iasm = dyn_cast<InlineAsm>(obj_op->getCalledOperand()))
#else
        if (auto iasm = dyn_cast<InlineAsm>(obj_op->getCalledValue()))
#endif
        {
          if (StringRef(iasm->getAsmString()).contains("exit"))
            return false;
        }
      }

      if (unnecessaryInstructions.count(inst2))
        return false;

      if (!inst2->mayWriteToMemory())
        return false;

      for (unsigned i = 0; i < args.size(); ++i) {
        if (!args_safe[i])
          continue;
        auto CD = TR.query(args[i])[{-1}];
        if (CD == BaseType::Integer || CD.isFloat())
          continue;

        if (llvm::isModSet(AA.getModRefInfo(
                inst2, MemoryLocation::getForArgument(callsite_op, i, TLI)))) {
          if (!isa<ConstantInt>(callsite_op->getArgOperand(i)) &&
              !isa<UndefValue>(callsite_op->getArgOperand(i)))
            EmitWarning("UncacheableArg", callsite_op->getDebugLoc(), oldFunc,
                        callsite_op->getParent(), "Callsite ", *callsite_op,
                        " arg ", i, " ", *callsite_op->getArgOperand(i),
                        " uncacheable due to ", *inst2);
          args_safe[i] = false;
        }
      }
      return false;
    });

    std::map<Argument *, bool> uncacheable_args;

    if (funcName == "__kmpc_fork_call") {
      Value *op = callsite_op->getArgOperand(2);
      Function *task = nullptr;
      while (!(task = dyn_cast<Function>(op))) {
        if (auto castinst = dyn_cast<ConstantExpr>(op))
          if (castinst->isCast()) {
            op = castinst->getOperand(0);
            continue;
          }
        if (auto CI = dyn_cast<CastInst>(op)) {
          op = CI->getOperand(0);
          continue;
        }
        llvm::errs() << "op: " << *op << "\n";
        assert(0 && "unknown fork call arg");
      }

      auto arg = task->arg_begin();

      // Global.tid is cacheable
      uncacheable_args[arg] = false;
      ++arg;
      // Bound.tid is cacheable
      uncacheable_args[arg] = false;
      ++arg;

      // Ignore first three arguments of fork call
      for (unsigned i = 3; i < args.size(); ++i) {
        uncacheable_args[arg] = !args_safe[i];
        ++arg;
        if (arg == Fn->arg_end()) {
          break;
        }
      }
    } else {
      auto arg = Fn->arg_begin();
      for (unsigned i = 0; i < args.size(); ++i) {
        uncacheable_args[arg] = !args_safe[i];
        ++arg;
        if (arg == Fn->arg_end()) {
          break;
        }
      }
    }

    return uncacheable_args;
  }

  // Given a function and the arguments passed to it by its caller that are
  // uncacheable (_uncacheable_args) compute
  //   the set of uncacheable arguments for each callsite inside the function. A
  //   pointer argument is uncacheable at a callsite if the memory pointed to
  //   might be modified after that callsite.
  std::map<CallInst *, const std::map<Argument *, bool>>
  compute_uncacheable_args_for_callsites() {
    std::map<CallInst *, const std::map<Argument *, bool>> uncacheable_args_map;

    for (inst_iterator I = inst_begin(*oldFunc), E = inst_end(*oldFunc); I != E;
         ++I) {
      Instruction &inst = *I;
      if (auto op = dyn_cast<CallInst>(&inst)) {

        // We do not need uncacheable args for intrinsic functions. So skip such
        // callsites.
        if (auto II = dyn_cast<IntrinsicInst>(&inst)) {
          if (!II->getCalledFunction()->getName().startswith("llvm.julia"))
            continue;
        }

        // For all other calls, we compute the uncacheable args for this
        // callsite.
        uncacheable_args_map.insert(
            std::pair<CallInst *, const std::map<Argument *, bool>>(
                op, compute_uncacheable_args_for_one_callsite(op)));
      }
    }
    return uncacheable_args_map;
  }
};

void calculateUnusedValuesInFunction(
    Function &func, llvm::SmallPtrSetImpl<const Value *> &unnecessaryValues,
    llvm::SmallPtrSetImpl<const Instruction *> &unnecessaryInstructions,
    bool returnValue, DerivativeMode mode, GradientUtils *gutils,
    TargetLibraryInfo &TLI, ArrayRef<DIFFE_TYPE> constant_args,
    const llvm::SmallPtrSetImpl<BasicBlock *> &oldUnreachable) {
  std::map<UsageKey, bool> CacheResults;
  for (auto pair : gutils->knownRecomputeHeuristic) {
    if (!pair.second) {
      CacheResults[UsageKey(pair.first, ValueType::Primal)] = false;
    }
  }
  std::map<UsageKey, bool> PrimalSeen;
  if (mode == DerivativeMode::ReverseModeGradient) {
    PrimalSeen = CacheResults;
  }

  for (const auto &pair : gutils->allocationsWithGuaranteedFree) {
    if (pair.second.size() == 0)
      continue;

    bool primalNeededInReverse = is_value_needed_in_reverse<ValueType::Primal>(
        gutils, pair.first, mode, CacheResults, oldUnreachable);
    bool cacheWholeAllocation = false;

    if (gutils->knownRecomputeHeuristic.count(pair.first)) {
      if (!gutils->knownRecomputeHeuristic[pair.first]) {
        primalNeededInReverse = true;
        cacheWholeAllocation = true;
      }
    }
    // If rematerializing a loop-level allocation, the primal allocation
    // is not needed in the reverse.
    if (!cacheWholeAllocation && primalNeededInReverse) {
      auto found = gutils->rematerializableAllocations.find(
          const_cast<CallInst *>(pair.first));
      if (found != gutils->rematerializableAllocations.end()) {
        if (auto inst = dyn_cast<Instruction>(pair.first))
          if (found->second.LI &&
              found->second.LI->contains(inst->getParent())) {
            primalNeededInReverse = false;
          }
      }
    }

    for (auto freeCall : pair.second) {
      if (!primalNeededInReverse)
        gutils->forwardDeallocations.insert(freeCall);
      else
        gutils->postDominatingFrees.insert(freeCall);
    }
  }

  calculateUnusedValues(
      func, unnecessaryValues, unnecessaryInstructions, returnValue,
      [&](const Value *val) {
        bool ivn = is_value_needed_in_reverse<ValueType::Primal>(
            gutils, val, mode, PrimalSeen, oldUnreachable);
        return ivn;
      },
      [&](const Instruction *inst) {
        if (auto II = dyn_cast<IntrinsicInst>(inst)) {
          if (II->getIntrinsicID() == Intrinsic::lifetime_start ||
              II->getIntrinsicID() == Intrinsic::lifetime_end ||
              II->getIntrinsicID() == Intrinsic::stacksave ||
              II->getIntrinsicID() == Intrinsic::stackrestore) {
            return UseReq::Cached;
          }
        }

        if (mode == DerivativeMode::ReverseModeGradient &&
            gutils->knownRecomputeHeuristic.find(inst) !=
                gutils->knownRecomputeHeuristic.end()) {
          if (!gutils->knownRecomputeHeuristic[inst]) {
            return UseReq::Cached;
          }
        }

        if (llvm::isa<llvm::ReturnInst>(inst) && returnValue) {
          return UseReq::Need;
        }
        if (llvm::isa<llvm::BranchInst>(inst) ||
            llvm::isa<llvm::SwitchInst>(inst)) {
          size_t num = 0;
          for (auto suc : successors(inst->getParent())) {
            if (!oldUnreachable.count(suc)) {
              num++;
            }
          }
          if (num > 1 || mode != DerivativeMode::ReverseModeGradient) {
            return UseReq::Need;
          }
        }

        // We still need this value if used as increment/induction variable for
        // a loop
        // TODO this really should be more simply replaced by doing the loop
        // normalization on the original code as preprocessing

        // Below we specifically check if the instructions or any of its
        // newly-generated (e.g. not in original function) uses are used in the
        // loop calculation
        auto NewI = gutils->getNewFromOriginal(inst);
        std::set<Instruction *> todo = {NewI};
        {
          std::deque<Instruction *> toAnalyze;
          // Here we get the uses of inst from the original function
          std::set<Instruction *> UsesFromOrig;
          for (auto u : inst->users()) {
            if (auto I = dyn_cast<Instruction>(u)) {
              UsesFromOrig.insert(gutils->getNewFromOriginal(I));
            }
          }
          // We only analyze uses that were not available in the original
          // function
          for (auto u : NewI->users()) {
            if (auto I = dyn_cast<Instruction>(u)) {
              if (UsesFromOrig.count(I) == 0)
                toAnalyze.push_back(I);
            }
          }

          while (toAnalyze.size()) {
            auto Next = toAnalyze.front();
            toAnalyze.pop_front();
            if (todo.count(Next))
              continue;
            todo.insert(Next);
            for (auto u : Next->users()) {
              if (auto I = dyn_cast<Instruction>(u)) {
                toAnalyze.push_back(I);
              }
            }
          }
        }

        for (auto I : todo) {
          if (gutils->isInstructionUsedInLoopInduction(*I)) {
            return UseReq::Need;
          }
        }

        bool isLibMFn = false;
        if (auto obj_op = dyn_cast<CallInst>(inst)) {
          Function *called = getFunctionFromCall((CallInst *)obj_op);
          if (called && isDeallocationFunction(*called, TLI)) {
            if (mode == DerivativeMode::ForwardMode ||
                mode == DerivativeMode::ForwardModeSplit ||
                ((mode == DerivativeMode::ReverseModePrimal ||
                  mode == DerivativeMode::ReverseModeCombined) &&
                 gutils->forwardDeallocations.count(obj_op)))
              return UseReq::Need;
            return UseReq::Recur;
          }
          Intrinsic::ID ID = Intrinsic::not_intrinsic;
          if (called && isMemFreeLibMFunction(called->getName(), &ID)) {
            isLibMFn = true;
          }
        }

        if (auto si = dyn_cast<StoreInst>(inst)) {
          if (isa<UndefValue>(si->getValueOperand()))
            return UseReq::Recur;
#if LLVM_VERSION_MAJOR >= 12
          auto at = getUnderlyingObject(si->getPointerOperand(), 100);
#else
          auto at = GetUnderlyingObject(
              si->getPointerOperand(),
              gutils->oldFunc->getParent()->getDataLayout(), 100);
#endif
          if (auto arg = dyn_cast<Argument>(at)) {
            if (constant_args[arg->getArgNo()] == DIFFE_TYPE::DUP_NONEED) {
              return UseReq::Recur;
            }
          }
        }

        if (auto mti = dyn_cast<MemTransferInst>(inst)) {
#if LLVM_VERSION_MAJOR >= 12
          auto at = getUnderlyingObject(mti->getArgOperand(1), 100);
#else
          auto at = GetUnderlyingObject(
              mti->getArgOperand(1),
              gutils->oldFunc->getParent()->getDataLayout(), 100);
#endif
          if (auto arg = dyn_cast<Argument>(at)) {
            if (constant_args[arg->getArgNo()] == DIFFE_TYPE::DUP_NONEED) {
              return UseReq::Recur;
            }
          }
          bool newMemory = false;
          if (isa<AllocaInst>(at))
            newMemory = true;
          else if (auto CI = dyn_cast<CallInst>(at))
            if (Function *F = getFunctionFromCall(CI))
              if (isAllocationFunction(*F, TLI))
                newMemory = true;
          if (newMemory) {
            bool foundStore = false;
            allInstructionsBetween(
                gutils->OrigLI, cast<Instruction>(at),
                const_cast<MemTransferInst *>(mti),
                [&](Instruction *I) -> bool {
                  if (!I->mayWriteToMemory())
                    return /*earlyBreak*/ false;
                  if (unnecessaryInstructions.count(I))
                    return /*earlyBreak*/ false;

                  if (writesToMemoryReadBy(
                          gutils->OrigAA,
                          /*maybeReader*/ const_cast<MemTransferInst *>(mti),
                          /*maybeWriter*/ I)) {
                    foundStore = true;
                    return /*earlyBreak*/ true;
                  }
                  return /*earlyBreak*/ false;
                });
            if (!foundStore) {
              return UseReq::Recur;
            }
          }
        }
        if ((mode == DerivativeMode::ReverseModePrimal ||
             mode == DerivativeMode::ReverseModeCombined ||
             mode == DerivativeMode::ForwardMode) &&
            inst->mayWriteToMemory() && !isLibMFn) {
          return UseReq::Need;
        }
        // Don't erase any store that needs to be preserved for a
        // rematerialization. However, if not used in a rematerialization, the
        // store should be eliminated in the reverse pass
        if (mode == DerivativeMode::ReverseModeGradient ||
            mode == DerivativeMode::ForwardModeSplit) {
          auto CI = dyn_cast<CallInst>(const_cast<Instruction *>(inst));
          Function *CF = CI ? getFunctionFromCall(CI) : nullptr;
          StringRef funcName = CF ? CF->getName() : "";
          if (isa<MemTransferInst>(inst) || isa<StoreInst>(inst) ||
              isa<MemSetInst>(inst) || funcName == "julia.write_barrier") {
            for (auto pair : gutils->rematerializableAllocations) {
              if (pair.second.stores.count(inst)) {
                if (is_value_needed_in_reverse<ValueType::Primal>(
                        gutils, pair.first, mode, PrimalSeen, oldUnreachable)) {
                  return UseReq::Need;
                }
              }
            }
            return UseReq::Recur;
          }
        }
        bool ivn = is_value_needed_in_reverse<ValueType::Primal>(
            gutils, inst, mode, PrimalSeen, oldUnreachable);
        if (ivn) {
          return UseReq::Need;
        }
        return UseReq::Recur;
      });

  if (EnzymePrintUnnecessary) {
    llvm::errs() << " val use analysis of " << func.getName()
                 << ": mode=" << to_string(mode) << "\n";
    for (auto &BB : func)
      for (auto &I : BB) {
        bool ivn = is_value_needed_in_reverse<ValueType::Primal>(
            gutils, &I, mode, PrimalSeen, oldUnreachable);
        bool isn = is_value_needed_in_reverse<ValueType::Shadow>(
            gutils, &I, mode, PrimalSeen, oldUnreachable);
        llvm::errs() << I << " ivn=" << (int)ivn << " isn: " << (int)isn;
        auto found = gutils->knownRecomputeHeuristic.find(&I);
        if (found != gutils->knownRecomputeHeuristic.end()) {
          llvm::errs() << " krc=" << (int)found->second;
        }
        llvm::errs() << "\n";
      }
    llvm::errs() << "unnecessaryValues of " << func.getName()
                 << ": mode=" << to_string(mode) << "\n";
    for (auto a : unnecessaryValues) {
      bool ivn = is_value_needed_in_reverse<ValueType::Primal>(
          gutils, a, mode, PrimalSeen, oldUnreachable);
      bool isn = is_value_needed_in_reverse<ValueType::Shadow>(
          gutils, a, mode, PrimalSeen, oldUnreachable);
      llvm::errs() << *a << " ivn=" << (int)ivn << " isn: " << (int)isn;
      auto found = gutils->knownRecomputeHeuristic.find(a);
      if (found != gutils->knownRecomputeHeuristic.end()) {
        llvm::errs() << " krc=" << (int)found->second;
      }
      llvm::errs() << "\n";
    }
    llvm::errs() << "unnecessaryInstructions " << func.getName() << ":\n";
    for (auto a : unnecessaryInstructions) {
      llvm::errs() << *a << "\n";
    }
  }
}

void calculateUnusedStoresInFunction(
    Function &func,
    llvm::SmallPtrSetImpl<const Instruction *> &unnecessaryStores,
    const llvm::SmallPtrSetImpl<const Instruction *> &unnecessaryInstructions,
    GradientUtils *gutils, TargetLibraryInfo &TLI) {
  calculateUnusedStores(func, unnecessaryStores, [&](const Instruction *inst) {
    if (auto si = dyn_cast<StoreInst>(inst)) {
      if (isa<UndefValue>(si->getValueOperand()))
        return false;
    }

    if (auto mti = dyn_cast<MemTransferInst>(inst)) {
#if LLVM_VERSION_MAJOR >= 12
      auto at = getUnderlyingObject(mti->getArgOperand(1), 100);
#else
      auto at = GetUnderlyingObject(
          mti->getArgOperand(1),
          func.getParent()->getDataLayout(), 100);
#endif
      bool newMemory = false;
      if (isa<AllocaInst>(at))
        newMemory = true;
      else if (auto CI = dyn_cast<CallInst>(at))
        if (Function *F = getFunctionFromCall(CI))
          if (isAllocationFunction(*F, TLI))
            newMemory = true;
      if (newMemory) {
        bool foundStore = false;
        allInstructionsBetween(
            gutils->OrigLI, cast<Instruction>(at),
            const_cast<MemTransferInst *>(mti), [&](Instruction *I) -> bool {
              if (!I->mayWriteToMemory())
                return /*earlyBreak*/ false;
              if (unnecessaryInstructions.count(I))
                return /*earlyBreak*/ false;

              // if (I == &MTI) return;
              if (writesToMemoryReadBy(
                      gutils->OrigAA,
                      /*maybeReader*/ const_cast<MemTransferInst *>(mti),
                      /*maybeWriter*/ I)) {
                foundStore = true;
                return /*earlyBreak*/ true;
              }
              return /*earlyBreak*/ false;
            });
        if (!foundStore) {
          // performing a memcpy out of unitialized memory
          return false;
        }
      }
    }

    return true;
  });
}

std::string to_string(const std::map<Argument *, bool> &us) {
  std::string s = "{";
  for (auto y : us)
    s += y.first->getName().str() + "@" +
         y.first->getParent()->getName().str() + ":" +
         std::to_string(y.second) + ",";
  return s + "}";
}

//! assuming not top level
std::pair<SmallVector<Type *, 4>, SmallVector<Type *, 4>>
getDefaultFunctionTypeForAugmentation(FunctionType *called, bool returnUsed,
                                      DIFFE_TYPE retType) {
  SmallVector<Type *, 4> args;
  SmallVector<Type *, 4> outs;
  for (auto &argType : called->params()) {
    args.push_back(argType);

    if (!argType->isFPOrFPVectorTy()) {
      args.push_back(argType);
    }
  }

  auto ret = called->getReturnType();
  // TODO CONSIDER a.getType()->isIntegerTy() &&
  // cast<IntegerType>(a.getType())->getBitWidth() < 16
  outs.push_back(Type::getInt8PtrTy(called->getContext()));
  if (!ret->isVoidTy() && !ret->isEmptyTy()) {
    if (returnUsed) {
      outs.push_back(ret);
    }
    if (retType == DIFFE_TYPE::DUP_ARG || retType == DIFFE_TYPE::DUP_NONEED) {
      outs.push_back(ret);
    }
  }

  return std::pair<SmallVector<Type *, 4>, SmallVector<Type *, 4>>(args, outs);
}

//! assuming not top level
std::pair<SmallVector<Type *, 4>, SmallVector<Type *, 4>>
getDefaultFunctionTypeForGradient(FunctionType *called, DIFFE_TYPE retType) {
  SmallVector<Type *, 4> args;
  SmallVector<Type *, 4> outs;
  // TODO CONSIDER a.getType()->isIntegerTy() &&
  // cast<IntegerType>(a.getType())->getBitWidth() < 16

  for (auto &argType : called->params()) {
    args.push_back(argType);

    if (!argType->isFPOrFPVectorTy()) {
      args.push_back(argType);
    } else {
      outs.push_back(argType);
    }
  }

  auto ret = called->getReturnType();

  if (retType == DIFFE_TYPE::OUT_DIFF) {
    args.push_back(ret);
  }

  return std::pair<SmallVector<Type *, 4>, SmallVector<Type *, 4>>(args, outs);
}

bool shouldAugmentCall(CallInst *op, const GradientUtils *gutils) {
  assert(op->getParent()->getParent() == gutils->oldFunc);

  Function *called = op->getCalledFunction();

  bool modifyPrimal = !called || !called->hasFnAttribute(Attribute::ReadNone);

  if (modifyPrimal) {
#ifdef PRINT_AUGCALL
    if (called)
      llvm::errs() << "primal modified " << called->getName()
                   << " modified via reading from memory"
                   << "\n";
    else
      llvm::errs() << "primal modified " << *op->getCalledValue()
                   << " modified via reading from memory"
                   << "\n";
#endif
  }

  if (!op->getType()->isFPOrFPVectorTy() && !gutils->isConstantValue(op) &&
      gutils->TR.query(op).Inner0().isPossiblePointer()) {
    modifyPrimal = true;

#ifdef PRINT_AUGCALL
    if (called)
      llvm::errs() << "primal modified " << called->getName()
                   << " modified via return"
                   << "\n";
    else
      llvm::errs() << "primal modified " << *op->getCalledValue()
                   << " modified via return"
                   << "\n";
#endif
  }

  if (!called || called->empty())
    modifyPrimal = true;

#if LLVM_VERSION_MAJOR >= 14
  for (unsigned i = 0; i < op->arg_size(); ++i)
#else
  for (unsigned i = 0; i < op->getNumArgOperands(); ++i)
#endif
  {
    if (gutils->isConstantValue(op->getArgOperand(i)) && called &&
        !called->empty()) {
      continue;
    }

    auto argType = op->getArgOperand(i)->getType();

    if (!argType->isFPOrFPVectorTy() &&
        !gutils->isConstantValue(op->getArgOperand(i)) &&
        gutils->TR.query(op->getArgOperand(i)).Inner0().isPossiblePointer()) {
      if (called && !(called->hasParamAttribute(i, Attribute::ReadOnly) ||
                      called->hasParamAttribute(i, Attribute::ReadNone))) {
        modifyPrimal = true;
#ifdef PRINT_AUGCALL
        if (called)
          llvm::errs() << "primal modified " << called->getName()
                       << " modified via arg " << i << "\n";
        else
          llvm::errs() << "primal modified " << *op->getCalledValue()
                       << " modified via arg " << i << "\n";
#endif
      }
    }
  }

  // Don't need to augment calls that are certain to not hit return
  if (isa<UnreachableInst>(op->getParent()->getTerminator())) {
    modifyPrimal = false;
  }

#ifdef PRINT_AUGCALL
  llvm::errs() << "PM: " << *op << " modifyPrimal: " << modifyPrimal
               << " cv: " << gutils->isConstantValue(op) << "\n";
#endif
  return modifyPrimal;
}

static inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                            ModRefInfo mri) {
  if (mri == ModRefInfo::NoModRef)
    return os << "nomodref";
  else if (mri == ModRefInfo::ModRef)
    return os << "modref";
  else if (mri == ModRefInfo::Mod)
    return os << "mod";
  else if (mri == ModRefInfo::Ref)
    return os << "ref";
  else if (mri == ModRefInfo::MustModRef)
    return os << "mustmodref";
  else if (mri == ModRefInfo::MustMod)
    return os << "mustmod";
  else if (mri == ModRefInfo::MustRef)
    return os << "mustref";
  else
    llvm_unreachable("unknown modref");
  return os;
}

bool legalCombinedForwardReverse(
    CallInst *origop,
    const std::map<ReturnInst *, StoreInst *> &replacedReturns,
    SmallVectorImpl<Instruction *> &postCreate,
    SmallVectorImpl<Instruction *> &userReplace, GradientUtils *gutils,
    const SmallPtrSetImpl<const Instruction *> &unnecessaryInstructions,
    const SmallPtrSetImpl<BasicBlock *> &oldUnreachable,
    const bool subretused) {
  Function *called = origop->getCalledFunction();
#if LLVM_VERSION_MAJOR >= 11
  Value *calledValue = origop->getCalledOperand();
#else
  Value *calledValue = origop->getCalledValue();
#endif

  if (isa<PointerType>(origop->getType())) {
    bool sret = subretused;
    if (!sret && !gutils->isConstantValue(origop)) {
      sret = is_value_needed_in_reverse<ValueType::Shadow>(
          gutils, origop, gutils->mode, oldUnreachable);
    }

    if (sret) {
      if (EnzymePrintPerf) {
        if (called)
          llvm::errs() << " [not implemented] pointer return for combined "
                          "forward/reverse "
                       << called->getName() << "\n";
        else
          llvm::errs() << " [not implemented] pointer return for combined "
                          "forward/reverse "
                       << *calledValue << "\n";
      }
      return false;
    }
  }

  // Check any users of the returned value and determine all values that would
  // be needed to be moved to reverse pass
  //  to ensure the forward pass would remain correct and everything computable
  SmallPtrSet<Instruction *, 4> usetree;
  std::deque<Instruction *> todo{origop};

  bool legal = true;

  // Given a function I we know must be moved to the reverse for legality
  // reasons
  auto propagate = [&](Instruction *I) {
    // if only used in unneeded return, don't need to move this to reverse
    // (unless this is the original function)
    if (usetree.count(I))
      return;
    if (gutils->notForAnalysis.count(I->getParent()))
      return;
    if (auto ri = dyn_cast<ReturnInst>(I)) {
      auto find = replacedReturns.find(ri);
      if (find != replacedReturns.end()) {
        usetree.insert(ri);
      }
      return;
    }

    if (isa<BranchInst>(I) || isa<SwitchInst>(I)) {
      legal = false;
      if (EnzymePrintPerf) {
        if (called)
          llvm::errs() << " [bi] failed to replace function "
                       << (called->getName()) << " due to " << *I << "\n";
        else
          llvm::errs() << " [bi] failed to replace function " << (*calledValue)
                       << " due to " << *I << "\n";
      }
      return;
    }

    // Even though there is a dependency on this value here, we can ignore it if
    // it isn't going to be used Unless this is a call that could have a
    // combined forward-reverse
    if (I != origop && unnecessaryInstructions.count(I)) {
      if (gutils->isConstantInstruction(I) || !isa<CallInst>(I)) {
        userReplace.push_back(I);
        return;
      }
    }

    if (auto op = dyn_cast<CallInst>(I)) {
      Function *called = getFunctionFromCall(op);
      if (called && (isAllocationFunction(*called, gutils->TLI) ||
                     isDeallocationFunction(*called, gutils->TLI)))
        return;
    }

    if (isa<BranchInst>(I)) {
      legal = false;

      return;
    }
    if (isa<PHINode>(I)) {
      legal = false;
      if (EnzymePrintPerf) {
        if (called)
          llvm::errs() << " [phi] failed to replace function "
                       << (called->getName()) << " due to " << *I << "\n";
        else
          llvm::errs() << " [phi] failed to replace function " << (*calledValue)
                       << " due to " << *I << "\n";
      }
      return;
    }
    if (is_value_needed_in_reverse<ValueType::Primal>(
            gutils, I, DerivativeMode::ReverseModeCombined, oldUnreachable)) {
      legal = false;
      if (EnzymePrintPerf) {
        if (called)
          llvm::errs() << " [nv] failed to replace function "
                       << (called->getName()) << " due to " << *I << "\n";
        else
          llvm::errs() << " [nv] failed to replace function " << (*calledValue)
                       << " due to " << *I << "\n";
      }
      return;
    }
    if (I != origop && !isa<IntrinsicInst>(I) && isa<CallInst>(I)) {
      legal = false;
      if (EnzymePrintPerf) {
        if (called)
          llvm::errs() << " [ci] failed to replace function "
                       << (called->getName()) << " due to " << *I << "\n";
        else
          llvm::errs() << " [ci] failed to replace function " << (*calledValue)
                       << " due to " << *I << "\n";
      }
      return;
    }
    // Do not try moving an instruction that modifies memory, if we already
    // moved it
    if (!isa<StoreInst>(I) || unnecessaryInstructions.count(I) == 0)
      if (I->mayReadOrWriteMemory() &&
          gutils->getNewFromOriginal(I)->getParent() !=
              gutils->getNewFromOriginal(I->getParent())) {
        legal = false;
        if (EnzymePrintPerf) {
          if (called)
            llvm::errs() << " [am] failed to replace function "
                         << (called->getName()) << " due to " << *I << "\n";
          else
            llvm::errs() << " [am] failed to replace function "
                         << (*calledValue) << " due to " << *I << "\n";
        }
        return;
      }

    usetree.insert(I);
    for (auto use : I->users()) {
      todo.push_back(cast<Instruction>(use));
    }
  };

  while (!todo.empty()) {
    auto inst = todo.front();
    todo.pop_front();

    if (inst->mayWriteToMemory()) {
      auto consider = [&](Instruction *user) {
        if (!user->mayReadFromMemory())
          return false;
        if (writesToMemoryReadBy(gutils->OrigAA, /*maybeReader*/ user,
                                 /*maybeWriter*/ inst)) {
          propagate(user);
          // Fast return if not legal
          if (!legal)
            return true;
        }
        return false;
      };
      allFollowersOf(inst, consider);
      if (!legal)
        return false;
    }

    propagate(inst);
    if (!legal)
      return false;
  }

  // Check if any of the unmoved operations will make it illegal to move the
  // instruction

  for (auto inst : usetree) {
    if (!inst->mayReadFromMemory())
      continue;
    allFollowersOf(inst, [&](Instruction *post) {
      if (unnecessaryInstructions.count(post))
        return false;
      if (!post->mayWriteToMemory())
        return false;
      if (writesToMemoryReadBy(gutils->OrigAA, /*maybeReader*/ inst,
                               /*maybeWriter*/ post)) {
        if (EnzymePrintPerf) {
          if (called)
            llvm::errs() << " failed to replace function "
                         << (called->getName()) << " due to " << *post
                         << " usetree: " << *inst << "\n";
          else
            llvm::errs() << " failed to replace function " << (*calledValue)
                         << " due to " << *post << " usetree: " << *inst
                         << "\n";
        }
        legal = false;
        return true;
      }
      return false;
    });
    if (!legal)
      break;
  }

  if (!legal)
    return false;

  allFollowersOf(origop, [&](Instruction *inst) {
    if (auto ri = dyn_cast<ReturnInst>(inst)) {
      auto find = replacedReturns.find(ri);
      if (find != replacedReturns.end()) {
        postCreate.push_back(find->second);
        return false;
      }
    }

    if (usetree.count(inst) == 0)
      return false;
    if (inst->getParent() != origop->getParent()) {
      // Don't move a writing instruction (may change speculatable/etc things)
      if (inst->mayWriteToMemory()) {
        if (EnzymePrintPerf) {
          if (called)
            llvm::errs() << " [nonspec] failed to replace function "
                         << (called->getName()) << " due to " << *inst << "\n";
          else
            llvm::errs() << " [nonspec] failed to replace function "
                         << (*calledValue) << " due to " << *inst << "\n";
        }
        legal = false;
        // Early exit
        return true;
      }
    }
    if (isa<CallInst>(inst) &&
        gutils->originalToNewFn.find(inst) == gutils->originalToNewFn.end()) {
      legal = false;
      if (EnzymePrintPerf) {
        if (called)
          llvm::errs() << " [premove] failed to replace function "
                       << (called->getName()) << " due to " << *inst << "\n";
        else
          llvm::errs() << " [premove] failed to replace function "
                       << (*calledValue) << " due to " << *inst << "\n";
      }
      // Early exit
      return true;
    }
    postCreate.push_back(gutils->getNewFromOriginal(inst));
    return false;
  });

  if (!legal)
    return false;

  if (EnzymePrintPerf) {
    if (called)
      llvm::errs() << " choosing to replace function " << (called->getName())
                   << " and do both forward/reverse\n";
    else
      llvm::errs() << " choosing to replace function " << (*calledValue)
                   << " and do both forward/reverse\n";
  }

  return true;
}

void clearFunctionAttributes(Function *f) {
  for (Argument &Arg : f->args()) {
    if (Arg.hasAttribute(Attribute::Returned))
      Arg.removeAttr(Attribute::Returned);
    if (Arg.hasAttribute(Attribute::StructRet))
      Arg.removeAttr(Attribute::StructRet);
  }
  if (f->hasFnAttribute(Attribute::OptimizeNone))
    f->removeFnAttr(Attribute::OptimizeNone);

#if LLVM_VERSION_MAJOR >= 14
  if (f->getAttributes().getRetDereferenceableBytes())
#else
  if (f->getDereferenceableBytes(llvm::AttributeList::ReturnIndex))
#endif
  {
#if LLVM_VERSION_MAJOR >= 14
    f->removeRetAttr(Attribute::Dereferenceable);
#else
    f->removeAttribute(llvm::AttributeList::ReturnIndex,
                       Attribute::Dereferenceable);
#endif
  }

  if (f->getAttributes().getRetAlignment()) {
#if LLVM_VERSION_MAJOR >= 14
    f->removeRetAttr(Attribute::Alignment);
#else
    f->removeAttribute(llvm::AttributeList::ReturnIndex, Attribute::Alignment);
#endif
  }
  Attribute::AttrKind attrs[] = {
#if LLVM_VERSION_MAJOR >= 11
    Attribute::NoUndef,
#endif
    Attribute::NonNull,
    Attribute::ZExt,
    Attribute::NoAlias
  };
  for (auto attr : attrs) {
#if LLVM_VERSION_MAJOR >= 14
    if (f->hasRetAttribute(attr)) {
      f->removeRetAttr(attr);
    }
#else
    if (f->hasAttribute(llvm::AttributeList::ReturnIndex, attr)) {
      f->removeAttribute(llvm::AttributeList::ReturnIndex, attr);
    }
#endif
  }
}

void cleanupInversionAllocs(DiffeGradientUtils *gutils, BasicBlock *entry) {
  while (gutils->inversionAllocs->size() > 0) {
    Instruction *inst = &gutils->inversionAllocs->back();
    if (isa<AllocaInst>(inst))
      inst->moveBefore(&gutils->newFunc->getEntryBlock().front());
    else
      inst->moveBefore(entry->getFirstNonPHIOrDbgOrLifetime());
  }

  (IRBuilder<>(gutils->inversionAllocs)).CreateUnreachable();
  DeleteDeadBlock(gutils->inversionAllocs);
  for (auto BBs : gutils->reverseBlocks) {
    if (pred_begin(BBs.second.front()) == pred_end(BBs.second.front())) {
      (IRBuilder<>(BBs.second.front())).CreateUnreachable();
      DeleteDeadBlock(BBs.second.front());
    }
  }
}

static FnTypeInfo preventTypeAnalysisLoops(const FnTypeInfo &oldTypeInfo_,
                                           llvm::Function *todiff) {
  FnTypeInfo oldTypeInfo = oldTypeInfo_;
  for (auto &pair : oldTypeInfo.KnownValues) {
    if (pair.second.size() != 0) {
      bool recursiveUse = false;
      for (auto user : pair.first->users()) {
        if (auto bi = dyn_cast<BinaryOperator>(user)) {
          for (auto biuser : bi->users()) {
            if (auto ci = dyn_cast<CallInst>(biuser)) {
              if (ci->getCalledFunction() == todiff &&
                  ci->getArgOperand(pair.first->getArgNo()) == bi) {
                recursiveUse = true;
                break;
              }
            }
          }
        }
        if (recursiveUse)
          break;
      }
      if (recursiveUse) {
        pair.second.clear();
      }
    }
  }
  return oldTypeInfo;
}

void restoreCache(
    DiffeGradientUtils *gutils,
    const std::map<std::pair<Instruction *, CacheType>, int> &mapping,
    const SmallPtrSetImpl<BasicBlock *> &guaranteedUnreachable) {
  // One must use this temporary map to first create all the replacements
  // prior to actually replacing to ensure that getSubLimits has the same
  // behavior and unwrap behavior for all replacements.
  SmallVector<std::pair<Value *, Value *>, 4> newIToNextI;

  for (const auto &m : mapping) {
    if (m.first.second == CacheType::Self &&
        gutils->knownRecomputeHeuristic.count(m.first.first)) {
      assert(gutils->knownRecomputeHeuristic.count(m.first.first));
      if (!isa<CallInst>(m.first.first)) {
        auto newi = gutils->getNewFromOriginal(m.first.first);
        if (auto PN = dyn_cast<PHINode>(newi))
          if (gutils->fictiousPHIs.count(PN)) {
            assert(gutils->fictiousPHIs[PN] == m.first.first);
            gutils->fictiousPHIs.erase(PN);
          }
        IRBuilder<> BuilderZ(newi->getNextNode());
        if (isa<PHINode>(m.first.first)) {
          BuilderZ.SetInsertPoint(
              cast<Instruction>(newi)->getParent()->getFirstNonPHI());
        }
        Value *nexti =
            gutils->cacheForReverse(BuilderZ, newi, m.second,
                                    /*ignoreType*/ false, /*replace*/ false);
        newIToNextI.emplace_back(newi, nexti);
      } else {
        auto newi = gutils->getNewFromOriginal((Value *)m.first.first);
        newIToNextI.emplace_back(newi, newi);
      }
    }
  }

  std::map<Value *, SmallVector<Instruction *, 4>> unwrapToOrig;
  for (auto pair : gutils->unwrappedLoads)
    unwrapToOrig[pair.second].push_back(const_cast<Instruction *>(pair.first));
  gutils->unwrappedLoads.clear();

  for (auto pair : newIToNextI) {
    auto newi = pair.first;
    auto nexti = pair.second;
    if (newi != nexti) {
      gutils->replaceAWithB(newi, nexti);
    }
  }

  // This most occur after all the replacements have been made
  // in the previous loop, lest a loop bound being unwrapped use
  // a value being replaced.
  for (auto pair : newIToNextI) {
    auto newi = pair.first;
    auto nexti = pair.second;
    for (auto V : unwrapToOrig[newi]) {
      ValueToValueMapTy available;
      if (auto MD = hasMetadata(V, "enzyme_available")) {
        for (auto &pair : MD->operands()) {
          auto tup = cast<MDNode>(pair);
          auto val = cast<ValueAsMetadata>(tup->getOperand(1))->getValue();
          assert(val);
          available[cast<ValueAsMetadata>(tup->getOperand(0))->getValue()] =
              val;
        }
      }
      IRBuilder<> lb(V);
      // This must disallow caching here as otherwise performing the loop in
      // the wrong order may result in first replacing the later unwrapped
      // value, caching it, then attempting to reuse it for an earlier
      // replacement.
      Value *nval = gutils->unwrapM(nexti, lb, available,
                                    UnwrapMode::LegalFullUnwrapNoTapeReplace,
                                    /*scope*/ nullptr, /*permitCache*/ false);
      assert(nval);
      V->replaceAllUsesWith(nval);
      V->eraseFromParent();
    }
  }

  // Erasure happens after to not erase the key of unwrapToOrig
  for (auto pair : newIToNextI) {
    auto newi = pair.first;
    auto nexti = pair.second;
    if (newi != nexti) {
      if (auto inst = dyn_cast<Instruction>(newi))
        gutils->erase(inst);
    }
  }

  // TODO also can consider switch instance as well
  // TODO can also insert to topLevel as well [note this requires putting the
  // intrinsic at the correct location]
  for (auto &BB : *gutils->oldFunc) {
    SmallVector<BasicBlock *, 4> unreachables;
    SmallVector<BasicBlock *, 4> reachables;
    for (auto Succ : successors(&BB)) {
      if (guaranteedUnreachable.find(Succ) != guaranteedUnreachable.end()) {
        unreachables.push_back(Succ);
      } else {
        reachables.push_back(Succ);
      }
    }

    if (unreachables.size() == 0 || reachables.size() == 0)
      continue;

    if (auto bi = dyn_cast<BranchInst>(BB.getTerminator())) {

      Value *condition = gutils->getNewFromOriginal(bi->getCondition());

      Constant *repVal = (bi->getSuccessor(0) == unreachables[0])
                             ? ConstantInt::getFalse(condition->getContext())
                             : ConstantInt::getTrue(condition->getContext());

      for (auto UI = condition->use_begin(), E = condition->use_end();
           UI != E;) {
        Use &U = *UI;
        ++UI;
        U.set(repVal);
      }
    }
    if (reachables.size() == 1)
      if (auto si = dyn_cast<SwitchInst>(BB.getTerminator())) {
        Value *condition = gutils->getNewFromOriginal(si->getCondition());

        Constant *repVal = nullptr;
        if (si->getDefaultDest() == reachables[0]) {
          std::set<int64_t> cases;
          for (auto c : si->cases()) {
            // TODO this doesnt work with unsigned 64 bit ints or higher
            // integer widths
            cases.insert(cast<ConstantInt>(c.getCaseValue())->getSExtValue());
          }
          int64_t legalNot = 0;
          while (cases.count(legalNot))
            legalNot++;
          repVal = ConstantInt::getSigned(condition->getType(), legalNot);
        } else {
          for (auto c : si->cases()) {
            if (c.getCaseSuccessor() == reachables[0]) {
              repVal = c.getCaseValue();
            }
          }
        }
        assert(repVal);
        for (auto UI = condition->use_begin(), E = condition->use_end();
             UI != E;) {
          Use &U = *UI;
          ++UI;
          U.set(repVal);
        }
      }
  }
}

//! return structtype if recursive function
const AugmentedReturn &EnzymeLogic::CreateAugmentedPrimal(
    Function *todiff, DIFFE_TYPE retType, ArrayRef<DIFFE_TYPE> constant_args,
    TypeAnalysis &TA, bool returnUsed, bool shadowReturnUsed,
    const FnTypeInfo &oldTypeInfo_,
    const std::map<Argument *, bool> _uncacheable_args, bool forceAnonymousTape,
    unsigned width, bool AtomicAdd, bool omp) {
  if (returnUsed)
    assert(!todiff->getReturnType()->isEmptyTy() &&
           !todiff->getReturnType()->isVoidTy());
  if (retType != DIFFE_TYPE::CONSTANT)
    assert(!todiff->getReturnType()->isEmptyTy() &&
           !todiff->getReturnType()->isVoidTy());

  FnTypeInfo oldTypeInfo = preventTypeAnalysisLoops(oldTypeInfo_, todiff);

  assert(constant_args.size() == todiff->getFunctionType()->getNumParams());
  AugmentedCacheKey tup = {todiff,
                           retType,
                           constant_args,
                           std::map<Argument *, bool>(_uncacheable_args.begin(),
                                                      _uncacheable_args.end()),
                           returnUsed,
                           shadowReturnUsed,
                           oldTypeInfo,
                           forceAnonymousTape,
                           AtomicAdd,
                           omp,
                           width};

  auto found = AugmentedCachedFunctions.find(tup);
  if (found != AugmentedCachedFunctions.end()) {
    return found->second;
  }
  TargetLibraryInfo &TLI = PPC.FAM.getResult<TargetLibraryAnalysis>(*todiff);

  // TODO make default typing (not just constant)

  if (auto md = hasMetadata(todiff, "enzyme_augment")) {
    if (!isa<MDTuple>(md)) {
      llvm::errs() << *todiff << "\n";
      llvm::errs() << *md << "\n";
      report_fatal_error(
          "unknown augment for noninvertible function -- metadata incorrect");
    }
    auto md2 = cast<MDTuple>(md);
    assert(md2->getNumOperands() == 1);
    auto gvemd = cast<ConstantAsMetadata>(md2->getOperand(0));
    auto foundcalled = cast<Function>(gvemd->getValue());

    bool hasconstant = false;
    for (auto v : constant_args) {
      if (v == DIFFE_TYPE::CONSTANT) {
        hasconstant = true;
        break;
      }
    }

    if (hasconstant) {
      EmitWarning("NoCustom", todiff,
                  "Massaging provided custom augmented forward pass to handle "
                  "constant argumented");
      SmallVector<Type *, 3> dupargs;
      SmallVector<DIFFE_TYPE, 4> next_constant_args =
          SmallVector<DIFFE_TYPE, 4>(constant_args.begin(),
                                     constant_args.end());
      {
        auto OFT = todiff->getFunctionType();
        for (size_t act_idx = 0; act_idx < constant_args.size(); act_idx++) {
          dupargs.push_back(OFT->getParamType(act_idx));
          switch (constant_args[act_idx]) {
          case DIFFE_TYPE::OUT_DIFF:
            break;
          case DIFFE_TYPE::DUP_ARG:
          case DIFFE_TYPE::DUP_NONEED:
            dupargs.push_back(OFT->getParamType(act_idx));
            break;
          case DIFFE_TYPE::CONSTANT:
            if (!OFT->getParamType(act_idx)->isFPOrFPVectorTy()) {
              next_constant_args[act_idx] = DIFFE_TYPE::DUP_ARG;
            } else {
              next_constant_args[act_idx] = DIFFE_TYPE::OUT_DIFF;
            }
            break;
          }
        }
      }
      FunctionType *FTy =
          FunctionType::get(foundcalled->getReturnType(), dupargs,
                            todiff->getFunctionType()->isVarArg());
      Function *NewF = Function::Create(
          FTy, Function::LinkageTypes::InternalLinkage,
          "fixaugment_" + todiff->getName(), todiff->getParent());

      BasicBlock *BB = BasicBlock::Create(NewF->getContext(), "entry", NewF);
      IRBuilder<> bb(BB);
      auto arg = NewF->arg_begin();
      SmallVector<Value *, 3> fwdargs;
      int act_idx = 0;
      while (arg != NewF->arg_end()) {
        fwdargs.push_back(arg);
        switch (constant_args[act_idx]) {
        case DIFFE_TYPE::OUT_DIFF:
          break;
        case DIFFE_TYPE::DUP_ARG:
        case DIFFE_TYPE::DUP_NONEED:
          arg++;
          fwdargs.push_back(arg);
          break;
        case DIFFE_TYPE::CONSTANT:
          if (next_constant_args[act_idx] != DIFFE_TYPE::OUT_DIFF) {
            fwdargs.push_back(arg);
          }
          break;
        }
        arg++;
        act_idx++;
      }
      auto &aug = CreateAugmentedPrimal(
          todiff, retType, next_constant_args, TA, returnUsed, shadowReturnUsed,
          oldTypeInfo_, _uncacheable_args, forceAnonymousTape, width, AtomicAdd,
          omp);
      auto cal = bb.CreateCall(aug.fn, fwdargs);
      cal->setCallingConv(aug.fn->getCallingConv());

      if (NewF->getReturnType()->isEmptyTy())
        bb.CreateRet(UndefValue::get(NewF->getReturnType()));
      else if (NewF->getReturnType()->isVoidTy())
        bb.CreateRetVoid();
      else
        bb.CreateRet(cal);

      return insert_or_assign<AugmentedCacheKey, AugmentedReturn>(
                 AugmentedCachedFunctions, tup,
                 AugmentedReturn(NewF, aug.tapeType, aug.tapeIndices,
                                 aug.returns, aug.uncacheable_args_map,
                                 aug.can_modref_map))
          ->second;
    }

    if (foundcalled->getReturnType() == todiff->getReturnType()) {
      std::map<AugmentedStruct, int> returnMapping;
      returnMapping[AugmentedStruct::Return] = -1;
      return insert_or_assign<AugmentedCacheKey, AugmentedReturn>(
                 AugmentedCachedFunctions, tup,
                 AugmentedReturn(foundcalled, nullptr, {}, returnMapping, {},
                                 {}))
          ->second;
    }

    if (auto ST = dyn_cast<StructType>(foundcalled->getReturnType())) {
      if (ST->getNumElements() == 3) {
        std::map<AugmentedStruct, int> returnMapping;
        returnMapping[AugmentedStruct::Tape] = 0;
        returnMapping[AugmentedStruct::Return] = 1;
        returnMapping[AugmentedStruct::DifferentialReturn] = 2;
        return insert_or_assign<AugmentedCacheKey, AugmentedReturn>(
                   AugmentedCachedFunctions, tup,
                   AugmentedReturn(foundcalled, nullptr, {}, returnMapping, {},
                                   {}))
            ->second;
      }
      if (ST->getNumElements() == 2 &&
          ST->getElementType(0) == ST->getElementType(1)) {
        std::map<AugmentedStruct, int> returnMapping;
        returnMapping[AugmentedStruct::Return] = 0;
        returnMapping[AugmentedStruct::DifferentialReturn] = 1;
        return insert_or_assign<AugmentedCacheKey, AugmentedReturn>(
                   AugmentedCachedFunctions, tup,
                   AugmentedReturn(foundcalled, nullptr, {}, returnMapping, {},
                                   {}))
            ->second;
      }
      if (ST->getNumElements() == 2) {
        std::map<AugmentedStruct, int> returnMapping;
        returnMapping[AugmentedStruct::Tape] = 0;
        returnMapping[AugmentedStruct::Return] = 1;
        return insert_or_assign<AugmentedCacheKey, AugmentedReturn>(
                   AugmentedCachedFunctions, tup,
                   AugmentedReturn(foundcalled, nullptr, {}, returnMapping, {},
                                   {}))
            ->second;
      }
    }

    std::map<AugmentedStruct, int> returnMapping;
    returnMapping[AugmentedStruct::Tape] = -1;

    return insert_or_assign<AugmentedCacheKey, AugmentedReturn>(
               AugmentedCachedFunctions, tup,
               AugmentedReturn(foundcalled, nullptr, {}, returnMapping, {}, {}))
        ->second; // dyn_cast<StructType>(st->getElementType(0)));
  }

  if (todiff->empty()) {
    if (todiff->empty() && CustomErrorHandler) {
      std::string s;
      llvm::raw_string_ostream ss(s);
      ss << "No augmented forward pass found for " + todiff->getName() << "\n";
      ss << *todiff << "\n";
      CustomErrorHandler(ss.str().c_str(), wrap(todiff),
                         ErrorType::NoDerivative, nullptr);
    }
    llvm::errs() << "mod: " << *todiff->getParent() << "\n";
    llvm::errs() << *todiff << "\n";
    assert(0 && "attempting to differentiate function without definition");
    llvm_unreachable("attempting to differentiate function without definition");
  }
  std::map<AugmentedStruct, int> returnMapping;

  GradientUtils *gutils = GradientUtils::CreateFromClone(
      *this, width, todiff, TLI, TA, oldTypeInfo, retType, constant_args,
      /*returnUsed*/ returnUsed, /*shadowReturnUsed*/ shadowReturnUsed,
      returnMapping, omp);
  gutils->AtomicAdd = AtomicAdd;
  const SmallPtrSet<BasicBlock *, 4> guaranteedUnreachable =
      getGuaranteedUnreachable(gutils->oldFunc);

  // Convert uncacheable args from the input function to the preprocessed
  // function
  std::map<Argument *, bool> _uncacheable_argsPP;
  {
    auto in_arg = todiff->arg_begin();
    auto pp_arg = gutils->oldFunc->arg_begin();
    for (; pp_arg != gutils->oldFunc->arg_end();) {
      if (_uncacheable_args.find(in_arg) == _uncacheable_args.end()) {
        llvm::errs() << " todiff: " << *todiff << "\n";
        llvm::errs() << " inargs: " << *in_arg << "\n";
      }
      assert(_uncacheable_args.find(in_arg) != _uncacheable_args.end());
      _uncacheable_argsPP[pp_arg] = _uncacheable_args.find(in_arg)->second;
      ++pp_arg;
      ++in_arg;
    }
  }

  gutils->forceActiveDetection();
  gutils->forceAugmentedReturns(guaranteedUnreachable);

  // TODO actually populate unnecessaryInstructions with what can be
  // derived without activity info
  SmallPtrSet<const Instruction *, 4> unnecessaryInstructionsTmp;
  for (auto BB : guaranteedUnreachable) {
    for (auto &I : *BB)
      unnecessaryInstructionsTmp.insert(&I);
  }
  gutils->computeGuaranteedFrees(guaranteedUnreachable);

  CacheAnalysis CA(gutils->allocationsWithGuaranteedFree,
                   gutils->rematerializableAllocations, gutils->TR,
                   gutils->OrigAA, gutils->oldFunc,
                   PPC.FAM.getResult<ScalarEvolutionAnalysis>(*gutils->oldFunc),
                   gutils->OrigLI, gutils->OrigDT, TLI,
                   unnecessaryInstructionsTmp, _uncacheable_argsPP,
                   DerivativeMode::ReverseModePrimal, omp);
  const std::map<CallInst *, const std::map<Argument *, bool>>
      uncacheable_args_map = CA.compute_uncacheable_args_for_callsites();

  const std::map<Instruction *, bool> can_modref_map =
      CA.compute_uncacheable_load_map();
  gutils->can_modref_map = &can_modref_map;

  gutils->computeMinCache(guaranteedUnreachable);

  SmallPtrSet<const Value *, 4> unnecessaryValues;
  SmallPtrSet<const Instruction *, 4> unnecessaryInstructions;
  calculateUnusedValuesInFunction(*gutils->oldFunc, unnecessaryValues,
                                  unnecessaryInstructions, returnUsed,
                                  DerivativeMode::ReverseModePrimal, gutils,
                                  TLI, constant_args, guaranteedUnreachable);

  SmallPtrSet<const Instruction *, 4> unnecessaryStores;
  calculateUnusedStoresInFunction(*gutils->oldFunc, unnecessaryStores,
                                  unnecessaryInstructions, gutils, TLI);

  insert_or_assign(AugmentedCachedFunctions, tup,
                   AugmentedReturn(gutils->newFunc, nullptr, {}, returnMapping,
                                   uncacheable_args_map, can_modref_map));
  AugmentedCachedFinished[tup] = false;

  auto getIndex = [&](Instruction *I, CacheType u) -> unsigned {
    return gutils->getIndex(
        std::make_pair(I, u),
        AugmentedCachedFunctions.find(tup)->second.tapeIndices);
  };

  //! Explicitly handle all returns first to ensure that all instructions know
  //! whether or not they are used
  SmallPtrSet<Instruction *, 4> returnuses;

  for (BasicBlock &BB : *gutils->oldFunc) {
    if (auto orig_ri = dyn_cast<ReturnInst>(BB.getTerminator())) {
      auto ri = gutils->getNewFromOriginal(orig_ri);
      Value *orig_oldval = orig_ri->getReturnValue();
      Value *oldval =
          orig_oldval ? gutils->getNewFromOriginal(orig_oldval) : nullptr;
      IRBuilder<> ib(ri);
      Value *rt = UndefValue::get(gutils->newFunc->getReturnType());
      if (oldval && returnUsed) {
        assert(returnMapping.find(AugmentedStruct::Return) !=
               returnMapping.end());
        auto idx = returnMapping.find(AugmentedStruct::Return)->second;
        if (idx < 0)
          rt = oldval;
        else
          rt = ib.CreateInsertValue(rt, oldval, {(unsigned)idx});
        if (Instruction *inst = dyn_cast<Instruction>(rt)) {
          returnuses.insert(inst);
        }
      }

      auto newri = ib.CreateRet(rt);
      gutils->originalToNewFn[orig_ri] = newri;
      gutils->newToOriginalFn.erase(ri);
      gutils->newToOriginalFn[newri] = orig_ri;
      gutils->erase(ri);
    }
  }

  AdjointGenerator<AugmentedReturn *> maker(
      DerivativeMode::ReverseModePrimal, gutils, constant_args, retType,
      getIndex, uncacheable_args_map, &returnuses,
      &AugmentedCachedFunctions.find(tup)->second, nullptr, unnecessaryValues,
      unnecessaryInstructions, unnecessaryStores, guaranteedUnreachable,
      nullptr);

  for (BasicBlock &oBB : *gutils->oldFunc) {
    auto term = oBB.getTerminator();
    assert(term);

    // Don't create derivatives for code that results in termination
    if (guaranteedUnreachable.find(&oBB) != guaranteedUnreachable.end()) {
      SmallVector<Instruction *, 4> toerase;

      // For having the prints still exist on bugs, check if indeed unused
      for (auto &I : oBB) {
        toerase.push_back(&I);
      }
      for (auto I : toerase) {
        maker.eraseIfUnused(*I, /*erase*/ true, /*check*/ true);
      }
      auto newBB = cast<BasicBlock>(gutils->getNewFromOriginal(&oBB));
      if (!newBB->getTerminator()) {
        for (auto next : successors(&oBB)) {
          auto sucBB = cast<BasicBlock>(gutils->getNewFromOriginal(next));
          sucBB->removePredecessor(newBB);
        }
        IRBuilder<> builder(newBB);
        builder.CreateUnreachable();
      }
      continue;
    }

    if (!isa<ReturnInst>(term) && !isa<BranchInst>(term) &&
        !isa<SwitchInst>(term)) {
      llvm::errs() << *oBB.getParent() << "\n";
      llvm::errs() << "unknown terminator instance " << *term << "\n";
      assert(0 && "unknown terminator inst");
      llvm_unreachable("unknown terminator inst");
    }

    BasicBlock::reverse_iterator I = oBB.rbegin(), E = oBB.rend();
    ++I;
    for (; I != E; ++I) {
      maker.visit(&*I);
      assert(oBB.rend() == E);
    }
  }

  if (gutils->knownRecomputeHeuristic.size()) {
    // Even though we could simply iterate through the heuristic map,
    // we explicity iterate in order of the instructions to maintain
    // a deterministic cache ordering.
    for (auto &BB : *gutils->oldFunc)
      for (auto &I : BB) {
        auto found = gutils->knownRecomputeHeuristic.find(&I);
        if (found != gutils->knownRecomputeHeuristic.end()) {
          if (!found->second && !isa<CallInst>(&I)) {
            auto newi = gutils->getNewFromOriginal(&I);
            IRBuilder<> BuilderZ(cast<Instruction>(newi)->getNextNode());
            if (isa<PHINode>(newi)) {
              BuilderZ.SetInsertPoint(
                  cast<Instruction>(newi)->getParent()->getFirstNonPHI());
            }
            gutils->cacheForReverse(BuilderZ, newi,
                                    getIndex(&I, CacheType::Self));
          }
        }
      }
  }

  auto nf = gutils->newFunc;

  while (gutils->inversionAllocs->size() > 0) {
    gutils->inversionAllocs->back().moveBefore(
        gutils->newFunc->getEntryBlock().getFirstNonPHIOrDbgOrLifetime());
  }

  (IRBuilder<>(gutils->inversionAllocs)).CreateUnreachable();
  DeleteDeadBlock(gutils->inversionAllocs);

  for (Argument &Arg : gutils->newFunc->args()) {
    if (Arg.hasAttribute(Attribute::Returned))
      Arg.removeAttr(Attribute::Returned);
    if (Arg.hasAttribute(Attribute::StructRet))
      Arg.removeAttr(Attribute::StructRet);
  }

  if (gutils->newFunc->hasFnAttribute(Attribute::OptimizeNone))
    gutils->newFunc->removeFnAttr(Attribute::OptimizeNone);

#if LLVM_VERSION_MAJOR >= 14
  if (gutils->newFunc->getAttributes().getRetDereferenceableBytes())
#else
  if (gutils->newFunc->getDereferenceableBytes(
          llvm::AttributeList::ReturnIndex))
#endif
  {
#if LLVM_VERSION_MAJOR >= 14
    gutils->newFunc->removeRetAttr(Attribute::Dereferenceable);
#else
    gutils->newFunc->removeAttribute(llvm::AttributeList::ReturnIndex,
                                     Attribute::Dereferenceable);
#endif
  }

  // TODO could keep nonnull if returning value -1
  if (gutils->newFunc->getAttributes().getRetAlignment()) {
#if LLVM_VERSION_MAJOR >= 14
    gutils->newFunc->removeRetAttr(Attribute::Alignment);
#else
    gutils->newFunc->removeAttribute(llvm::AttributeList::ReturnIndex,
                                     Attribute::Alignment);
#endif
  }

  llvm::Attribute::AttrKind attrs[] = {
    llvm::Attribute::NoAlias,
#if LLVM_VERSION_MAJOR >= 11
    llvm::Attribute::NoUndef,
#endif
    llvm::Attribute::NonNull,
    llvm::Attribute::ZExt,
  };
  for (auto attr : attrs) {
#if LLVM_VERSION_MAJOR >= 14
    if (gutils->newFunc->hasRetAttribute(attr)) {
      gutils->newFunc->removeRetAttr(attr);
    }
#else
    if (gutils->newFunc->hasAttribute(llvm::AttributeList::ReturnIndex, attr)) {
      gutils->newFunc->removeAttribute(llvm::AttributeList::ReturnIndex, attr);
    }
#endif
  }

  //! Keep track of inverted pointers we may need to return
  ValueToValueMapTy invertedRetPs;
  if (shadowReturnUsed) {
    for (BasicBlock &BB : *gutils->oldFunc) {
      if (auto ri = dyn_cast<ReturnInst>(BB.getTerminator())) {
        if (Value *orig_oldval = ri->getReturnValue()) {
          auto newri = gutils->getNewFromOriginal(ri);
          IRBuilder<> BuilderZ(newri);
          invertedRetPs[newri] = gutils->invertPointerM(orig_oldval, BuilderZ);
        }
      }
    }
  }

  gutils->eraseFictiousPHIs();

  if (llvm::verifyFunction(*gutils->newFunc, &llvm::errs())) {
    llvm::errs() << *gutils->oldFunc << "\n";
    llvm::errs() << *gutils->newFunc << "\n";
    report_fatal_error("function failed verification (2)");
  }

  StructType *sty = cast<StructType>(gutils->newFunc->getReturnType());
  SmallVector<Type *, 4> RetTypes(sty->elements().begin(),
                                  sty->elements().end());

  SmallVector<Type *, 4> MallocTypes;

  for (auto a : gutils->getTapeValues()) {
    MallocTypes.push_back(a->getType());
  }

  Type *tapeType = StructType::get(nf->getContext(), MallocTypes);

  bool removeTapeStruct = MallocTypes.size() == 1;
  if (removeTapeStruct) {
    tapeType = MallocTypes[0];

    for (auto &a : AugmentedCachedFunctions.find(tup)->second.tapeIndices) {
      a.second = -1;
    }
  }

  bool recursive =
      AugmentedCachedFunctions.find(tup)->second.fn->getNumUses() > 0 ||
      forceAnonymousTape;
  bool noTape = MallocTypes.size() == 0 && !forceAnonymousTape;

  int oldretIdx = -1;
  if (returnMapping.find(AugmentedStruct::Return) != returnMapping.end()) {
    oldretIdx = returnMapping[AugmentedStruct::Return];
  }

  if (noTape || omp) {
    auto tidx = returnMapping.find(AugmentedStruct::Tape)->second;
    if (noTape)
      returnMapping.erase(AugmentedStruct::Tape);
    if (noTape)
      AugmentedCachedFunctions.find(tup)->second.returns.erase(
          AugmentedStruct::Tape);
    if (returnMapping.find(AugmentedStruct::Return) != returnMapping.end()) {
      AugmentedCachedFunctions.find(tup)
          ->second.returns[AugmentedStruct::Return] -=
          (returnMapping[AugmentedStruct::Return] > tidx) ? 1 : 0;
      returnMapping[AugmentedStruct::Return] -=
          (returnMapping[AugmentedStruct::Return] > tidx) ? 1 : 0;
    }
    if (returnMapping.find(AugmentedStruct::DifferentialReturn) !=
        returnMapping.end()) {
      AugmentedCachedFunctions.find(tup)
          ->second.returns[AugmentedStruct::DifferentialReturn] -=
          (returnMapping[AugmentedStruct::DifferentialReturn] > tidx) ? 1 : 0;
      returnMapping[AugmentedStruct::DifferentialReturn] -=
          (returnMapping[AugmentedStruct::DifferentialReturn] > tidx) ? 1 : 0;
    }
    RetTypes.erase(RetTypes.begin() + tidx);
  } else if (recursive) {
    assert(RetTypes[returnMapping.find(AugmentedStruct::Tape)->second] ==
           Type::getInt8PtrTy(nf->getContext()));
  } else {
    RetTypes[returnMapping.find(AugmentedStruct::Tape)->second] = tapeType;
  }

  bool noReturn = RetTypes.size() == 0;
  Type *RetType = StructType::get(nf->getContext(), RetTypes);
  if (noReturn)
    RetType = Type::getVoidTy(RetType->getContext());
  if (noReturn)
    assert(noTape || omp);

  bool removeStruct = RetTypes.size() == 1;

  if (removeStruct) {
    RetType = RetTypes[0];
    for (auto &a : returnMapping) {
      a.second = -1;
    }
    for (auto &a : AugmentedCachedFunctions.find(tup)->second.returns) {
      a.second = -1;
    }
  }

  ValueToValueMapTy VMap;
  SmallVector<Type *, 4> ArgTypes;
  for (const Argument &I : nf->args()) {
    ArgTypes.push_back(I.getType());
  }

  if (omp && !noTape) {
    // see lack of struct type for openmp
    ArgTypes.push_back(PointerType::getUnqual(tapeType));
    // ArgTypes.push_back(tapeType);
  }

  // Create a new function type...
  FunctionType *FTy =
      FunctionType::get(RetType, ArgTypes, nf->getFunctionType()->isVarArg());

  // Create the new function...
  Function *NewF = Function::Create(
      FTy, nf->getLinkage(), "augmented_" + todiff->getName(), nf->getParent());

  unsigned ii = 0, jj = 0;
  auto i = nf->arg_begin(), j = NewF->arg_begin();
  for (; i != nf->arg_end();) {
    VMap[i] = j;
    if (nf->hasParamAttribute(ii, Attribute::NoCapture)) {
      NewF->addParamAttr(jj, Attribute::NoCapture);
    }
    if (nf->hasParamAttribute(ii, Attribute::NoAlias)) {
      NewF->addParamAttr(jj, Attribute::NoAlias);
    }

    j->setName(i->getName());
    ++j;
    ++jj;
    ++i;
    ++ii;
  }

  SmallVector<ReturnInst *, 4> Returns;
#if LLVM_VERSION_MAJOR >= 13
  CloneFunctionInto(NewF, nf, VMap, CloneFunctionChangeType::LocalChangesOnly,
                    Returns, "", nullptr);
#else
  CloneFunctionInto(NewF, nf, VMap, nf->getSubprogram() != nullptr, Returns, "",
                    nullptr);
#endif

  IRBuilder<> ib(NewF->getEntryBlock().getFirstNonPHI());

  Value *ret = noReturn ? nullptr : ib.CreateAlloca(RetType);
  if (!noReturn && EnzymeZeroCache) {
    ib.CreateStore(Constant::getNullValue(RetType), ret);
  }

  if (!noTape) {
    Value *tapeMemory;
    if (recursive && !omp) {
      auto i64 = Type::getInt64Ty(NewF->getContext());
      ConstantInt *size = ConstantInt::get(
          i64,
          NewF->getParent()->getDataLayout().getTypeAllocSizeInBits(tapeType) /
              8);
      Value *memory;
      if (!size->isZero()) {
        tapeMemory =
            CallInst::CreateMalloc(NewF->getEntryBlock().getFirstNonPHI(), i64,
                                   tapeType, size, nullptr, nullptr, "tapemem");
        CallInst *malloccall = dyn_cast<CallInst>(tapeMemory);
        if (malloccall == nullptr) {
          malloccall =
              cast<CallInst>(cast<Instruction>(tapeMemory)->getOperand(0));
        }
        for (auto attr : {Attribute::NoAlias, Attribute::NonNull}) {
#if LLVM_VERSION_MAJOR >= 14
          malloccall->addRetAttr(attr);
#else
          malloccall->addAttribute(AttributeList::ReturnIndex, attr);
#endif
        }
        if (EnzymeZeroCache) {
          IRBuilder<> B(malloccall->getNextNode());
          Value *args[] = {
              malloccall,
              ConstantInt::get(Type::getInt8Ty(malloccall->getContext()), 0),
              malloccall->getArgOperand(0),
              ConstantInt::getFalse(malloccall->getContext())};
          Type *tys[] = {args[0]->getType(), args[2]->getType()};

          B.CreateCall(Intrinsic::getDeclaration(NewF->getParent(),
                                                 Intrinsic::memset, tys),
                       args);
        }
#if LLVM_VERSION_MAJOR >= 14
        malloccall->addDereferenceableRetAttr(size->getLimitedValue());
#if !defined(FLANG) && !defined(ROCM)
        AttrBuilder B(malloccall->getContext());
#else
        AttrBuilder B;
#endif
        B.addDereferenceableOrNullAttr(size->getLimitedValue());
        malloccall->setAttributes(malloccall->getAttributes().addRetAttributes(
            malloccall->getContext(), B));
#else
        malloccall->addDereferenceableAttr(llvm::AttributeList::ReturnIndex,
                                           size->getLimitedValue());
        malloccall->addDereferenceableOrNullAttr(
            llvm::AttributeList::ReturnIndex, size->getLimitedValue());
#endif
        memory = malloccall;
      } else {
        memory =
            ConstantPointerNull::get(Type::getInt8PtrTy(NewF->getContext()));
      }
      Value *Idxs[] = {
          ib.getInt32(0),
          ib.getInt32(returnMapping.find(AugmentedStruct::Tape)->second),
      };
      assert(memory);
      assert(ret);
      Value *gep = ret;
      if (!removeStruct) {
#if LLVM_VERSION_MAJOR > 7
        gep = ib.CreateGEP(ret->getType()->getPointerElementType(), ret, Idxs,
                           "");
#else
        gep = ib.CreateGEP(ret, Idxs, "");
#endif
        cast<GetElementPtrInst>(gep)->setIsInBounds(true);
      }
      ib.CreateStore(memory, gep);
    } else if (omp) {
      j->setName("tape");
      tapeMemory = j;
      // if structs were supported by openmp we could do this, but alas, no
      // IRBuilder<> B(NewF->getEntryBlock().getFirstNonPHI());
      // tapeMemory = B.CreateAlloca(j->getType());
      // B.CreateStore(j, tapeMemory);
    } else {
      Value *Idxs[] = {
          ib.getInt32(0),
          ib.getInt32(returnMapping.find(AugmentedStruct::Tape)->second),
      };
      tapeMemory = ret;
      if (!removeStruct) {
#if LLVM_VERSION_MAJOR > 7
        tapeMemory = ib.CreateGEP(ret->getType()->getPointerElementType(), ret,
                                  Idxs, "");
#else
        tapeMemory = ib.CreateGEP(ret, Idxs, "");
#endif
        cast<GetElementPtrInst>(tapeMemory)->setIsInBounds(true);
      }
    }

    unsigned i = 0;
    for (auto v : gutils->getTapeValues()) {
      if (!isa<UndefValue>(v)) {
        auto inst = cast<Instruction>(VMap[v]);
        IRBuilder<> ib(inst->getNextNode());
        if (isa<PHINode>(inst))
          ib.SetInsertPoint(inst->getParent()->getFirstNonPHI());
        Value *Idxs[] = {ib.getInt32(0), ib.getInt32(i)};
        Value *gep = tapeMemory;
        if (!removeTapeStruct) {
#if LLVM_VERSION_MAJOR > 7
          gep = ib.CreateGEP(tapeMemory->getType()->getPointerElementType(),
                             tapeMemory, Idxs, "");
#else
          gep = ib.CreateGEP(tapeMemory, Idxs, "");
#endif
          cast<GetElementPtrInst>(gep)->setIsInBounds(true);
        }
        ib.CreateStore(VMap[v], gep);
      }
      ++i;
    }
  }

  for (BasicBlock &BB : *nf) {
    auto ri = dyn_cast<ReturnInst>(BB.getTerminator());
    if (ri == nullptr)
      continue;
    ReturnInst *rim = cast<ReturnInst>(VMap[ri]);
    IRBuilder<> ib(rim);
    if (returnUsed) {
      Value *rv = rim->getReturnValue();
      assert(rv);
      Value *actualrv = nullptr;
      if (auto iv = dyn_cast<InsertValueInst>(rv)) {
        if (iv->getNumIndices() == 1 && (int)iv->getIndices()[0] == oldretIdx) {
          actualrv = iv->getInsertedValueOperand();
        }
      }
      if (actualrv == nullptr) {
        if (oldretIdx < 0)
          actualrv = rv;
        else
          actualrv = ib.CreateExtractValue(rv, {(unsigned)oldretIdx});
      }
      Value *gep =
          removeStruct
              ? ret
              : ib.CreateConstGEP2_32(
                    RetType, ret, 0,
                    returnMapping.find(AugmentedStruct::Return)->second, "");
      if (auto ggep = dyn_cast<GetElementPtrInst>(gep)) {
        ggep->setIsInBounds(true);
      }
      ib.CreateStore(actualrv, gep);
    }

    if (shadowReturnUsed) {
      assert(invertedRetPs[ri]);
      if (!isa<UndefValue>(invertedRetPs[ri])) {
        Value *gep =
            removeStruct
                ? ret
                : ib.CreateConstGEP2_32(
                      RetType, ret, 0,
                      returnMapping.find(AugmentedStruct::DifferentialReturn)
                          ->second,
                      "");
        if (auto ggep = dyn_cast<GetElementPtrInst>(gep)) {
          ggep->setIsInBounds(true);
        }
        if (isa<ConstantData>(invertedRetPs[ri]))
          ib.CreateStore(invertedRetPs[ri], gep);
        else {
          assert(VMap[invertedRetPs[ri]]);
          ib.CreateStore(VMap[invertedRetPs[ri]], gep);
        }
      }
    }
    if (noReturn)
      ib.CreateRetVoid();
    else {
#if LLVM_VERSION_MAJOR > 7
      ib.CreateRet(ib.CreateLoad(ret->getType()->getPointerElementType(), ret));
#else
      ib.CreateRet(ib.CreateLoad(ret));
#endif
    }
    cast<Instruction>(VMap[ri])->eraseFromParent();
  }

  clearFunctionAttributes(NewF);

  if (llvm::verifyFunction(*NewF, &llvm::errs())) {
    llvm::errs() << *gutils->oldFunc << "\n";
    llvm::errs() << *NewF << "\n";
    report_fatal_error("augmented function failed verification (3)");
  }
  {
    PreservedAnalyses PA;
    PPC.FAM.invalidate(*NewF, PA);
  }

  SmallVector<CallInst *, 4> fnusers;
  SmallVector<std::pair<GlobalVariable *, DerivativeMode>, 1> gfnusers;
  for (auto user : AugmentedCachedFunctions.find(tup)->second.fn->users()) {
    if (auto CI = dyn_cast<CallInst>(user)) {
      fnusers.push_back(CI);
    } else {
      if (auto CS = dyn_cast<ConstantStruct>(user)) {
        for (auto cuser : CS->users()) {
          if (auto G = dyn_cast<GlobalVariable>(cuser)) {
            if (("_enzyme_reverse_" + todiff->getName() + "'").str() ==
                G->getName()) {
              gfnusers.emplace_back(G, DerivativeMode::ReverseModeGradient);
              continue;
            }
            if (("_enzyme_forwardsplit_" + todiff->getName() + "'").str() ==
                G->getName()) {
              gfnusers.emplace_back(G, DerivativeMode::ForwardModeSplit);
              continue;
            }
          }
          llvm::errs() << *gutils->newFunc->getParent() << "\n";
          llvm::errs() << *cuser << "\n";
          llvm::errs() << *user << "\n";
          llvm_unreachable("Bad cuser of staging augmented forward fn");
        }
        continue;
      }
      llvm::errs() << *gutils->newFunc->getParent() << "\n";
      llvm::errs() << *user << "\n";
      llvm_unreachable("Bad user of staging augmented forward fn");
    }
  }
  for (auto user : fnusers) {
    if (removeStruct) {
      IRBuilder<> B(user);
      auto n = user->getName().str();
      user->setName("");
      SmallVector<Value *, 4> args(user->arg_begin(), user->arg_end());
      auto rep = B.CreateCall(NewF, args);
      rep->copyIRFlags(user);
      rep->setAttributes(user->getAttributes());
      rep->setCallingConv(user->getCallingConv());
      rep->setTailCallKind(user->getTailCallKind());
      rep->setDebugLoc(gutils->getNewFromOriginal(user->getDebugLoc()));
      assert(user);
      SmallVector<ExtractValueInst *, 4> torep;
      for (auto u : user->users()) {
        assert(u);
        if (auto ei = dyn_cast<ExtractValueInst>(u)) {
          torep.push_back(ei);
        }
      }
      for (auto ei : torep) {
        ei->replaceAllUsesWith(rep);
        ei->eraseFromParent();
      }
      user->eraseFromParent();
    } else {
      user->setCalledFunction(NewF);
    }
  }
  PPC.AlwaysInline(NewF);
  auto Arch = llvm::Triple(NewF->getParent()->getTargetTriple()).getArch();
  if (Arch == Triple::nvptx || Arch == Triple::nvptx64)
    PPC.ReplaceReallocs(NewF, /*mem2reg*/ true);

  AugmentedCachedFunctions.find(tup)->second.fn = NewF;
  if (recursive || (omp && !noTape))
    AugmentedCachedFunctions.find(tup)->second.tapeType = tapeType;
  insert_or_assign(AugmentedCachedFinished, tup, true);

  for (auto pair : gfnusers) {
    auto GV = pair.first;
    GV->setName("_tmp");
    auto R = gutils->GetOrCreateShadowFunction(
        *this, TLI, TA, todiff, pair.second, width, gutils->AtomicAdd);
    SmallVector<ConstantExpr *, 1> users;
    for (auto U : GV->users()) {
      if (auto CE = dyn_cast<ConstantExpr>(U)) {
        if (CE->isCast()) {
          users.push_back(CE);
        }
      }
    }
    for (auto U : users) {
      U->replaceAllUsesWith(ConstantExpr::getPointerCast(R, U->getType()));
    }
    GV->eraseFromParent();
  }

  {
    PreservedAnalyses PA;
    PPC.FAM.invalidate(*gutils->newFunc, PA);
  }

  Function *tempFunc = gutils->newFunc;
  delete gutils;
  tempFunc->eraseFromParent();

  // Do not run post processing optimizations if the body of an openmp
  // parallel so the adjointgenerator can successfully extract the allocation
  // and frees and hoist them into the parent. Optimizing before then may
  // make the IR different to traverse, and thus impossible to find the allocs.
  if (PostOpt && !omp)
    PPC.optimizeIntermediate(NewF);
  if (EnzymePrint)
    llvm::errs() << *NewF << "\n";
  return AugmentedCachedFunctions.find(tup)->second;
}

void createTerminator(DiffeGradientUtils *gutils, BasicBlock *oBB,
                      DIFFE_TYPE retType, ReturnType retVal) {
  TypeResults &TR = gutils->TR;
  ReturnInst *inst = dyn_cast<ReturnInst>(oBB->getTerminator());
  // In forward mode we only need to update the return value
  if (inst == nullptr)
    return;

  ReturnInst *newInst = cast<ReturnInst>(gutils->getNewFromOriginal(inst));
  BasicBlock *nBB = newInst->getParent();
  assert(nBB);
  IRBuilder<> nBuilder(nBB);
  nBuilder.setFastMathFlags(getFast());

  SmallVector<Value *, 2> retargs;

  Value *toret = UndefValue::get(gutils->newFunc->getReturnType());

  switch (retVal) {
  case ReturnType::Return: {
    auto ret = inst->getOperand(0);

    if (retType == DIFFE_TYPE::CONSTANT) {
      toret = gutils->getNewFromOriginal(ret);
    } else if (!ret->getType()->isFPOrFPVectorTy() &&
               TR.getReturnAnalysis().Inner0().isPossiblePointer()) {
      toret = gutils->invertPointerM(ret, nBuilder);
    } else if (!gutils->isConstantValue(ret)) {
      toret = gutils->diffe(ret, nBuilder);
    } else {
      Type *retTy = gutils->getShadowType(ret->getType());
      toret = Constant::getNullValue(retTy);
    }

    break;
  }
  case ReturnType::TwoReturns: {
    if (retType == DIFFE_TYPE::CONSTANT)
      assert(false && "Invalid return type");
    auto ret = inst->getOperand(0);

    toret =
        nBuilder.CreateInsertValue(toret, gutils->getNewFromOriginal(ret), 0);

    if (!ret->getType()->isFPOrFPVectorTy() &&
        TR.getReturnAnalysis().Inner0().isPossiblePointer()) {
      toret = nBuilder.CreateInsertValue(
          toret, gutils->invertPointerM(ret, nBuilder), 1);
    } else if (!gutils->isConstantValue(ret)) {
      toret =
          nBuilder.CreateInsertValue(toret, gutils->diffe(ret, nBuilder), 1);
    } else {
      toret = nBuilder.CreateInsertValue(
          toret, Constant::getNullValue(ret->getType()), 1);
    }
    break;
  }
  case ReturnType::Void: {
    gutils->erase(gutils->getNewFromOriginal(inst));
    nBuilder.CreateRetVoid();
    return;
  }
  default: {
    llvm::errs() << "Invalid return type: " << to_string(retVal)
                 << "for function: \n"
                 << gutils->newFunc << "\n";
    assert(false && "Invalid return type for function");
    return;
  }
  }

  gutils->erase(newInst);
  nBuilder.CreateRet(toret);
  return;
}

void createInvertedTerminator(DiffeGradientUtils *gutils,
                              ArrayRef<DIFFE_TYPE> argTypes, BasicBlock *oBB,
                              AllocaInst *retAlloca, AllocaInst *dretAlloca,
                              unsigned extraArgs, DIFFE_TYPE retType) {
  LoopContext loopContext;
  BasicBlock *BB = cast<BasicBlock>(gutils->getNewFromOriginal(oBB));
  bool inLoop = gutils->getContext(BB, loopContext);
  BasicBlock *BB2 = gutils->reverseBlocks[BB].back();
  assert(BB2);
  IRBuilder<> Builder(BB2);
  Builder.setFastMathFlags(getFast());

  std::map<BasicBlock *, SmallVector<BasicBlock *, 4>> targetToPreds;
  for (auto pred : predecessors(BB)) {
    targetToPreds[gutils->getReverseOrLatchMerge(pred, BB)].emplace_back(pred);
  }

  if (targetToPreds.size() == 0) {
    SmallVector<Value *, 4> retargs;

    if (retAlloca) {
#if LLVM_VERSION_MAJOR > 7
      auto result = Builder.CreateLoad(retAlloca->getAllocatedType(), retAlloca,
                                       "retreload");
#else
      auto result = Builder.CreateLoad(retAlloca, "retreload");
#endif
      // TODO reintroduce invariant load/group
      // result->setMetadata(LLVMContext::MD_invariant_load,
      // MDNode::get(retAlloca->getContext(), {}));
      retargs.push_back(result);
    }

    if (dretAlloca) {
#if LLVM_VERSION_MAJOR > 7
      auto result = Builder.CreateLoad(dretAlloca->getAllocatedType(),
                                       dretAlloca, "dretreload");
#else
      auto result = Builder.CreateLoad(dretAlloca, "dretreload");
#endif
      // TODO reintroduce invariant load/group
      // result->setMetadata(LLVMContext::MD_invariant_load,
      // MDNode::get(dretAlloca->getContext(), {}));
      retargs.push_back(result);
    }

    for (auto &I : gutils->oldFunc->args()) {
      if (!gutils->isConstantValue(&I) &&
          argTypes[I.getArgNo()] == DIFFE_TYPE::OUT_DIFF) {
        retargs.push_back(gutils->diffe(&I, Builder));
      }
    }

    if (gutils->newFunc->getReturnType()->isVoidTy()) {
      assert(retargs.size() == 0);
      Builder.CreateRetVoid();
      return;
    }

    Value *toret = UndefValue::get(gutils->newFunc->getReturnType());
    for (unsigned i = 0; i < retargs.size(); ++i) {
      unsigned idx[] = {i};
      toret = Builder.CreateInsertValue(toret, retargs[i], idx);
    }
    Builder.CreateRet(toret);
    return;
  }

  // PHINodes to replace that will contain true iff the predecessor was given
  // basicblock
  std::map<BasicBlock *, PHINode *> replacePHIs;
  SmallVector<SelectInst *, 4> selects;

  IRBuilder<> phibuilder(BB2);
  bool setphi = false;

  // Ensure phi values have their derivatives propagated
  for (auto I = oBB->begin(), E = oBB->end(); I != E; ++I) {
    PHINode *orig = dyn_cast<PHINode>(&*I);
    if (orig == nullptr)
      break;
    if (gutils->isConstantInstruction(orig))
      continue;

    size_t size = 1;
    if (orig->getType()->isSized())
      size = (gutils->newFunc->getParent()->getDataLayout().getTypeSizeInBits(
                  orig->getType()) +
              7) /
             8;

    auto PNtype = gutils->TR.intType(size, orig, /*necessary*/ false);

    // TODO remove explicit type check and only use PNtype
    if (PNtype == BaseType::Anything || PNtype == BaseType::Pointer ||
        orig->getType()->isPointerTy())
      continue;

    Type *PNfloatType = PNtype.isFloat();
    if (!PNfloatType) {
      if (looseTypeAnalysis) {
        if (orig->getType()->isFPOrFPVectorTy())
          PNfloatType = orig->getType()->getScalarType();
        if (orig->getType()->isIntOrIntVectorTy())
          continue;
      }
    }
    if (!PNfloatType) {
      if (CustomErrorHandler) {
        std::string str;
        raw_string_ostream ss(str);
        ss << "Cannot deduce type of phi " << *orig;
        CustomErrorHandler(str.c_str(), wrap(orig), ErrorType::NoType,
                           &gutils->TR.analyzer);
      }
      llvm::errs() << *gutils->oldFunc->getParent() << "\n";
      llvm::errs() << *gutils->oldFunc << "\n";
      llvm::errs() << " for orig " << *orig << " saw "
                   << gutils->TR.intType(size, orig, /*necessary*/ false).str()
                   << " - "
                   << "\n";
      gutils->TR.intType(size, orig, /*necessary*/ true);
    }

    auto prediff = gutils->diffe(orig, Builder);

    bool handled = false;

    SmallVector<Instruction *, 4> activeUses;
    for (auto u : orig->users()) {
      if (!gutils->isConstantInstruction(cast<Instruction>(u)))
        activeUses.push_back(cast<Instruction>(u));
      else if (retType == DIFFE_TYPE::OUT_DIFF && isa<ReturnInst>(u))
        activeUses.push_back(cast<Instruction>(u));
    }
    if (activeUses.size() == 1 && inLoop &&
        gutils->getNewFromOriginal(orig->getParent()) == loopContext.header &&
        loopContext.exitBlocks.size() == 1) {
      SmallVector<BasicBlock *, 1> Latches;
      gutils->OrigLI.getLoopFor(orig->getParent())->getLoopLatches(Latches);
      bool allIncoming = true;
      for (auto Latch : Latches) {
        if (activeUses[0] != orig->getIncomingValueForBlock(Latch)) {
          allIncoming = false;
          break;
        }
      }
      if (allIncoming) {
        if (auto SI = dyn_cast<SelectInst>(activeUses[0])) {
          for (int i = 0; i < 2; i++) {
            if (SI->getOperand(i + 1) == orig) {
              auto oval = orig->getIncomingValueForBlock(
                  gutils->getOriginalFromNew(loopContext.preheader));
              BasicBlock *pred = loopContext.preheader;
              if (replacePHIs.find(pred) == replacePHIs.end()) {
                replacePHIs[pred] = Builder.CreatePHI(
                    Type::getInt1Ty(pred->getContext()), 1, "replacePHI");
                if (!setphi) {
                  phibuilder.SetInsertPoint(replacePHIs[pred]);
                  setphi = true;
                }
              }

              auto ddiff = gutils->diffe(SI, Builder);
              gutils->setDiffe(SI,
                               Builder.CreateSelect(
                                   replacePHIs[pred],
                                   Constant::getNullValue(prediff->getType()),
                                   ddiff),
                               Builder);
              handled = true;
              if (!gutils->isConstantValue(oval)) {
                BasicBlock *REB =
                    gutils->reverseBlocks[*loopContext.exitBlocks.begin()]
                        .back();
                IRBuilder<> EB(REB);
                if (REB->getTerminator())
                  EB.SetInsertPoint(REB->getTerminator());

                auto index = gutils->getOrInsertConditionalIndex(
                    gutils->getNewFromOriginal(SI->getOperand(0)), loopContext,
                    i == 1);
                Value *sdif = Builder.CreateSelect(
                    Builder.CreateICmpEQ(
                        gutils->lookupM(index, EB),
                        Constant::getNullValue(index->getType())),
                    ddiff, Constant::getNullValue(ddiff->getType()));

                SelectInst *dif = cast<SelectInst>(Builder.CreateSelect(
                    replacePHIs[pred], sdif,
                    Constant::getNullValue(prediff->getType())));
                auto addedSelects =
                    gutils->addToDiffe(oval, dif, Builder, PNfloatType);

                for (auto select : addedSelects)
                  selects.emplace_back(select);
              }
              break;
            }
          }
        }
        if (auto BO = dyn_cast<BinaryOperator>(activeUses[0])) {

          if (BO->getOpcode() == Instruction::FDiv &&
              BO->getOperand(0) == orig) {

            auto oval = orig->getIncomingValueForBlock(
                gutils->getOriginalFromNew(loopContext.preheader));
            BasicBlock *pred = loopContext.preheader;
            if (replacePHIs.find(pred) == replacePHIs.end()) {
              replacePHIs[pred] = Builder.CreatePHI(
                  Type::getInt1Ty(pred->getContext()), 1, "replacePHI");
              if (!setphi) {
                phibuilder.SetInsertPoint(replacePHIs[pred]);
                setphi = true;
              }
            }

            auto ddiff = gutils->diffe(BO, Builder);
            gutils->setDiffe(
                BO,
                Builder.CreateSelect(replacePHIs[pred],
                                     Constant::getNullValue(prediff->getType()),
                                     ddiff),
                Builder);
            handled = true;

            if (!gutils->isConstantValue(oval)) {

              BasicBlock *REB =
                  gutils->reverseBlocks[*loopContext.exitBlocks.begin()].back();
              IRBuilder<> EB(REB);
              if (REB->getTerminator())
                EB.SetInsertPoint(REB->getTerminator());

              auto product = gutils->getOrInsertTotalMultiplicativeProduct(
                  gutils->getNewFromOriginal(BO->getOperand(1)), loopContext);

              SelectInst *dif = cast<SelectInst>(Builder.CreateSelect(
                  replacePHIs[pred],
                  Builder.CreateFDiv(ddiff, gutils->lookupM(product, EB)),
                  Constant::getNullValue(prediff->getType())));
              auto addedSelects =
                  gutils->addToDiffe(oval, dif, Builder, PNfloatType);

              for (auto select : addedSelects)
                selects.emplace_back(select);
            }
          }
        }
      }
    }

    if (!handled) {
      gutils->setDiffe(
          orig, Constant::getNullValue(gutils->getShadowType(orig->getType())),
          Builder);

      for (BasicBlock *opred : predecessors(oBB)) {
        auto oval = orig->getIncomingValueForBlock(opred);
        if (gutils->isConstantValue(oval)) {
          continue;
        }

        if (orig->getNumIncomingValues() == 1) {
          gutils->addToDiffe(oval, prediff, Builder, PNfloatType);
        } else {
          BasicBlock *pred =
              cast<BasicBlock>(gutils->getNewFromOriginal(opred));
          if (replacePHIs.find(pred) == replacePHIs.end()) {
            replacePHIs[pred] = Builder.CreatePHI(
                Type::getInt1Ty(pred->getContext()), 1, "replacePHI");
            if (!setphi) {
              phibuilder.SetInsertPoint(replacePHIs[pred]);
              setphi = true;
            }
          }
          SelectInst *dif = cast<SelectInst>(
              Builder.CreateSelect(replacePHIs[pred], prediff,
                                   Constant::getNullValue(prediff->getType())));
          auto addedSelects =
              gutils->addToDiffe(oval, dif, Builder, PNfloatType);

          for (auto select : addedSelects)
            selects.emplace_back(select);
        }
      }
    }
  }
  if (!setphi) {
    phibuilder.SetInsertPoint(Builder.GetInsertBlock(),
                              Builder.GetInsertPoint());
  }

  if (inLoop && BB == loopContext.header) {
    std::map<BasicBlock *, SmallVector<BasicBlock *, 4>> targetToPreds;
    for (auto pred : predecessors(BB)) {
      if (pred == loopContext.preheader)
        continue;
      targetToPreds[gutils->getReverseOrLatchMerge(pred, BB)].emplace_back(
          pred);
    }

    assert(targetToPreds.size() &&
           "only loops with one backedge are presently supported");

#if LLVM_VERSION_MAJOR > 7
    Value *av = phibuilder.CreateLoad(loopContext.var->getType(),
                                      loopContext.antivaralloc);
#else
    Value *av = phibuilder.CreateLoad(loopContext.antivaralloc);
#endif
    Value *phi =
        phibuilder.CreateICmpEQ(av, Constant::getNullValue(av->getType()));
    Value *nphi = phibuilder.CreateNot(phi);

    for (auto pair : replacePHIs) {
      Value *replaceWith = nullptr;

      if (pair.first == loopContext.preheader) {
        replaceWith = phi;
      } else {
        replaceWith = nphi;
      }

      pair.second->replaceAllUsesWith(replaceWith);
      pair.second->eraseFromParent();
    }
    BB2 = gutils->reverseBlocks[BB].back();
    Builder.SetInsertPoint(BB2);

    Builder.CreateCondBr(
        phi, gutils->getReverseOrLatchMerge(loopContext.preheader, BB),
        targetToPreds.begin()->first);

  } else {
    std::map<BasicBlock *, std::vector<std::pair<BasicBlock *, BasicBlock *>>>
        phiTargetToPreds;
    for (auto pair : replacePHIs) {
      phiTargetToPreds[pair.first].emplace_back(std::make_pair(pair.first, BB));
    }
    BasicBlock *fakeTarget = nullptr;
    for (auto pred : predecessors(BB)) {
      if (phiTargetToPreds.find(pred) != phiTargetToPreds.end())
        continue;
      if (fakeTarget == nullptr)
        fakeTarget = pred;
      phiTargetToPreds[fakeTarget].emplace_back(std::make_pair(pred, BB));
    }
    gutils->branchToCorrespondingTarget(BB, phibuilder, phiTargetToPreds,
                                        &replacePHIs);

    std::map<BasicBlock *, std::vector<std::pair<BasicBlock *, BasicBlock *>>>
        targetToPreds;
    for (auto pred : predecessors(BB)) {
      targetToPreds[gutils->getReverseOrLatchMerge(pred, BB)].emplace_back(
          std::make_pair(pred, BB));
    }
    BB2 = gutils->reverseBlocks[BB].back();
    Builder.SetInsertPoint(BB2);
    gutils->branchToCorrespondingTarget(BB, Builder, targetToPreds);
  }

  // Optimize select of not to just be a select with operands switched
  for (SelectInst *select : selects) {
    if (BinaryOperator *bo = dyn_cast<BinaryOperator>(select->getCondition())) {
      if (bo->getOpcode() == BinaryOperator::Xor) {
        if (isa<ConstantInt>(bo->getOperand(0)) &&
            cast<ConstantInt>(bo->getOperand(0))->isOne()) {
          select->setCondition(bo->getOperand(1));
          auto tmp = select->getTrueValue();
          select->setTrueValue(select->getFalseValue());
          select->setFalseValue(tmp);
          if (bo->getNumUses() == 0)
            bo->eraseFromParent();
        } else if (isa<ConstantInt>(bo->getOperand(1)) &&
                   cast<ConstantInt>(bo->getOperand(1))->isOne()) {
          select->setCondition(bo->getOperand(0));
          auto tmp = select->getTrueValue();
          select->setTrueValue(select->getFalseValue());
          select->setFalseValue(tmp);
          if (bo->getNumUses() == 0)
            bo->eraseFromParent();
        }
      }
    }
  }
}

Function *EnzymeLogic::CreatePrimalAndGradient(
    const ReverseCacheKey &&key, TypeAnalysis &TA,
    const AugmentedReturn *augmenteddata, bool omp) {

  assert(key.mode == DerivativeMode::ReverseModeCombined ||
         key.mode == DerivativeMode::ReverseModeGradient);

  FnTypeInfo oldTypeInfo = preventTypeAnalysisLoops(key.typeInfo, key.todiff);

  if (key.retType != DIFFE_TYPE::CONSTANT)
    assert(!key.todiff->getReturnType()->isVoidTy());
  if (ReverseCachedFunctions.find(key) != ReverseCachedFunctions.end()) {
    return ReverseCachedFunctions.find(key)->second;
  }

  if (key.returnUsed)
    assert(key.mode == DerivativeMode::ReverseModeCombined);

  TargetLibraryInfo &TLI =
      PPC.FAM.getResult<TargetLibraryAnalysis>(*key.todiff);

  // TODO change this to go by default function type assumptions
  bool hasconstant = false;
  for (auto v : key.constant_args) {
    if (v == DIFFE_TYPE::CONSTANT) {
      hasconstant = true;
      break;
    }
  }

  if (hasMetadata(key.todiff, "enzyme_gradient")) {

    DIFFE_TYPE subretType = key.todiff->getReturnType()->isFPOrFPVectorTy()
                                ? DIFFE_TYPE::OUT_DIFF
                                : DIFFE_TYPE::DUP_ARG;
    if (key.todiff->getReturnType()->isVoidTy() ||
        key.todiff->getReturnType()->isEmptyTy())
      subretType = DIFFE_TYPE::CONSTANT;
    assert(subretType == key.retType);

    auto res = getDefaultFunctionTypeForGradient(key.todiff->getFunctionType(),
                                                 /*retType*/ key.retType);

    if (key.mode == DerivativeMode::ReverseModeCombined) {

      Type *FRetTy =
          res.second.empty()
              ? Type::getVoidTy(key.todiff->getContext())
              : StructType::get(key.todiff->getContext(), {res.second});

      FunctionType *FTy = FunctionType::get(
          FRetTy, res.first, key.todiff->getFunctionType()->isVarArg());

      Function *NewF = Function::Create(
          FTy, Function::LinkageTypes::InternalLinkage,
          "fixgradient_" + key.todiff->getName(), key.todiff->getParent());

      size_t argnum = 0;
      for (Argument &Arg : NewF->args()) {
        Arg.setName("arg" + std::to_string(argnum));
        ++argnum;
      }

      BasicBlock *BB = BasicBlock::Create(NewF->getContext(), "entry", NewF);
      IRBuilder<> bb(BB);

      auto &aug = CreateAugmentedPrimal(
          key.todiff, key.retType, key.constant_args, TA, key.returnUsed,
          key.shadowReturnUsed, key.typeInfo, key.uncacheable_args,
          /*forceAnonymousTape*/ false, key.width, key.AtomicAdd, omp);

      SmallVector<Value *, 4> fwdargs;
      for (auto &a : NewF->args())
        fwdargs.push_back(&a);
      if (key.retType == DIFFE_TYPE::OUT_DIFF)
        fwdargs.pop_back();
      auto cal = bb.CreateCall(aug.fn, fwdargs);
      cal->setCallingConv(aug.fn->getCallingConv());

      llvm::Value *tape = nullptr;

      if (aug.returns.find(AugmentedStruct::Tape) != aug.returns.end()) {
        auto tapeIdx = aug.returns.find(AugmentedStruct::Tape)->second;
        tape = (tapeIdx == -1) ? cal : bb.CreateExtractValue(cal, tapeIdx);
        if (tape->getType()->isEmptyTy())
          tape = UndefValue::get(tape->getType());
      }

      if (aug.tapeType) {
        assert(tape);
        auto tapep = bb.CreatePointerCast(
            tape, PointerType::get(
                      aug.tapeType,
                      cast<PointerType>(tape->getType())->getAddressSpace()));
#if LLVM_VERSION_MAJOR > 7
        auto truetape = bb.CreateLoad(aug.tapeType, tapep, "tapeld");
#else
        auto truetape = bb.CreateLoad(tapep, "tapeld");
#endif
        truetape->setMetadata("enzyme_mustcache",
                              MDNode::get(truetape->getContext(), {}));

        if (key.freeMemory) {
          auto size = NewF->getParent()->getDataLayout().getTypeAllocSizeInBits(
              aug.tapeType);
          if (size != 0) {
            CallInst *ci = cast<CallInst>(CallInst::CreateFree(tape, BB));
            bb.Insert(ci);
            ci->addAttribute(AttributeList::FirstArgIndex, Attribute::NonNull);
          }
        }
        tape = truetape;
      }

      auto revfn = CreatePrimalAndGradient(
          (ReverseCacheKey){.todiff = key.todiff,
                            .retType = key.retType,
                            .constant_args = key.constant_args,
                            .uncacheable_args = key.uncacheable_args,
                            .returnUsed = false,
                            .shadowReturnUsed = false,
                            .mode = DerivativeMode::ReverseModeGradient,
                            .width = key.width,
                            .freeMemory = key.freeMemory,
                            .AtomicAdd = key.AtomicAdd,
                            .additionalType = tape ? tape->getType() : nullptr,
                            .typeInfo = key.typeInfo},
          TA, &aug, omp);

      SmallVector<Value *, 4> revargs;
      for (auto &a : NewF->args()) {
        revargs.push_back(&a);
      }
      if (tape) {
        revargs.push_back(tape);
      }
      auto revcal = bb.CreateCall(revfn, revargs);
      revcal->setCallingConv(revfn->getCallingConv());

      if (NewF->getReturnType()->isEmptyTy()) {
        bb.CreateRet(UndefValue::get(NewF->getReturnType()));
      } else if (NewF->getReturnType()->isVoidTy()) {
        bb.CreateRetVoid();
      } else {
        bb.CreateRet(revcal);
      }
      assert(!key.returnUsed);

      return insert_or_assign2<ReverseCacheKey, Function *>(
                 ReverseCachedFunctions, key, NewF)
          ->second;
    }

    auto md = key.todiff->getMetadata("enzyme_gradient");
    if (!isa<MDTuple>(md)) {
      llvm::errs() << *key.todiff << "\n";
      llvm::errs() << *md << "\n";
      report_fatal_error(
          "unknown gradient for noninvertible function -- metadata incorrect");
    }
    auto md2 = cast<MDTuple>(md);
    assert(md2->getNumOperands() == 1);
    auto gvemd = cast<ConstantAsMetadata>(md2->getOperand(0));
    auto foundcalled = cast<Function>(gvemd->getValue());

    if (hasconstant) {
      EmitWarning("NoCustom",
                  key.todiff->getEntryBlock().begin()->getDebugLoc(),
                  key.todiff, &key.todiff->getEntryBlock(),
                  "Massaging provided custom reverse pass");
      SmallVector<Type *, 3> dupargs;
      std::vector<DIFFE_TYPE> next_constant_args(key.constant_args);
      {
        auto OFT = key.todiff->getFunctionType();
        for (size_t act_idx = 0; act_idx < key.constant_args.size();
             act_idx++) {
          dupargs.push_back(OFT->getParamType(act_idx));
          switch (key.constant_args[act_idx]) {
          case DIFFE_TYPE::OUT_DIFF:
            break;
          case DIFFE_TYPE::DUP_ARG:
          case DIFFE_TYPE::DUP_NONEED:
            dupargs.push_back(OFT->getParamType(act_idx));
            break;
          case DIFFE_TYPE::CONSTANT:
            if (!OFT->getParamType(act_idx)->isFPOrFPVectorTy()) {
              next_constant_args[act_idx] = DIFFE_TYPE::DUP_ARG;
            } else {
              next_constant_args[act_idx] = DIFFE_TYPE::OUT_DIFF;
            }
            break;
          }
        }
      }

      auto revfn = CreatePrimalAndGradient(
          (ReverseCacheKey){.todiff = key.todiff,
                            .retType = key.retType,
                            .constant_args = next_constant_args,
                            .uncacheable_args = key.uncacheable_args,
                            .returnUsed = key.returnUsed,
                            .shadowReturnUsed = false,
                            .mode = DerivativeMode::ReverseModeGradient,
                            .width = key.width,
                            .freeMemory = key.freeMemory,
                            .AtomicAdd = key.AtomicAdd,
                            .additionalType = nullptr,
                            .typeInfo = key.typeInfo},
          TA, augmenteddata, omp);

      {
        auto arg = revfn->arg_begin();
        for (auto cidx : next_constant_args) {
          arg++;
          if (cidx == DIFFE_TYPE::DUP_ARG || cidx == DIFFE_TYPE::DUP_NONEED)
            arg++;
        }
        while (arg != revfn->arg_end()) {
          dupargs.push_back(arg->getType());
          arg++;
        }
      }

      FunctionType *FTy =
          FunctionType::get(revfn->getReturnType(), dupargs,
                            revfn->getFunctionType()->isVarArg());
      Function *NewF = Function::Create(
          FTy, Function::LinkageTypes::InternalLinkage,
          "fixgradient_" + key.todiff->getName(), key.todiff->getParent());

      BasicBlock *BB = BasicBlock::Create(NewF->getContext(), "entry", NewF);
      IRBuilder<> bb(BB);
      auto arg = NewF->arg_begin();
      SmallVector<Value *, 3> revargs;
      size_t act_idx = 0;
      while (act_idx != key.constant_args.size()) {
        revargs.push_back(arg);
        switch (key.constant_args[act_idx]) {
        case DIFFE_TYPE::OUT_DIFF:
          break;
        case DIFFE_TYPE::DUP_ARG:
        case DIFFE_TYPE::DUP_NONEED:
          arg++;
          revargs.push_back(arg);
          break;
        case DIFFE_TYPE::CONSTANT:
          if (next_constant_args[act_idx] != DIFFE_TYPE::OUT_DIFF) {
            revargs.push_back(arg);
          }
          break;
        }
        arg++;
        act_idx++;
      }
      while (arg != NewF->arg_end()) {
        revargs.push_back(arg);
        arg++;
      }
      auto cal = bb.CreateCall(revfn, revargs);
      cal->setCallingConv(revfn->getCallingConv());

      if (NewF->getReturnType()->isEmptyTy())
        bb.CreateRet(UndefValue::get(NewF->getReturnType()));
      else if (NewF->getReturnType()->isVoidTy())
        bb.CreateRetVoid();
      else
        bb.CreateRet(cal);

      return insert_or_assign2<ReverseCacheKey, Function *>(
                 ReverseCachedFunctions, key, NewF)
          ->second;
    }

    if (!key.returnUsed && key.freeMemory) {
      assert(augmenteddata);
      if (foundcalled->arg_size() == res.first.size() + 1 /*tape*/) {
        auto lastarg = foundcalled->arg_end();
        lastarg--;
        res.first.push_back(lastarg->getType());
      } else if (foundcalled->arg_size() == res.first.size()) {
        // res.first.push_back(StructType::get(todiff->getContext(), {}));
      } else {
        llvm::errs() << "expected args: [";
        for (auto a : res.first) {
          llvm::errs() << *a << " ";
        }
        llvm::errs() << "]\n";
        llvm::errs() << *foundcalled << "\n";
        assert(0 && "bad type for custom gradient");
        llvm_unreachable("bad type for custom gradient");
      }

      auto st = dyn_cast<StructType>(foundcalled->getReturnType());
      bool wrongRet =
          st == nullptr && !foundcalled->getReturnType()->isVoidTy();
      if (wrongRet) {
        // if (wrongRet || !hasTape) {
        Type *FRetTy =
            res.second.empty()
                ? Type::getVoidTy(key.todiff->getContext())
                : StructType::get(key.todiff->getContext(), {res.second});

        FunctionType *FTy = FunctionType::get(
            FRetTy, res.first, key.todiff->getFunctionType()->isVarArg());
        Function *NewF = Function::Create(
            FTy, Function::LinkageTypes::InternalLinkage,
            "fixgradient_" + key.todiff->getName(), key.todiff->getParent());
        NewF->setAttributes(foundcalled->getAttributes());
        if (NewF->hasFnAttribute(Attribute::NoInline)) {
          NewF->removeFnAttr(Attribute::NoInline);
        }
        size_t argnum = 0;
        for (Argument &Arg : NewF->args()) {
          if (Arg.hasAttribute(Attribute::Returned))
            Arg.removeAttr(Attribute::Returned);
          if (Arg.hasAttribute(Attribute::StructRet))
            Arg.removeAttr(Attribute::StructRet);
          Arg.setName("arg" + std::to_string(argnum));
          ++argnum;
        }

        BasicBlock *BB = BasicBlock::Create(NewF->getContext(), "entry", NewF);
        IRBuilder<> bb(BB);
        SmallVector<Value *, 4> args;
        for (auto &a : NewF->args())
          args.push_back(&a);
        // if (!hasTape) {
        //  args.pop_back();
        //}
        auto cal = bb.CreateCall(foundcalled, args);
        cal->setCallingConv(foundcalled->getCallingConv());
        Value *val = cal;
        if (wrongRet) {
          auto ut = UndefValue::get(NewF->getReturnType());
          if (val->getType()->isEmptyTy() && res.second.size() == 0) {
            val = ut;
          } else if (res.second.size() == 1 &&
                     res.second[0] == val->getType()) {
            val = bb.CreateInsertValue(ut, cal, {0u});
          } else {
            llvm::errs() << *foundcalled << "\n";
            assert(0 && "illegal type for reverse");
            llvm_unreachable("illegal type for reverse");
          }
        }
        bb.CreateRet(val);
        foundcalled = NewF;
      }
      return insert_or_assign2<ReverseCacheKey, Function *>(
                 ReverseCachedFunctions, key, foundcalled)
          ->second;
    }

    EmitWarning("NoCustom", key.todiff->getEntryBlock().begin()->getDebugLoc(),
                key.todiff, &key.todiff->getEntryBlock(),
                "Not using provided custom reverse pass as require either "
                "return or non-constant");
  }

  if (key.todiff->empty()) {
    std::string s;
    llvm::raw_string_ostream ss(s);
    ss << "No reverse pass found for " + key.todiff->getName() << "\n";
    ss << *key.todiff << "\n";
    if (CustomErrorHandler) {
      CustomErrorHandler(ss.str().c_str(), wrap(key.todiff),
                         ErrorType::NoDerivative, nullptr);
    } else {
      llvm_unreachable(ss.str().c_str());
    }
  }
  assert(!key.todiff->empty());

  ReturnType retVal =
      key.returnUsed ? (key.shadowReturnUsed ? ReturnType::ArgsWithTwoReturns
                                             : ReturnType::ArgsWithReturn)
                     : (key.shadowReturnUsed ? ReturnType::ArgsWithReturn
                                             : ReturnType::Args);

  bool diffeReturnArg = key.retType == DIFFE_TYPE::OUT_DIFF;

  DiffeGradientUtils *gutils = DiffeGradientUtils::CreateFromClone(
      *this, key.mode, key.width, key.todiff, TLI, TA, oldTypeInfo, key.retType,
      diffeReturnArg, key.constant_args, retVal, key.additionalType, omp);

  gutils->AtomicAdd = key.AtomicAdd;
  gutils->FreeMemory = key.freeMemory;
  insert_or_assign2<ReverseCacheKey, Function *>(ReverseCachedFunctions, key,
                                                 gutils->newFunc);

  const SmallPtrSet<BasicBlock *, 4> guaranteedUnreachable =
      getGuaranteedUnreachable(gutils->oldFunc);

  // Convert uncacheable args from the input function to the preprocessed
  // function
  std::map<Argument *, bool> _uncacheable_argsPP;
  {
    auto in_arg = key.todiff->arg_begin();
    auto pp_arg = gutils->oldFunc->arg_begin();
    for (; pp_arg != gutils->oldFunc->arg_end();) {
      _uncacheable_argsPP[pp_arg] = key.uncacheable_args.find(in_arg)->second;
      ++pp_arg;
      ++in_arg;
    }
  }

  gutils->forceActiveDetection();
  gutils->forceAugmentedReturns(guaranteedUnreachable);

  gutils->computeGuaranteedFrees(guaranteedUnreachable);

  // TODO populate with actual unnecessaryInstructions once the dependency
  // cycle with activity analysis is removed
  SmallPtrSet<const Instruction *, 4> unnecessaryInstructionsTmp;
  for (auto BB : guaranteedUnreachable) {
    for (auto &I : *BB)
      unnecessaryInstructionsTmp.insert(&I);
  }
  CacheAnalysis CA(gutils->allocationsWithGuaranteedFree,
                   gutils->rematerializableAllocations, gutils->TR,
                   gutils->OrigAA, gutils->oldFunc,
                   PPC.FAM.getResult<ScalarEvolutionAnalysis>(*gutils->oldFunc),
                   gutils->OrigLI, gutils->OrigDT, TLI,
                   unnecessaryInstructionsTmp, _uncacheable_argsPP, key.mode,
                   omp);
  const std::map<CallInst *, const std::map<Argument *, bool>>
      uncacheable_args_map =
          (augmenteddata) ? augmenteddata->uncacheable_args_map
                          : CA.compute_uncacheable_args_for_callsites();

  const std::map<Instruction *, bool> can_modref_map =
      augmenteddata ? augmenteddata->can_modref_map
                    : CA.compute_uncacheable_load_map();
  gutils->can_modref_map = &can_modref_map;

  std::map<std::pair<Instruction *, CacheType>, int> mapping;
  if (augmenteddata)
    mapping = augmenteddata->tapeIndices;

  auto getIndex = [&](Instruction *I, CacheType u) -> unsigned {
    return gutils->getIndex(std::make_pair(I, u), mapping);
  };

  gutils->computeMinCache(guaranteedUnreachable);

  SmallPtrSet<const Value *, 4> unnecessaryValues;
  SmallPtrSet<const Instruction *, 4> unnecessaryInstructions;
  calculateUnusedValuesInFunction(*gutils->oldFunc, unnecessaryValues,
                                  unnecessaryInstructions, key.returnUsed,
                                  key.mode, gutils, TLI, key.constant_args,
                                  guaranteedUnreachable);

  SmallPtrSet<const Instruction *, 4> unnecessaryStores;
  calculateUnusedStoresInFunction(*gutils->oldFunc, unnecessaryStores,
                                  unnecessaryInstructions, gutils, TLI);

  Value *additionalValue = nullptr;
  if (key.additionalType) {
    auto v = gutils->newFunc->arg_end();
    v--;
    additionalValue = v;
    assert(key.mode != DerivativeMode::ReverseModeCombined);
    assert(augmenteddata);

    // TODO VERIFY THIS
    if (augmenteddata->tapeType &&
        augmenteddata->tapeType != additionalValue->getType()) {
      IRBuilder<> BuilderZ(gutils->inversionAllocs);
      if (!augmenteddata->tapeType->isEmptyTy()) {
        auto tapep = BuilderZ.CreatePointerCast(
            additionalValue, PointerType::getUnqual(augmenteddata->tapeType));
#if LLVM_VERSION_MAJOR > 7
        LoadInst *truetape =
            BuilderZ.CreateLoad(augmenteddata->tapeType, tapep, "truetape");
#else
        LoadInst *truetape = BuilderZ.CreateLoad(tapep, "truetape");
#endif
        truetape->setMetadata("enzyme_mustcache",
                              MDNode::get(truetape->getContext(), {}));

        if (!omp && gutils->FreeMemory) {
          CallInst *ci =
              cast<CallInst>(CallInst::CreateFree(additionalValue, truetape));
          ci->moveAfter(truetape);
          ci->addAttribute(AttributeList::FirstArgIndex, Attribute::NonNull);
        }
        additionalValue = truetape;
      } else {
        if (gutils->FreeMemory) {
          CallInst *ci = cast<CallInst>(
              CallInst::CreateFree(additionalValue, gutils->inversionAllocs));
          ci->addAttribute(AttributeList::FirstArgIndex, Attribute::NonNull);
          BuilderZ.Insert(ci);
        }
        additionalValue = UndefValue::get(augmenteddata->tapeType);
      }
    }

    // TODO here finish up making recursive structs simply pass in i8*
    gutils->setTape(additionalValue);
  }

  Argument *differetval = nullptr;
  if (key.retType == DIFFE_TYPE::OUT_DIFF) {
    auto endarg = gutils->newFunc->arg_end();
    endarg--;
    if (key.additionalType)
      endarg--;
    differetval = endarg;

    if (!key.todiff->getReturnType()->isVoidTy()) {
      if (!(differetval->getType() ==
            gutils->getShadowType(key.todiff->getReturnType()))) {
        llvm::errs() << *gutils->oldFunc << "\n";
        llvm::errs() << *gutils->newFunc << "\n";
      }
      assert(differetval->getType() ==
             gutils->getShadowType(key.todiff->getReturnType()));
    }
  }

  // Explicitly handle all returns first to ensure that return instructions know
  // if they are used or not before
  //   processessing instructions
  std::map<ReturnInst *, StoreInst *> replacedReturns;
  llvm::AllocaInst *retAlloca = nullptr;
  llvm::AllocaInst *dretAlloca = nullptr;
  if (key.returnUsed) {
    retAlloca =
        IRBuilder<>(&gutils->newFunc->getEntryBlock().front())
            .CreateAlloca(key.todiff->getReturnType(), nullptr, "toreturn");
  }
  if (key.shadowReturnUsed) {
    assert(key.retType == DIFFE_TYPE::DUP_ARG ||
           key.retType == DIFFE_TYPE::DUP_NONEED);
    assert(key.mode == DerivativeMode::ReverseModeCombined);
    dretAlloca =
        IRBuilder<>(&gutils->newFunc->getEntryBlock().front())
            .CreateAlloca(key.todiff->getReturnType(), nullptr, "dtoreturn");
  }
  if (key.mode == DerivativeMode::ReverseModeCombined ||
      key.mode == DerivativeMode::ReverseModeGradient) {
    for (BasicBlock &oBB : *gutils->oldFunc) {
      if (ReturnInst *orig = dyn_cast<ReturnInst>(oBB.getTerminator())) {
        ReturnInst *op = cast<ReturnInst>(gutils->getNewFromOriginal(orig));
        BasicBlock *BB = op->getParent();
        IRBuilder<> rb(op);
        rb.setFastMathFlags(getFast());

        if (retAlloca) {
          StoreInst *si = rb.CreateStore(
              gutils->getNewFromOriginal(orig->getReturnValue()), retAlloca);
          replacedReturns[orig] = si;
        }

        if (key.retType == DIFFE_TYPE::DUP_ARG ||
            key.retType == DIFFE_TYPE::DUP_NONEED) {
          if (dretAlloca) {
            rb.CreateStore(gutils->invertPointerM(orig->getReturnValue(), rb),
                           dretAlloca);
          }
        } else if (key.retType == DIFFE_TYPE::OUT_DIFF) {
          assert(orig->getReturnValue());
          assert(differetval);
          if (!gutils->isConstantValue(orig->getReturnValue())) {
            IRBuilder<> reverseB(gutils->reverseBlocks[BB].back());
            gutils->setDiffe(orig->getReturnValue(), differetval, reverseB);
          }
        } else {
          assert(dretAlloca == nullptr);
        }

        rb.CreateBr(gutils->reverseBlocks[BB].front());
        gutils->erase(op);
      }
    }
  }

  AdjointGenerator<const AugmentedReturn *> maker(
      key.mode, gutils, key.constant_args, key.retType, getIndex,
      uncacheable_args_map,
      /*returnuses*/ nullptr, augmenteddata, &replacedReturns,
      unnecessaryValues, unnecessaryInstructions, unnecessaryStores,
      guaranteedUnreachable, dretAlloca);

  for (BasicBlock &oBB : *gutils->oldFunc) {
    // Don't create derivatives for code that results in termination
    if (guaranteedUnreachable.find(&oBB) != guaranteedUnreachable.end()) {
      auto newBB = cast<BasicBlock>(gutils->getNewFromOriginal(&oBB));
      SmallVector<BasicBlock *, 4> toRemove;
      if (auto II = dyn_cast<InvokeInst>(oBB.getTerminator())) {
        toRemove.push_back(
            cast<BasicBlock>(gutils->getNewFromOriginal(II->getNormalDest())));
      } else {
        for (auto next : successors(&oBB)) {
          auto sucBB = cast<BasicBlock>(gutils->getNewFromOriginal(next));
          toRemove.push_back(sucBB);
        }
      }

      for (auto sucBB : toRemove) {
        if (sucBB->empty() || !isa<PHINode>(sucBB->begin()))
          continue;

        SmallVector<PHINode *, 2> phis;
        for (PHINode &Phi : sucBB->phis()) {
          phis.push_back(&Phi);
        }
        for (PHINode *Phi : phis) {
          unsigned NumPreds = Phi->getNumIncomingValues();
          if (NumPreds == 0)
            continue;
          Phi->removeIncomingValue(newBB);
        }
      }

      SmallVector<Instruction *, 2> toerase;
      for (auto &I : oBB) {
        toerase.push_back(&I);
      }
      for (auto I : llvm::reverse(toerase)) {
        maker.eraseIfUnused(*I, /*erase*/ true,
                            /*check*/ key.mode ==
                                DerivativeMode::ReverseModeCombined);
      }
      if (newBB->getTerminator())
        newBB->getTerminator()->eraseFromParent();
      IRBuilder<> builder(newBB);
      builder.CreateUnreachable();
      continue;
    }

    auto term = oBB.getTerminator();
    assert(term);
    if (!isa<ReturnInst>(term) && !isa<BranchInst>(term) &&
        !isa<SwitchInst>(term)) {
      llvm::errs() << *oBB.getParent() << "\n";
      llvm::errs() << "unknown terminator instance " << *term << "\n";
      assert(0 && "unknown terminator inst");
    }

    BasicBlock::reverse_iterator I = oBB.rbegin(), E = oBB.rend();
    ++I;
    for (; I != E; ++I) {
      maker.visit(&*I);
      assert(oBB.rend() == E);
    }

    createInvertedTerminator(gutils, key.constant_args, &oBB, retAlloca,
                             dretAlloca,
                             0 + (key.additionalType ? 1 : 0) +
                                 ((key.retType == DIFFE_TYPE::DUP_ARG ||
                                   key.retType == DIFFE_TYPE::DUP_NONEED)
                                      ? 1
                                      : 0),
                             key.retType);
  }

  if (key.mode == DerivativeMode::ReverseModeGradient)
    restoreCache(gutils, mapping, guaranteedUnreachable);

  gutils->eraseFictiousPHIs();

  BasicBlock *entry = &gutils->newFunc->getEntryBlock();

  auto Arch =
      llvm::Triple(gutils->newFunc->getParent()->getTargetTriple()).getArch();
  unsigned int SharedAddrSpace =
      Arch == Triple::amdgcn ? (int)AMDGPU::HSAMD::AddressSpaceQualifier::Local
                             : 3;

  if (key.mode == DerivativeMode::ReverseModeCombined) {
    BasicBlock *sharedBlock = nullptr;
    for (auto &g : gutils->newFunc->getParent()->globals()) {
      if (hasMetadata(&g, "enzyme_internalshadowglobal")) {
        IRBuilder<> entryBuilder(gutils->inversionAllocs,
                                 gutils->inversionAllocs->begin());

        if ((Arch == Triple::nvptx || Arch == Triple::nvptx64 ||
             Arch == Triple::amdgcn) &&
            g.getType()->getAddressSpace() == SharedAddrSpace) {
          if (sharedBlock == nullptr)
            sharedBlock = BasicBlock::Create(entry->getContext(), "shblock",
                                             gutils->newFunc);
          entryBuilder.SetInsertPoint(sharedBlock);
        }
        auto store = entryBuilder.CreateStore(
            Constant::getNullValue(g.getValueType()), &g);
#if LLVM_VERSION_MAJOR >= 11
        if (g.getAlign())
          store->setAlignment(g.getAlign().getValue());
#elif LLVM_VERSION_MAJOR >= 10
        store->setAlignment(Align(g.getAlignment()));
#else
        store->setAlignment(g.getAlignment());
#endif
      }
    }
    if (sharedBlock) {
      BasicBlock *OldEntryInsts = entry->splitBasicBlock(entry->begin());
      entry->getTerminator()->eraseFromParent();
      IRBuilder<> ebuilder(entry);

      Value *tx, *ty, *tz;
      if (Arch == Triple::nvptx || Arch == Triple::nvptx64) {
        tx = ebuilder.CreateCall(Intrinsic::getDeclaration(
            gutils->newFunc->getParent(), Intrinsic::nvvm_read_ptx_sreg_tid_x));
        ty = ebuilder.CreateCall(Intrinsic::getDeclaration(
            gutils->newFunc->getParent(), Intrinsic::nvvm_read_ptx_sreg_tid_y));
        tz = ebuilder.CreateCall(Intrinsic::getDeclaration(
            gutils->newFunc->getParent(), Intrinsic::nvvm_read_ptx_sreg_tid_z));
      } else if (Arch == Triple::amdgcn) {
        tx = ebuilder.CreateCall(Intrinsic::getDeclaration(
            gutils->newFunc->getParent(), Intrinsic::amdgcn_workitem_id_x));
        ty = ebuilder.CreateCall(Intrinsic::getDeclaration(
            gutils->newFunc->getParent(), Intrinsic::amdgcn_workitem_id_y));
        tz = ebuilder.CreateCall(Intrinsic::getDeclaration(
            gutils->newFunc->getParent(), Intrinsic::amdgcn_workitem_id_z));
      } else {
        llvm_unreachable("unknown gpu architecture");
      }
      Value *OrVal = ebuilder.CreateOr(ebuilder.CreateOr(tx, ty), tz);

      ebuilder.CreateCondBr(
          ebuilder.CreateICmpEQ(OrVal, ConstantInt::get(OrVal->getType(), 0)),
          sharedBlock, OldEntryInsts);

      IRBuilder<> instbuilder(OldEntryInsts, OldEntryInsts->begin());

      auto BarrierInst = Arch == Triple::amdgcn
                             ? (llvm::Intrinsic::ID)Intrinsic::amdgcn_s_barrier
                             : (llvm::Intrinsic::ID)Intrinsic::nvvm_barrier0;
      cast<CallInst>(instbuilder.CreateCall(
          Intrinsic::getDeclaration(gutils->newFunc->getParent(), BarrierInst),
          {}));
      OldEntryInsts->moveAfter(entry);
      sharedBlock->moveAfter(entry);
      IRBuilder<> sbuilder(sharedBlock);
      sbuilder.CreateBr(OldEntryInsts);
      SmallVector<AllocaInst *, 3> AIs;
      for (auto &I : *OldEntryInsts) {
        if (auto AI = dyn_cast<AllocaInst>(&I))
          AIs.push_back(AI);
      }
      for (auto AI : AIs)
        AI->moveBefore(entry->getFirstNonPHIOrDbgOrLifetime());
      entry = OldEntryInsts;
    }
  }

  cleanupInversionAllocs(gutils, entry);
  clearFunctionAttributes(gutils->newFunc);

  if (llvm::verifyFunction(*gutils->newFunc, &llvm::errs())) {
    llvm::errs() << *gutils->oldFunc << "\n";
    llvm::errs() << *gutils->newFunc << "\n";
    report_fatal_error("function failed verification (4)");
  }

  auto nf = gutils->newFunc;
  delete gutils;

  {
    PreservedAnalyses PA;
    PPC.FAM.invalidate(*nf, PA);
  }
  PPC.AlwaysInline(nf);
  if (Arch == Triple::nvptx || Arch == Triple::nvptx64)
    PPC.ReplaceReallocs(nf, /*mem2reg*/ true);

  // Do not run post processing optimizations if the body of an openmp
  // parallel so the adjointgenerator can successfully extract the allocation
  // and frees and hoist them into the parent. Optimizing before then may
  // make the IR different to traverse, and thus impossible to find the allocs.
  if (PostOpt && !omp)
    PPC.optimizeIntermediate(nf);
  if (EnzymePrint) {
    llvm::errs() << *nf << "\n";
  }
  return nf;
}

Function *EnzymeLogic::CreateForwardDiff(
    Function *todiff, DIFFE_TYPE retType, ArrayRef<DIFFE_TYPE> constant_args,
    TypeAnalysis &TA, bool returnUsed, DerivativeMode mode, bool freeMemory,
    unsigned width, llvm::Type *additionalArg, const FnTypeInfo &oldTypeInfo_,
    const std::map<Argument *, bool> _uncacheable_args,
    const AugmentedReturn *augmenteddata, bool omp) {
  assert(retType != DIFFE_TYPE::OUT_DIFF);

  assert(mode == DerivativeMode::ForwardMode ||
         mode == DerivativeMode::ForwardModeSplit);

  FnTypeInfo oldTypeInfo = preventTypeAnalysisLoops(oldTypeInfo_, todiff);

  if (retType != DIFFE_TYPE::CONSTANT)
    assert(!todiff->getReturnType()->isVoidTy());

  ForwardCacheKey tup = {todiff,
                         retType,
                         constant_args,
                         std::map<Argument *, bool>(_uncacheable_args.begin(),
                                                    _uncacheable_args.end()),
                         returnUsed,
                         mode,
                         width,
                         additionalArg,
                         oldTypeInfo};

  if (ForwardCachedFunctions.find(tup) != ForwardCachedFunctions.end()) {
    return ForwardCachedFunctions.find(tup)->second;
  }

  TargetLibraryInfo &TLI = PPC.FAM.getResult<TargetLibraryAnalysis>(*todiff);

  // TODO change this to go by default function type assumptions
  bool hasconstant = false;
  for (auto v : constant_args) {
    assert(v != DIFFE_TYPE::OUT_DIFF);
    if (v == DIFFE_TYPE::CONSTANT) {
      hasconstant = true;
      break;
    }
  }

  if (auto md = hasMetadata(todiff, (mode == DerivativeMode::ForwardMode)
                                        ? "enzyme_derivative"
                                        : "enzyme_splitderivative")) {
    if (!isa<MDTuple>(md)) {
      llvm::errs() << *todiff << "\n";
      llvm::errs() << *md << "\n";
      report_fatal_error(
          "unknown derivative for function -- metadata incorrect");
    }
    auto md2 = cast<MDTuple>(md);
    assert(md2->getNumOperands() == 1);
    auto gvemd = cast<ConstantAsMetadata>(md2->getOperand(0));
    auto foundcalled = cast<Function>(gvemd->getValue());

    if ((foundcalled->getReturnType()->isVoidTy() ||
         retType != DIFFE_TYPE::CONSTANT) &&
        !hasconstant && returnUsed)
      return foundcalled;

    if (!foundcalled->getReturnType()->isVoidTy() && !hasconstant) {
      if (returnUsed && retType == DIFFE_TYPE::CONSTANT) {
      }
      if (!returnUsed && retType != DIFFE_TYPE::CONSTANT && !hasconstant) {
        FunctionType *FTy = FunctionType::get(
            todiff->getReturnType(), foundcalled->getFunctionType()->params(),
            foundcalled->getFunctionType()->isVarArg());
        Function *NewF = Function::Create(
            FTy, Function::LinkageTypes::InternalLinkage,
            "fixderivative_" + todiff->getName(), todiff->getParent());
        for (auto pair : llvm::zip(NewF->args(), foundcalled->args())) {
          std::get<0>(pair).setName(std::get<1>(pair).getName());
        }

        BasicBlock *BB = BasicBlock::Create(NewF->getContext(), "entry", NewF);
        IRBuilder<> bb(BB);
        SmallVector<Value *, 2> args;
        for (auto &a : NewF->args())
          args.push_back(&a);
        auto cal = bb.CreateCall(foundcalled, args);
        cal->setCallingConv(foundcalled->getCallingConv());

        bb.CreateRet(bb.CreateExtractValue(cal, 1));
        return ForwardCachedFunctions[tup] = NewF;
      }
      assert(returnUsed);
    }

    SmallVector<Type *, 2> curTypes;
    bool legal = true;
    SmallVector<DIFFE_TYPE, 4> nextConstantArgs;
    for (auto tup : llvm::zip(todiff->args(), constant_args)) {
      auto &arg = std::get<0>(tup);
      curTypes.push_back(arg.getType());
      if (std::get<1>(tup) != DIFFE_TYPE::CONSTANT) {
        curTypes.push_back(arg.getType());
        nextConstantArgs.push_back(std::get<1>(tup));
        continue;
      }
      auto TT = oldTypeInfo.Arguments.find(&arg)->second[{-1}];
      if (TT.isFloat()) {
        nextConstantArgs.push_back(DIFFE_TYPE::DUP_ARG);
        continue;
      } else if (TT == BaseType::Integer) {
        nextConstantArgs.push_back(DIFFE_TYPE::DUP_ARG);
        continue;
      } else {
        legal = false;
        break;
      }
    }
    if (augmenteddata && augmenteddata->returns.find(AugmentedStruct::Tape) !=
                             augmenteddata->returns.end()) {
      assert(additionalArg);
      curTypes.push_back(additionalArg);
    }
    if (legal) {
      Type *RT = todiff->getReturnType();
      if (returnUsed && retType != DIFFE_TYPE::CONSTANT) {
        RT = StructType::get(RT->getContext(), {RT, RT});
      }
      if (!returnUsed && retType == DIFFE_TYPE::CONSTANT) {
        RT = Type::getVoidTy(RT->getContext());
      }

      FunctionType *FTy = FunctionType::get(
          RT, curTypes, todiff->getFunctionType()->isVarArg());

      Function *NewF = Function::Create(
          FTy, Function::LinkageTypes::InternalLinkage,
          "fixderivative_" + todiff->getName(), todiff->getParent());

      auto foundArg = NewF->arg_begin();
      SmallVector<Value *, 2> nextArgs;
      for (auto tup : llvm::zip(todiff->args(), constant_args)) {
        nextArgs.push_back(foundArg);
        auto &arg = std::get<0>(tup);
        foundArg->setName(arg.getName());
        foundArg++;
        if (std::get<1>(tup) != DIFFE_TYPE::CONSTANT) {
          foundArg->setName(arg.getName() + "'");
          nextConstantArgs.push_back(std::get<1>(tup));
          nextArgs.push_back(foundArg);
          foundArg++;
          continue;
        }
        auto TT = oldTypeInfo.Arguments.find(&arg)->second[{-1}];
        if (TT.isFloat()) {
          nextArgs.push_back(Constant::getNullValue(arg.getType()));
          nextConstantArgs.push_back(DIFFE_TYPE::DUP_ARG);
          continue;
        } else if (TT == BaseType::Integer) {
          nextArgs.push_back(nextArgs.back());
          nextConstantArgs.push_back(DIFFE_TYPE::DUP_ARG);
          continue;
        } else {
          legal = false;
          break;
        }
      }
      if (augmenteddata && augmenteddata->returns.find(AugmentedStruct::Tape) !=
                               augmenteddata->returns.end()) {
        foundArg->setName("tapeArg");
        nextArgs.push_back(foundArg);
        foundArg++;
      }

      BasicBlock *BB = BasicBlock::Create(NewF->getContext(), "entry", NewF);
      IRBuilder<> bb(BB);
      auto cal = bb.CreateCall(foundcalled, nextArgs);
      cal->setCallingConv(foundcalled->getCallingConv());

      if (returnUsed && retType != DIFFE_TYPE::CONSTANT) {
        bb.CreateRet(cal);
      } else if (returnUsed) {
        bb.CreateRet(bb.CreateExtractValue(cal, 0));
      } else if (retType != DIFFE_TYPE::CONSTANT) {
        bb.CreateRet(bb.CreateExtractValue(cal, 1));
      } else {
        bb.CreateRetVoid();
      }

      return ForwardCachedFunctions[tup] = NewF;
    }

    EmitWarning("NoCustom", todiff->getEntryBlock().begin()->getDebugLoc(),
                todiff, &todiff->getEntryBlock(),
                "Cannot use provided custom derivative pass");
  }
  if (todiff->empty() && CustomErrorHandler) {
    std::string s;
    llvm::raw_string_ostream ss(s);
    ss << "No forward derivative found for " + todiff->getName() << "\n";
    ss << *todiff << "\n";
    CustomErrorHandler(s.c_str(), wrap(todiff), ErrorType::NoDerivative,
                       nullptr);
  }
  if (todiff->empty())
    llvm::errs() << *todiff << "\n";
  assert(!todiff->empty());

  bool retActive = retType != DIFFE_TYPE::CONSTANT;

  ReturnType retVal =
      returnUsed ? (retActive ? ReturnType::TwoReturns : ReturnType::Return)
                 : (retActive ? ReturnType::Return : ReturnType::Void);

  bool diffeReturnArg = false;

  DiffeGradientUtils *gutils = DiffeGradientUtils::CreateFromClone(
      *this, mode, width, todiff, TLI, TA, oldTypeInfo, retType, diffeReturnArg,
      constant_args, retVal, additionalArg, omp);

  insert_or_assign2<ForwardCacheKey, Function *>(ForwardCachedFunctions, tup,
                                                 gutils->newFunc);

  gutils->FreeMemory = freeMemory;

  const SmallPtrSet<BasicBlock *, 4> guaranteedUnreachable =
      getGuaranteedUnreachable(gutils->oldFunc);

  gutils->forceActiveDetection();
  gutils->forceAugmentedReturns(guaranteedUnreachable);

  // TODO populate with actual unnecessaryInstructions once the dependency
  // cycle with activity analysis is removed
  SmallPtrSet<const Instruction *, 4> unnecessaryInstructionsTmp;
  for (auto BB : guaranteedUnreachable) {
    for (auto &I : *BB)
      unnecessaryInstructionsTmp.insert(&I);
  }
  if (mode == DerivativeMode::ForwardModeSplit)
    gutils->computeGuaranteedFrees(guaranteedUnreachable);

  SmallPtrSet<const Value *, 4> unnecessaryValues;
  SmallPtrSet<const Instruction *, 4> unnecessaryInstructions;
  calculateUnusedValuesInFunction(
      *gutils->oldFunc, unnecessaryValues, unnecessaryInstructions, returnUsed,
      mode, gutils, TLI, constant_args, guaranteedUnreachable);

  SmallPtrSet<const Instruction *, 4> unnecessaryStores;
  calculateUnusedStoresInFunction(*gutils->oldFunc, unnecessaryStores,
                                  unnecessaryInstructions, gutils, TLI);

  AdjointGenerator<const AugmentedReturn *> *maker;

  std::unique_ptr<const std::map<Instruction *, bool>> can_modref_map;
  if (mode == DerivativeMode::ForwardModeSplit) {
    std::map<Argument *, bool> _uncacheable_argsPP;
    {
      auto in_arg = todiff->arg_begin();
      auto pp_arg = gutils->oldFunc->arg_begin();
      for (; pp_arg != gutils->oldFunc->arg_end();) {
        _uncacheable_argsPP[pp_arg] = _uncacheable_args.find(in_arg)->second;
        ++pp_arg;
        ++in_arg;
      }
    }

    gutils->computeGuaranteedFrees(guaranteedUnreachable);
    CacheAnalysis CA(
        gutils->allocationsWithGuaranteedFree,
        gutils->rematerializableAllocations, gutils->TR, gutils->OrigAA,
        gutils->oldFunc,
        PPC.FAM.getResult<ScalarEvolutionAnalysis>(*gutils->oldFunc),
        gutils->OrigLI, gutils->OrigDT, TLI, unnecessaryInstructionsTmp,
        _uncacheable_argsPP, mode, omp);
    const std::map<CallInst *, const std::map<Argument *, bool>>
        uncacheable_args_map = CA.compute_uncacheable_args_for_callsites();
    can_modref_map = std::make_unique<const std::map<Instruction *, bool>>(
        CA.compute_uncacheable_load_map());
    gutils->can_modref_map = can_modref_map.get();

    auto getIndex = [&](Instruction *I, CacheType u) -> unsigned {
      assert(augmenteddata);
      return gutils->getIndex(std::make_pair(I, u), augmenteddata->tapeIndices);
    };

    gutils->computeMinCache(guaranteedUnreachable);

    maker = new AdjointGenerator<const AugmentedReturn *>(
        mode, gutils, constant_args, retType, getIndex, uncacheable_args_map,
        /*returnuses*/ nullptr, augmenteddata, nullptr, unnecessaryValues,
        unnecessaryInstructions, unnecessaryStores, guaranteedUnreachable,
        nullptr);

    if (additionalArg) {
      auto v = gutils->newFunc->arg_end();
      v--;
      Value *additionalValue = v;
      assert(augmenteddata);

      // TODO VERIFY THIS
      if (augmenteddata->tapeType &&
          augmenteddata->tapeType != additionalValue->getType()) {
        IRBuilder<> BuilderZ(gutils->inversionAllocs);
        if (!augmenteddata->tapeType->isEmptyTy()) {
          auto tapep = BuilderZ.CreatePointerCast(
              additionalValue, PointerType::getUnqual(augmenteddata->tapeType));
#if LLVM_VERSION_MAJOR > 7
          LoadInst *truetape =
              BuilderZ.CreateLoad(augmenteddata->tapeType, tapep, "truetape");
#else
          LoadInst *truetape = BuilderZ.CreateLoad(tapep, "truetape");
#endif
          truetape->setMetadata("enzyme_mustcache",
                                MDNode::get(truetape->getContext(), {}));

          if (!omp && gutils->FreeMemory) {
            CallInst *ci =
                cast<CallInst>(CallInst::CreateFree(additionalValue, truetape));
            ci->moveAfter(truetape);
            ci->addAttribute(AttributeList::FirstArgIndex, Attribute::NonNull);
          }
          additionalValue = truetape;
        } else {
          if (gutils->FreeMemory) {
            auto size = gutils->newFunc->getParent()
                            ->getDataLayout()
                            .getTypeAllocSizeInBits(augmenteddata->tapeType);
            if (size != 0) {
              CallInst *ci = cast<CallInst>(CallInst::CreateFree(
                  additionalValue, gutils->inversionAllocs));
              ci->addAttribute(AttributeList::FirstArgIndex,
                               Attribute::NonNull);
              BuilderZ.Insert(ci);
            }
          }
          additionalValue = UndefValue::get(augmenteddata->tapeType);
        }
      }

      // TODO here finish up making recursive structs simply pass in i8*
      gutils->setTape(additionalValue);
    }
  } else {

    maker = new AdjointGenerator<const AugmentedReturn *>(
        mode, gutils, constant_args, retType, nullptr, {},
        /*returnuses*/ nullptr, nullptr, nullptr, unnecessaryValues,
        unnecessaryInstructions, unnecessaryStores, guaranteedUnreachable,
        nullptr);
  }

  for (BasicBlock &oBB : *gutils->oldFunc) {
    // Don't create derivatives for code that results in termination
    if (guaranteedUnreachable.find(&oBB) != guaranteedUnreachable.end()) {
      auto newBB = cast<BasicBlock>(gutils->getNewFromOriginal(&oBB));
      SmallVector<BasicBlock *, 4> toRemove;
      if (auto II = dyn_cast<InvokeInst>(oBB.getTerminator())) {
        toRemove.push_back(
            cast<BasicBlock>(gutils->getNewFromOriginal(II->getNormalDest())));
      } else {
        for (auto next : successors(&oBB)) {
          auto sucBB = cast<BasicBlock>(gutils->getNewFromOriginal(next));
          toRemove.push_back(sucBB);
        }
      }

      for (auto sucBB : toRemove) {
        sucBB->removePredecessor(newBB);
      }

      SmallVector<Instruction *, 4> toerase;
      for (auto &I : oBB) {
        toerase.push_back(&I);
      }
      for (auto I : toerase) {
        maker->eraseIfUnused(*I, /*erase*/ true, /*check*/ true);
      }
      if (newBB->getTerminator())
        newBB->getTerminator()->eraseFromParent();
      IRBuilder<> builder(newBB);
      builder.CreateUnreachable();
      continue;
    }

    auto term = oBB.getTerminator();
    assert(term);
    if (!isa<ReturnInst>(term) && !isa<BranchInst>(term) &&
        !isa<SwitchInst>(term)) {
      llvm::errs() << *oBB.getParent() << "\n";
      llvm::errs() << "unknown terminator instance " << *term << "\n";
      assert(0 && "unknown terminator inst");
    }

    auto first = oBB.begin();
    auto last = oBB.empty() ? oBB.end() : std::prev(oBB.end());
    for (auto it = first; it != last; ++it) {
      maker->visit(&*it);
    }

    createTerminator(gutils, &oBB, retType, retVal);
  }

  if (mode == DerivativeMode::ForwardModeSplit && augmenteddata)
    restoreCache(gutils, augmenteddata->tapeIndices, guaranteedUnreachable);

  gutils->eraseFictiousPHIs();

  BasicBlock *entry = &gutils->newFunc->getEntryBlock();

  auto Arch =
      llvm::Triple(gutils->newFunc->getParent()->getTargetTriple()).getArch();

  cleanupInversionAllocs(gutils, entry);
  clearFunctionAttributes(gutils->newFunc);

  if (llvm::verifyFunction(*gutils->newFunc, &llvm::errs())) {
    llvm::errs() << *gutils->oldFunc << "\n";
    llvm::errs() << *gutils->newFunc << "\n";
    report_fatal_error("function failed verification (4)");
  }

  auto nf = gutils->newFunc;
  delete gutils;
  delete maker;

  {
    PreservedAnalyses PA;
    PPC.FAM.invalidate(*nf, PA);
  }
  PPC.AlwaysInline(nf);
  if (Arch == Triple::nvptx || Arch == Triple::nvptx64)
    PPC.ReplaceReallocs(nf, /*mem2reg*/ true);

  if (PostOpt)
    PPC.optimizeIntermediate(nf);
  if (EnzymePrint) {
    llvm::errs() << *nf << "\n";
  }
  return nf;
}

llvm::Function *EnzymeLogic::CreateBatch(Function *tobatch, unsigned width,
                                         ArrayRef<BATCH_TYPE> arg_types,
                                         BATCH_TYPE ret_type) {

  BatchCacheKey tup = std::make_tuple(tobatch, width, arg_types, ret_type);
  if (BatchCachedFunctions.find(tup) != BatchCachedFunctions.end()) {
    return BatchCachedFunctions.find(tup)->second;
  }

  FunctionType *orig_FTy = tobatch->getFunctionType();
  SmallVector<Type *, 4> params;
  unsigned long numVecParams =
      std::count(arg_types.begin(), arg_types.end(), BATCH_TYPE::VECTOR);

  for (unsigned i = 0; i < orig_FTy->getNumParams(); ++i) {
    if (arg_types[i] == BATCH_TYPE::VECTOR) {
      Type *ty = GradientUtils::getShadowType(orig_FTy->getParamType(i), width);
      params.push_back(ty);
    } else {
      params.push_back(orig_FTy->getParamType(i));
    }
  }

  Type *NewTy = GradientUtils::getShadowType(tobatch->getReturnType(), width);

  FunctionType *FTy = FunctionType::get(NewTy, params, tobatch->isVarArg());
  Function *NewF =
      Function::Create(FTy, tobatch->getLinkage(),
                       "batch_" + tobatch->getName(), tobatch->getParent());

  NewF->setLinkage(Function::LinkageTypes::InternalLinkage);

  ValueToValueMapTy originalToNewFn;

  // Create placeholder for the old arguments
  BasicBlock *placeholderBB =
      BasicBlock::Create(NewF->getContext(), "placeholders", NewF);

  IRBuilder<> PlaceholderBuilder(placeholderBB);
  PlaceholderBuilder.SetCurrentDebugLocation(DebugLoc());
  ValueToValueMapTy vmap;
  auto DestArg = NewF->arg_begin();
  auto SrcArg = tobatch->arg_begin();

  for (unsigned i = 0; i < orig_FTy->getNumParams(); ++i) {
    Argument *arg = SrcArg;
    if (arg_types[i] == BATCH_TYPE::VECTOR) {
      auto placeholder = PlaceholderBuilder.CreatePHI(
          arg->getType(), 0, "placeholder." + arg->getName());
      vmap[arg] = placeholder;
    } else {
      vmap[arg] = DestArg;
    }
    DestArg->setName(arg->getName());
    DestArg++;
    SrcArg++;
  }

  SmallVector<ReturnInst *, 4> Returns;
#if LLVM_VERSION_MAJOR >= 13
  CloneFunctionInto(NewF, tobatch, vmap,
                    CloneFunctionChangeType::LocalChangesOnly, Returns, "",
                    nullptr);
#else
  CloneFunctionInto(NewF, tobatch, vmap, true, Returns, "", nullptr);
#endif

  NewF->setLinkage(Function::LinkageTypes::InternalLinkage);

  // find instructions to vectorize (going up / overestimation)
  SmallPtrSet<Value *, 32> toVectorize;
  SetVector<llvm::Value *, std::deque<llvm::Value *>> refinelist;

  for (unsigned i = 0; i < tobatch->getFunctionType()->getNumParams(); i++) {
    if (arg_types[i] == BATCH_TYPE::VECTOR) {
      Argument *arg = tobatch->arg_begin() + i;
      toVectorize.insert(arg);
    }
  }

  for (auto &BB : *tobatch)
    for (auto &Inst : BB) {
      toVectorize.insert(&Inst);
      refinelist.insert(&Inst);
    }

  // find scalar instructions
  while (!refinelist.empty()) {
    Value *todo = *refinelist.begin();
    refinelist.erase(refinelist.begin());

    if (isa<ReturnInst>(todo) && ret_type == BATCH_TYPE::VECTOR)
      continue;

    if (auto branch_inst = dyn_cast<BranchInst>(todo)) {
      if (!branch_inst->isConditional()) {
        toVectorize.erase(todo);
        continue;
      }
    }

    if (auto call_inst = dyn_cast<CallInst>(todo)) {
      if (call_inst->getFunctionType()->isVoidTy() &&
          call_inst->getFunctionType()->getNumParams() == 0)
        toVectorize.erase(todo);
      continue;
    }

    if (auto todo_inst = dyn_cast<Instruction>(todo)) {

      if (todo_inst->mayReadOrWriteMemory())
        continue;

      if (isa<AllocaInst>(todo_inst))
        continue;

      SetVector<llvm::Value *, std::deque<llvm::Value *>> toCheck;
      toCheck.insert(todo_inst->op_begin(), todo_inst->op_end());
      SmallPtrSet<Value *, 8> safe;
      bool legal = true;
      while (!toCheck.empty()) {
        Value *cur = *toCheck.begin();
        toCheck.erase(toCheck.begin());

        if (!std::get<1>(safe.insert(cur)))
          continue;

        if (toVectorize.count(cur) == 0)
          continue;

        if (Instruction *cur_inst = dyn_cast<Instruction>(cur)) {
          if (!isa<CallInst>(cur_inst) && !cur_inst->mayReadOrWriteMemory()) {
            for (auto &op : todo_inst->operands())
              toCheck.insert(op);
            continue;
          }
        }

        legal = false;
        break;
      }

      if (legal)
        if (toVectorize.erase(todo))
          for (auto user : todo_inst->users())
            refinelist.insert(user);
    }
  }

  // unwrap arguments
  ValueMap<const Value *, std::vector<Value *>> vectorizedValues;
  auto entry = std::next(NewF->begin());
  IRBuilder<> Builder2(entry->getFirstNonPHI());
  Builder2.SetCurrentDebugLocation(DebugLoc());
  for (unsigned i = 0; i < FTy->getNumParams(); ++i) {
    Argument *orig_arg = tobatch->arg_begin() + i;
    Argument *arg = NewF->arg_begin() + i;

    if (arg_types[i] == BATCH_TYPE::SCALAR) {
      originalToNewFn[tobatch->arg_begin() + i] = arg;
      continue;
    }

    Instruction *placeholder = cast<Instruction>(vmap[orig_arg]);

    for (unsigned j = 0; j < width; ++j) {
      ExtractValueInst *argVecElem =
          cast<ExtractValueInst>(Builder2.CreateExtractValue(
              arg, {j},
              "unwrap" + (orig_arg->hasName()
                              ? "." + orig_arg->getName() + Twine(j)
                              : "")));
      if (j == 0) {
        placeholder->replaceAllUsesWith(argVecElem);
        placeholder->eraseFromParent();
      }
      vectorizedValues[orig_arg].push_back(argVecElem);
    }
  }

  placeholderBB->eraseFromParent();

  // update mapping with cloned basic blocks
  for (auto i = tobatch->begin(), j = NewF->begin();
       i != tobatch->end() && j != NewF->end(); ++i, ++j) {
    originalToNewFn[&*i] = &*j;
  }

  // update mapping with cloned scalar values and the first vectorized values
  auto J = inst_begin(NewF);
  // skip the unwrapped vector params
  std::advance(J, width * numVecParams);
  for (auto I = inst_begin(tobatch);
       I != inst_end(tobatch) && J != inst_end(NewF); ++I) {
    if (toVectorize.count(&*I) != 0) {
      vectorizedValues[&*I].push_back(&*J);
      ++J;
    } else {
      originalToNewFn[&*I] = &*J;
      ++J;
    }
  }

  // create placeholders for vector instructions 1..<n
  for (BasicBlock &BB : *tobatch) {
    for (Instruction &I : BB) {
      if (I.getType()->isVoidTy())
        continue;

      auto found = vectorizedValues.find(&I);
      if (found != vectorizedValues.end()) {
        Instruction *new_val_1 = cast<Instruction>(found->second.front());
        if (I.hasName())
          new_val_1->setName(I.getName() + "0");
        Instruction *insertPoint =
            new_val_1->getNextNode() ? new_val_1->getNextNode() : new_val_1;
        IRBuilder<> Builder2(insertPoint);
        Builder2.SetCurrentDebugLocation(DebugLoc());
        for (unsigned i = 1; i < width; ++i) {
          PHINode *placeholder = Builder2.CreatePHI(I.getType(), 0);
          vectorizedValues[&I].push_back(placeholder);
          if (I.hasName())
            placeholder->setName("placeholder." + I.getName() + Twine(i));
        }
      }
    }
  }

  InstructionBatcher *batcher =
      new InstructionBatcher(tobatch, NewF, width, vectorizedValues,
                             originalToNewFn, toVectorize, *this);

  for (auto val : toVectorize) {
    if (auto inst = dyn_cast<Instruction>(val))
      batcher->visit(inst);
  }

  if (llvm::verifyFunction(*NewF, &llvm::errs())) {
    llvm::errs() << *tobatch << "\n";
    llvm::errs() << *NewF << "\n";
    report_fatal_error("function failed verification (4)");
  }

  delete batcher;

  return NewF;
};

void EnzymeLogic::clear() {
  PPC.clear();
  AugmentedCachedFunctions.clear();
  AugmentedCachedFinished.clear();
  ReverseCachedFunctions.clear();
}
