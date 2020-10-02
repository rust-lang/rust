//===- EnzymeLogic.cpp - Implementation of forward and reverse pass generation//
//
//                             Enzyme Project
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// If using this code in an academic setting, please cite the following:
// @incollection{enzymeNeurips,
// title = {Instead of Rewriting Foreign Code for Machine Learning, Automatically Synthesize Fast Gradients},
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
// arguments and a bool `topLevel`. It creates a corresponding gradient
// function, computing the forward pass as well if at `topLevel`.
// CreateAugmentedPrimal takes similar arguments and creates an augmented
// forward pass.
//
//===----------------------------------------------------------------------===//
#include "AdjointGenerator.h"

#include "SCEV/ScalarEvolutionExpander.h"

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

#include "FunctionUtils.h"
#include "GradientUtils.h"
#include "LibraryFuncs.h"
#include "Utils.h"

using namespace llvm;

llvm::cl::opt<bool>
    enzyme_print("enzyme_print", cl::init(false), cl::Hidden,
                 cl::desc("Print before and after fns for autodiff"));

cl::opt<bool> looseTypeAnalysis("enzyme_loosetypes", cl::init(false),
                                cl::Hidden,
                                cl::desc("Allow looser use of types"));

cl::opt<bool> cache_reads_always("enzyme_always_cache_reads", cl::init(false),
                                 cl::Hidden,
                                 cl::desc("Force always caching of all reads"));

cl::opt<bool> cache_reads_never("enzyme_never_cache_reads", cl::init(false),
                                cl::Hidden,
                                cl::desc("Force never caching of all reads"));

cl::opt<bool> nonmarkedglobals_inactiveloads(
    "enzyme_nonmarkedglobals_inactiveloads", cl::init(true), cl::Hidden,
    cl::desc("Consider loads of nonmarked globals to be inactive"));

bool is_load_uncacheable(
    LoadInst &li, AAResults &AA, GradientUtils *gutils, TargetLibraryInfo &TLI,
    const SmallPtrSetImpl<const Instruction *> &unnecessaryInstructions,
    const std::map<Argument *, bool> &uncacheable_args);

bool is_value_mustcache_from_origin(
    Value *obj, AAResults &AA, GradientUtils *gutils, TargetLibraryInfo &TLI,
    const SmallPtrSetImpl<const Instruction *> &unnecessaryInstructions,
    const std::map<Argument *, bool> &uncacheable_args) {
  bool mustcache = false;

  // If the pointer operand is from an argument to the function, we need to
  // check if the argument
  //   received from the caller is uncacheable.
  if (isa<UndefValue>(obj)) {
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
    }
  } else {

    // Pointer operands originating from call instructions that are not
    // malloc/free are conservatively considered uncacheable.
    if (auto obj_op = dyn_cast<CallInst>(obj)) {
      Function *called = obj_op->getCalledFunction();
      #if LLVM_VERSION_MAJOR >= 11
      if (auto castinst = dyn_cast<ConstantExpr>(obj_op->getCalledOperand())) {
      #else
      if (auto castinst = dyn_cast<ConstantExpr>(obj_op->getCalledValue())) {
      #endif
        if (castinst->isCast()) {
          if (auto fn = dyn_cast<Function>(castinst->getOperand(0))) {
            if (isAllocationFunction(*fn, TLI) ||
                isDeallocationFunction(*fn, TLI)) {
              called = fn;
            }
          }
        }
      }
      if (called && isCertainMallocOrFree(called)) {
      } else {
        // OP is a non malloc/free call so we need to cache
        mustcache = true;
      }
    } else if (isa<AllocaInst>(obj)) {
      // No change to modref if alloca
    } else if (auto sli = dyn_cast<LoadInst>(obj)) {
      // If obj is from a load instruction conservatively consider it
      // uncacheable if that load itself cannot be cached
      mustcache = is_load_uncacheable(
          *sli, AA, gutils, TLI, unnecessaryInstructions, uncacheable_args);
    } else {
      // In absence of more information, assume that the underlying object for
      // pointer operand is uncacheable in caller.
      mustcache = true;
    }
  }
  return mustcache;
}

bool is_load_uncacheable(
    LoadInst &li, AAResults &AA, GradientUtils *gutils, TargetLibraryInfo &TLI,
    const SmallPtrSetImpl<const Instruction *> &unnecessaryInstructions,
    const std::map<Argument *, bool> &uncacheable_args) {
  assert(li.getParent()->getParent() == gutils->oldFunc);

  // Find the underlying object for the pointer operand of the load instruction.
  auto obj =
  #if LLVM_VERSION_MAJOR >= 12
      getUnderlyingObject(li.getPointerOperand(), 100);
  #else
      GetUnderlyingObject(li.getPointerOperand(),
                          gutils->oldFunc->getParent()->getDataLayout(), 100);
  #endif

  bool can_modref = is_value_mustcache_from_origin(
      obj, AA, gutils, TLI, unnecessaryInstructions, uncacheable_args);

  if (!can_modref) {
    allFollowersOf(&li, [&](Instruction *inst2) {
      // Don't consider modref from malloc/free as a need to cache
      if (auto obj_op = dyn_cast<CallInst>(inst2)) {
        Function *called = obj_op->getCalledFunction();
        #if LLVM_VERSION_MAJOR >= 11
        if (auto castinst = dyn_cast<ConstantExpr>(obj_op->getCalledOperand())) {
        #else
        if (auto castinst = dyn_cast<ConstantExpr>(obj_op->getCalledValue())) {
        #endif
          if (castinst->isCast()) {
            if (auto fn = dyn_cast<Function>(castinst->getOperand(0))) {
              if (isAllocationFunction(*fn, TLI) ||
                  isDeallocationFunction(*fn, TLI)) {
                called = fn;
              }
            }
          }
        }
        if (called && isCertainMallocOrFree(called)) {
          return false;
        }
      }

      if (unnecessaryInstructions.count(inst2)) {
        return false;
      }

      if (llvm::isModSet(AA.getModRefInfo(inst2, MemoryLocation::get(&li)))) {
        can_modref = true;
        // Early exit
        return true;
      }
      return false;
    });
  }

  return can_modref;
}

// Computes a map of LoadInst -> boolean for a function indicating whether that
// load is "uncacheable".
//   A load is considered "uncacheable" if the data at the loaded memory
//   location can be modified after the load instruction.
std::map<Instruction *, bool> compute_uncacheable_load_map(
    GradientUtils *gutils, AAResults &AA, TargetLibraryInfo &TLI,
    const SmallPtrSetImpl<const Instruction *> &unnecessaryInstructions,
    const std::map<Argument *, bool> uncacheable_args) {
  std::map<Instruction *, bool> can_modref_map;
  for (inst_iterator I = inst_begin(*gutils->oldFunc),
                     E = inst_end(*gutils->oldFunc);
       I != E; ++I) {
    Instruction *inst = &*I;
    // For each load instruction, determine if it is uncacheable.
    if (auto op = dyn_cast<LoadInst>(inst)) {
      can_modref_map[inst] = is_load_uncacheable(
          *op, AA, gutils, TLI, unnecessaryInstructions, uncacheable_args);
    }
  }
  return can_modref_map;
}

std::map<Argument *, bool> compute_uncacheable_args_for_one_callsite(
    CallInst *callsite_op, DominatorTree &DT, TargetLibraryInfo &TLI,
    const SmallPtrSetImpl<const Instruction *> &unnecessaryInstructions,
    AAResults &AA, GradientUtils *gutils,
    const std::map<Argument *, bool> parent_uncacheable_args) {

  if (!callsite_op->getCalledFunction())
    return {};

  std::vector<Value *> args;
  std::vector<bool> args_safe;

  // First, we need to propagate the uncacheable status from the parent function
  // to the callee.
  //   because memory location x modified after parent returns => x modified
  //   after callee returns.
  for (unsigned i = 0; i < callsite_op->getNumArgOperands(); ++i) {
    args.push_back(callsite_op->getArgOperand(i));

    // If the UnderlyingObject is from one of this function's arguments, then we
    // need to propagate the volatility.
    #if LLVM_VERSION_MAJOR >= 12
    Value *obj = getUnderlyingObject(
        callsite_op->getArgOperand(i), 100);
    #else
    Value *obj = GetUnderlyingObject(
        callsite_op->getArgOperand(i),
        callsite_op->getParent()->getModule()->getDataLayout(), 100);
    #endif

    bool init_safe = !is_value_mustcache_from_origin(
        obj, AA, gutils, TLI, unnecessaryInstructions, parent_uncacheable_args);
    args_safe.push_back(init_safe);
  }

  // Second, we check for memory modifications that can occur in the
  // continuation of the
  //   callee inside the parent function.
  allFollowersOf(callsite_op, [&](Instruction *inst2) {
    // Don't consider modref from malloc/free as a need to cache
    if (auto obj_op = dyn_cast<CallInst>(inst2)) {
      Function *called = obj_op->getCalledFunction();
      #if LLVM_VERSION_MAJOR >= 11
      if (auto castinst = dyn_cast<ConstantExpr>(obj_op->getCalledOperand())) {
      #else
      if (auto castinst = dyn_cast<ConstantExpr>(obj_op->getCalledValue())) {
      #endif
        if (castinst->isCast()) {
          if (auto fn = dyn_cast<Function>(castinst->getOperand(0))) {
            if (isAllocationFunction(*fn, TLI) ||
                isDeallocationFunction(*fn, TLI)) {
              called = fn;
            }
          }
        }
      }
      if (called && isCertainMallocOrFree(called)) {
        return false;
      }
    }

    if (unnecessaryInstructions.count(inst2))
      return false;

    for (unsigned i = 0; i < args.size(); ++i) {
      if (llvm::isModSet(AA.getModRefInfo(
              inst2, MemoryLocation::getForArgument(callsite_op, i, TLI)))) {
        args_safe[i] = false;
      }
    }
    return false;
  });

  std::map<Argument *, bool> uncacheable_args;

  auto arg = callsite_op->getCalledFunction()->arg_begin();
  for (unsigned i = 0; i < args.size(); ++i) {
    uncacheable_args[arg] = !args_safe[i];
    ++arg;
    if (arg == callsite_op->getCalledFunction()->arg_end()) {
      break;
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
compute_uncacheable_args_for_callsites(
    Function *F, DominatorTree &DT, TargetLibraryInfo &TLI,
    const SmallPtrSetImpl<const Instruction *> &unnecessaryInstructions,
    AAResults &AA, GradientUtils *gutils,
    const std::map<Argument *, bool> uncacheable_args) {
  std::map<CallInst *, const std::map<Argument *, bool>> uncacheable_args_map;

  for (inst_iterator I = inst_begin(*gutils->oldFunc),
                     E = inst_end(*gutils->oldFunc);
       I != E; ++I) {
    Instruction &inst = *I;
    if (auto op = dyn_cast<CallInst>(&inst)) {

      // We do not need uncacheable args for intrinsic functions. So skip such
      // callsites.
      if (isa<IntrinsicInst>(&inst)) {
        continue;
      }

      // For all other calls, we compute the uncacheable args for this callsite.
      uncacheable_args_map.insert(
          std::pair<CallInst *, const std::map<Argument *, bool>>(
              op, compute_uncacheable_args_for_one_callsite(
                      op, DT, TLI, unnecessaryInstructions, AA, gutils,
                      uncacheable_args)));
    }
  }
  return uncacheable_args_map;
}

void calculateUnusedValuesInFunction(Function& func, llvm::SmallPtrSetImpl<const Value*> &unnecessaryValues, llvm::SmallPtrSetImpl<const Instruction*> &unnecessaryInstructions,
                                      bool returnValue, DerivativeMode mode, TypeResults& TR, GradientUtils* gutils, TargetLibraryInfo& TLI, const std::vector<DIFFE_TYPE> &constant_args) {
  calculateUnusedValues(
      func, unnecessaryValues, unnecessaryInstructions, returnValue,
      [&](const Value *val) {
        return is_value_needed_in_reverse<Primal>(TR, gutils, val, /*topLevel*/mode == DerivativeMode::Both);
      },
      [&](const Instruction *inst) {
        if (auto II = dyn_cast<IntrinsicInst>(inst)) {
          if (II->getIntrinsicID() == Intrinsic::lifetime_start ||
              II->getIntrinsicID() == Intrinsic::lifetime_end ||
              II->getIntrinsicID() == Intrinsic::stacksave ||
              II->getIntrinsicID() == Intrinsic::stackrestore) {
            return false;
          }
        }

        if (auto obj_op = dyn_cast<CallInst>(inst)) {
          Function *called = obj_op->getCalledFunction();
          #if LLVM_VERSION_MAJOR >= 11
          if (auto castinst =
                  dyn_cast<ConstantExpr>(obj_op->getCalledOperand())) {
          #else
          if (auto castinst =
                  dyn_cast<ConstantExpr>(obj_op->getCalledValue())) {
          #endif
            if (castinst->isCast()) {
              if (auto fn = dyn_cast<Function>(castinst->getOperand(0))) {
                if (isDeallocationFunction(*fn, TLI)) {
                  return false;
                }
              }
            }
          }
          if (called && isDeallocationFunction(*called, TLI)) {
            return false;
          }
        }

        if (auto si = dyn_cast<StoreInst>(inst)) {
          if (isa<UndefValue>(si->getValueOperand()))
            return false;
          #if LLVM_VERSION_MAJOR >= 12
          auto at = getUnderlyingObject(
              si->getPointerOperand(), 100);
          #else
          auto at = GetUnderlyingObject(
              si->getPointerOperand(),
              gutils->oldFunc->getParent()->getDataLayout(), 100);
          #endif
          if (auto arg = dyn_cast<Argument>(at)) {
            if (constant_args[arg->getArgNo()] == DIFFE_TYPE::DUP_NONEED) {
              return false;
            }
          }
        }

        if (auto mti = dyn_cast<MemTransferInst>(inst)) {
          #if LLVM_VERSION_MAJOR >= 12
          auto at = getUnderlyingObject(
              mti->getArgOperand(1), 100);
          #else
          auto at = GetUnderlyingObject(
              mti->getArgOperand(1),
              gutils->oldFunc->getParent()->getDataLayout(), 100);
          #endif
          if (auto arg = dyn_cast<Argument>(at)) {
            if (constant_args[arg->getArgNo()] == DIFFE_TYPE::DUP_NONEED) {
              return false;
            }
          }
          if (auto ai = dyn_cast<AllocaInst>(at)) {
            bool foundStore = false;
            allInstructionsBetween(
                gutils->OrigLI, ai, const_cast<MemTransferInst *>(mti),
                [&](Instruction *I) -> bool {
                  if (!I->mayWriteToMemory())
                    return /*earlyBreak*/ false;
                  if (unnecessaryInstructions.count(I))
                    return /*earlyBreak*/ false;

                  if (writesToMemoryReadBy(
                          gutils->AA,
                          /*maybeReader*/ const_cast<MemTransferInst *>(mti),
                          /*maybeWriter*/ I)) {
                    foundStore = true;
                    return /*earlyBreak*/ true;
                  }
                  return /*earlyBreak*/ false;
                });
            if (!foundStore) {
              return false;
            }
          }
        }
        return ( (mode == DerivativeMode::Forward || mode == DerivativeMode::Both) && inst->mayWriteToMemory()) ||
               is_value_needed_in_reverse<Primal>(TR, gutils, inst,
                                          /*topLevel*/ mode == DerivativeMode::Both);
      });
}

void calculateUnusedStoresInFunction(Function& func, llvm::SmallPtrSetImpl<const Instruction*> &unnecessaryStores, const llvm::SmallPtrSetImpl<const Instruction*> &unnecessaryInstructions,
                                      GradientUtils* gutils) {
  calculateUnusedStores(
      func, unnecessaryStores, [&](const Instruction *inst) {
        if (auto si = dyn_cast<StoreInst>(inst)) {
          if (isa<UndefValue>(si->getValueOperand()))
            return false;
        }

        if (auto mti = dyn_cast<MemTransferInst>(inst)) {
          #if LLVM_VERSION_MAJOR >= 12
          auto at = getUnderlyingObject(
              mti->getArgOperand(1), 100);
          #else
          auto at = GetUnderlyingObject(
              mti->getArgOperand(1),
              func.getParent()->getDataLayout(), 100);
          #endif
          if (auto ai = dyn_cast<AllocaInst>(at)) {
            bool foundStore = false;
            allInstructionsBetween(
                gutils->OrigLI, ai, const_cast<MemTransferInst *>(mti),
                [&](Instruction *I) -> bool {
                  if (!I->mayWriteToMemory())
                    return /*earlyBreak*/ false;
                  if (unnecessaryInstructions.count(I))
                    return /*earlyBreak*/ false;

                  // if (I == &MTI) return;
                  if (writesToMemoryReadBy(
                          gutils->AA,
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
  // TODO CONSIDER a.getType()->isIntegerTy() && cast<IntegerType>(a.getType())->getBitWidth() < 16
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
    // TODO CONSIDER a.getType()->isIntegerTy() && cast<IntegerType>(a.getType())->getBitWidth() < 16

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

bool shouldAugmentCall(CallInst *op, const GradientUtils *gutils,
                                     TypeResults &TR) {
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
      TR.query(op).Data0()[{}].isPossiblePointer()) {
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

  for (unsigned i = 0; i < op->getNumArgOperands(); ++i) {
    if (gutils->isConstantValue(op->getArgOperand(i)) && called &&
        !called->empty()) {
      continue;
    }

    auto argType = op->getArgOperand(i)->getType();

    if (!argType->isFPOrFPVectorTy() &&
        !gutils->isConstantValue(op->getArgOperand(i)) &&
        TR.query(op->getArgOperand(i)).Data0()[{}].isPossiblePointer()) {
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
    llvm::errs() << "augunreachable op " << *op << "\n";
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
    std::vector<Instruction *> &postCreate,
    std::vector<Instruction *> &userReplace, GradientUtils *gutils,
    TypeResults &TR,
    const SmallPtrSetImpl<const Instruction *> &unnecessaryInstructions,
    const bool subretused) {
  Function *called = origop->getCalledFunction();
  #if LLVM_VERSION_MAJOR >= 11
  Value* calledValue = origop->getCalledOperand();
  #else
  Value* calledValue = origop->getCalledValue();
  #endif

  if (origop->getNumUses() != 0 && isa<PointerType>(origop->getType())) {
    if (called)
      llvm::errs()
          << " [not implemented] pointer return for combined forward/reverse "
          << called->getName() << "\n";
    else
      llvm::errs()
          << " [not implemented] pointer return for combined forward/reverse "
          << *calledValue << "\n";
    return false;
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
    if (auto ri = dyn_cast<ReturnInst>(I)) {
      auto find = replacedReturns.find(ri);
      if (find != replacedReturns.end()) {
        usetree.insert(ri);
      }
      return;
    }

    if (isa<BranchInst>(I) || isa<SwitchInst>(I)) {
      legal = false;
      if (called)
        llvm::errs() << " [bi] failed to replace function "
                     << (called->getName()) << " due to " << *I << "\n";
      else
        llvm::errs() << " [bi] ailed to replace function "
                     << (*calledValue) << " due to " << *I << "\n";
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
      Function *called = op->getCalledFunction();

      if (auto castinst = dyn_cast<ConstantExpr>(calledValue)) {
        if (castinst->isCast()) {
          if (auto fn = dyn_cast<Function>(castinst->getOperand(0))) {
            if (isAllocationFunction(*fn, gutils->TLI) ||
                isDeallocationFunction(*fn, gutils->TLI)) {
              return;
            }
          }
        }
      }
      if (called && isDeallocationFunction(*called, gutils->TLI))
        return;
    }

    if (isa<BranchInst>(I)) {
      legal = false;

      return;
    }
    if (isa<PHINode>(I)) {
      legal = false;
      if (called)
        llvm::errs() << " [phi] failed to replace function "
                     << (called->getName()) << " due to " << *I << "\n";
      else
        llvm::errs() << " [phi] ailed to replace function "
                     << (*calledValue) << " due to " << *I << "\n";
      return;
    }
    if (is_value_needed_in_reverse<Primal>(TR, gutils, I, /*topLevel*/ true)) {
      legal = false;
      if (called)
        llvm::errs() << " [nv] failed to replace function "
                     << (called->getName()) << " due to " << *I << "\n";
      else
        llvm::errs() << " [nv] ailed to replace function "
                     << (*calledValue) << " due to " << *I << "\n";
      return;
    }
    if (I != origop && !isa<IntrinsicInst>(I) && isa<CallInst>(I)) {
      legal = false;
      if (called)
        llvm::errs() << " [ci] failed to replace function "
                     << (called->getName()) << " due to " << *I << "\n";
      else
        llvm::errs() << " [ci] ailed to replace function "
                     << (*calledValue) << " due to " << *I << "\n";
      return;
    }
    // Do not try moving an instruction that modifies memory, if we already
    // moved it
    if (!isa<StoreInst>(I) || unnecessaryInstructions.count(I) == 0)
      if (I->mayReadOrWriteMemory() &&
          gutils->getNewFromOriginal(I)->getParent() !=
              gutils->getNewFromOriginal(I->getParent())) {
        legal = false;
        if (called)
          llvm::errs() << " [am] failed to replace function "
                       << (called->getName()) << " due to " << *I << "\n";
        else
          llvm::errs() << " [am] ailed to replace function "
                       << (*calledValue) << " due to " << *I
                       << "\n";
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
        if (writesToMemoryReadBy(gutils->AA, /*maybeReader*/ user,
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
      if (writesToMemoryReadBy(gutils->AA, /*maybeReader*/ inst,
                               /*maybeWriter*/ post)) {
        if (called)
          llvm::errs() << " failed to replace function " << (called->getName())
                       << " due to " << *post << " usetree: " << *inst << "\n";
        else
          llvm::errs() << " failed to replace function "
                       << (*calledValue) << " due to " << *post
                       << " usetree: " << *inst << "\n";
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
        if (called)
          llvm::errs() << " [nonspec] failed to replace function "
                       << (called->getName()) << " due to " << *inst << "\n";
        else
          llvm::errs() << " [nonspec] ailed to replace function "
                       << (*calledValue) << " due to " << *inst
                       << "\n";
        legal = false;
        // Early exit
        return true;
      }
    }
    if (isa<CallInst>(inst) &&
        gutils->originalToNewFn.find(inst) == gutils->originalToNewFn.end()) {
      legal = false;
      if (called)
        llvm::errs() << " [premove] failed to replace function "
                     << (called->getName()) << " due to " << *inst << "\n";
      else
        llvm::errs() << " [premove] ailed to replace function "
                     << (*calledValue) << " due to " << *inst
                     << "\n";
      // Early exit
      return true;
    }
    postCreate.push_back(gutils->getNewFromOriginal(inst));
    return false;
  });

  if (!legal)
    return false;

  if (called)
    llvm::errs() << " choosing to replace function " << (called->getName())
                 << " and do both forward/reverse\n";
  else
    llvm::errs() << " choosing to replace function "
                 << (*calledValue)
                 << " and do both forward/reverse\n";

  return true;
}

//! return structtype if recursive function
const AugmentedReturn &
CreateAugmentedPrimal(Function *todiff, DIFFE_TYPE retType,
                      const std::vector<DIFFE_TYPE> &constant_args,
                      TargetLibraryInfo &TLI, TypeAnalysis &TA,
                      AAResults &global_AA, bool returnUsed,
                      const FnTypeInfo &oldTypeInfo_,
                      const std::map<Argument *, bool> _uncacheable_args,
                      bool forceAnonymousTape) {
  if (returnUsed)
    assert(!todiff->getReturnType()->isEmptyTy() &&
           !todiff->getReturnType()->isVoidTy());
  if (retType != DIFFE_TYPE::CONSTANT)
    assert(!todiff->getReturnType()->isEmptyTy() &&
           !todiff->getReturnType()->isVoidTy());

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
      if (recursiveUse)
        pair.second.clear();
    }
  }

  using CacheKey = std::tuple<Function *, DIFFE_TYPE /*retType*/,
                             std::vector<DIFFE_TYPE> /*constant_args*/,
                             std::map<Argument *, bool> /*uncacheable_args*/,
                             bool /*returnUsed*/, const FnTypeInfo>;
  static std::map<CacheKey, AugmentedReturn>
      cachedfunctions;
  static std::map<CacheKey, bool>
      cachedfinished;
  CacheKey tup =
          std::make_tuple(todiff, retType, constant_args,
                          std::map<Argument *, bool>(_uncacheable_args.begin(),
                                                     _uncacheable_args.end()),
                          returnUsed, oldTypeInfo);
  auto found = cachedfunctions.find(tup);
  if (found != cachedfunctions.end()) {
    return found->second;
  }

  // TODO make default typing (not just constant)
  bool hasconstant = false;
  for (auto v : constant_args) {
    if (v == DIFFE_TYPE::CONSTANT) {
      hasconstant = true;
      break;
    }
  }

  if (!hasconstant && hasMetadata(todiff, "enzyme_augment")) {
    auto md = todiff->getMetadata("enzyme_augment");
    if (!isa<MDTuple>(md)) {
      llvm::errs() << *todiff << "\n";
      llvm::errs() << *md << "\n";
      report_fatal_error(
          "unknown augment for noninvertible function -- metadata incorrect");
    }
    std::map<AugmentedStruct, int> returnMapping;
    returnMapping[AugmentedStruct::Tape] = 0;
    returnMapping[AugmentedStruct::Return] = 1;
    returnMapping[AugmentedStruct::DifferentialReturn] = 2;

    auto md2 = cast<MDTuple>(md);
    assert(md2->getNumOperands() == 1);
    auto gvemd = cast<ConstantAsMetadata>(md2->getOperand(0));
    auto foundcalled = cast<Function>(gvemd->getValue());

    if (foundcalled->getReturnType() == todiff->getReturnType()) {
      FunctionType *FTy = FunctionType::get(
          StructType::get(todiff->getContext(),
                          {StructType::get(todiff->getContext(), {}),
                           foundcalled->getReturnType()}),
          foundcalled->getFunctionType()->params(),
          foundcalled->getFunctionType()->isVarArg());
      Function *NewF = Function::Create(
          FTy, Function::LinkageTypes::InternalLinkage,
          "fixaugmented_" + todiff->getName(), todiff->getParent());
      NewF->setAttributes(foundcalled->getAttributes());
      if (NewF->hasFnAttribute(Attribute::NoInline)) {
        NewF->removeFnAttr(Attribute::NoInline);
      }
      for (auto i = foundcalled->arg_begin(), j = NewF->arg_begin();
           i != foundcalled->arg_end();) {
        j->setName(i->getName());
        if (j->hasAttribute(Attribute::Returned))
          j->removeAttr(Attribute::Returned);
        if (j->hasAttribute(Attribute::StructRet))
          j->removeAttr(Attribute::StructRet);
        ++i;
        ++j;
      }
      BasicBlock *BB = BasicBlock::Create(NewF->getContext(), "entry", NewF);
      IRBuilder<> bb(BB);
      SmallVector<Value *, 4> args;
      for (auto &a : NewF->args())
        args.push_back(&a);
      auto cal = bb.CreateCall(foundcalled, args);
      cal->setCallingConv(foundcalled->getCallingConv());
      auto ut = UndefValue::get(NewF->getReturnType());
      auto val = bb.CreateInsertValue(ut, cal, {1u});
      bb.CreateRet(val);
      return insert_or_assign<CacheKey, AugmentedReturn>(
                 cachedfunctions, tup,
                 AugmentedReturn(NewF, nullptr, {}, returnMapping, {}, {}))
          ->second;
    }

    // assert(st->getNumElements() > 0);
    return insert_or_assign<CacheKey, AugmentedReturn>(
               cachedfunctions, tup,
               AugmentedReturn(foundcalled, nullptr, {}, returnMapping, {}, {}))
        ->second; // dyn_cast<StructType>(st->getElementType(0)));
  }

  if (todiff->empty()) {
    llvm::errs() << "mod: " << *todiff->getParent() << "\n";
    llvm::errs() << *todiff << "\n";
  }
  assert(!todiff->empty());
  std::map<AugmentedStruct, int> returnMapping;
  AAResults AA(TLI);
  // AA.addAAResult(global_AA);

  GradientUtils *gutils = GradientUtils::CreateFromClone(
      todiff, TLI, TA, AA, retType, constant_args, /*returnUsed*/ returnUsed,
      returnMapping);
  const SmallPtrSet<BasicBlock *, 4> guaranteedUnreachable =
      getGuaranteedUnreachable(gutils->oldFunc);

  gutils->forceContexts();

  FnTypeInfo typeInfo(gutils->oldFunc);
  {
    auto toarg = todiff->arg_begin();
    auto olarg = gutils->oldFunc->arg_begin();
    for (; toarg != todiff->arg_end(); ++toarg, ++olarg) {

      {
        auto fd = oldTypeInfo.Arguments.find(toarg);
        assert(fd != oldTypeInfo.Arguments.end());
        typeInfo.Arguments.insert(
            std::pair<Argument *, TypeTree>(olarg, fd->second));
      }

      {
        auto cfd = oldTypeInfo.KnownValues.find(toarg);
        assert(cfd != oldTypeInfo.KnownValues.end());
        typeInfo.KnownValues.insert(
            std::pair<Argument *, std::set<int64_t>>(olarg, cfd->second));
      }
    }
    typeInfo.Return = oldTypeInfo.Return;
  }
  TypeResults TR = TA.analyzeFunction(typeInfo);
  assert(TR.info.Function == gutils->oldFunc);
  gutils->forceActiveDetection(AA, TR);

  gutils->forceAugmentedReturns(TR, guaranteedUnreachable);

  // Convert uncacheable args from the input function to the preprocessed
  // function
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

  SmallPtrSet<const Value *, 4> unnecessaryValues;
  SmallPtrSet<const Instruction *, 4> unnecessaryInstructions;
  calculateUnusedValuesInFunction(*gutils->oldFunc, unnecessaryValues, unnecessaryInstructions,
                                      returnUsed, DerivativeMode::Forward, TR, gutils, TLI, constant_args);

  SmallPtrSet<const Instruction *, 4> unnecessaryStores;
  calculateUnusedStoresInFunction(*gutils->oldFunc, unnecessaryStores, unnecessaryInstructions, gutils);

  const std::map<CallInst *, const std::map<Argument *, bool>>
      uncacheable_args_map = compute_uncacheable_args_for_callsites(
          gutils->oldFunc, gutils->DT, TLI, unnecessaryInstructions, AA, gutils,
          _uncacheable_argsPP);

  const std::map<Instruction *, bool> can_modref_map =
      compute_uncacheable_load_map(gutils, AA, TLI, unnecessaryInstructions,
                                   _uncacheable_argsPP);

  insert_or_assign(cachedfunctions, tup,
                   AugmentedReturn(gutils->newFunc, nullptr, {}, returnMapping,
                                   uncacheable_args_map, can_modref_map));
  cachedfinished[tup] = false;

  auto getIndex = [&](Instruction *I, CacheType u) -> unsigned {
    // std::map<std::pair<Instruction*,std::string>,unsigned>& mapping =
    // cachedfunctions[tup].tapeIndices;
    return gutils->getIndex(std::make_pair(I, u),
                            cachedfunctions.find(tup)->second.tapeIndices);
  };
  gutils->can_modref_map = &can_modref_map;

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
      gutils->erase(ri);
    }
  }

  AdjointGenerator<AugmentedReturn *> maker(
      DerivativeMode::Forward, gutils, constant_args, TR, getIndex,
      uncacheable_args_map, &returnuses, &cachedfunctions.find(tup)->second,
      nullptr, unnecessaryValues, unnecessaryInstructions,
      unnecessaryStores, nullptr);

  for (BasicBlock &oBB : *gutils->oldFunc) {
    auto term = oBB.getTerminator();
    assert(term);

    // Don't create derivatives for code that results in termination
    if (guaranteedUnreachable.find(&oBB) != guaranteedUnreachable.end()) {
      std::vector<Instruction *> toerase;

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
    }

    BasicBlock::reverse_iterator I = oBB.rbegin(), E = oBB.rend();
    ++I;
    for (; I != E; ++I) {
      maker.visit(&*I);
      assert(oBB.rend() == E);
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

  if (auto bytes = gutils->newFunc->getDereferenceableBytes(
          llvm::AttributeList::ReturnIndex)) {
    AttrBuilder ab;
    ab.addDereferenceableAttr(bytes);
    gutils->newFunc->removeAttributes(llvm::AttributeList::ReturnIndex, ab);
  }

  // TODO could keep nonnull if returning value -1
  if (gutils->newFunc->getAttributes().getRetAlignment()) {
    AttrBuilder ab;
    ab.addAlignmentAttr(gutils->newFunc->getAttributes().getRetAlignment());
    gutils->newFunc->removeAttributes(llvm::AttributeList::ReturnIndex, ab);
  }
  if (gutils->newFunc->hasAttribute(llvm::AttributeList::ReturnIndex,
                                    llvm::Attribute::NoAlias)) {
    gutils->newFunc->removeAttribute(llvm::AttributeList::ReturnIndex,
                                     llvm::Attribute::NoAlias);
  }
  if (gutils->newFunc->hasAttribute(llvm::AttributeList::ReturnIndex,
                                    llvm::Attribute::NonNull)) {
    gutils->newFunc->removeAttribute(llvm::AttributeList::ReturnIndex,
                                     llvm::Attribute::NonNull);
  }
  if (gutils->newFunc->hasAttribute(llvm::AttributeList::ReturnIndex,
                                    llvm::Attribute::ZExt)) {
    gutils->newFunc->removeAttribute(llvm::AttributeList::ReturnIndex,
                                     llvm::Attribute::ZExt);
  }

  //! Keep track of inverted pointers we may need to return
  ValueToValueMapTy invertedRetPs;
  if (retType == DIFFE_TYPE::DUP_ARG || retType == DIFFE_TYPE::DUP_NONEED) {
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

  std::vector<Type *> RetTypes(
      cast<StructType>(gutils->newFunc->getReturnType())->elements());

  std::vector<Type *> MallocTypes;

  for (auto a : gutils->getTapeValues()) {
    MallocTypes.push_back(a->getType());
  }

  Type *tapeType = StructType::get(nf->getContext(), MallocTypes);

  bool removeTapeStruct = MallocTypes.size() == 1;
  if (removeTapeStruct) {
    tapeType = MallocTypes[0];

    for (auto &a : cachedfunctions.find(tup)->second.tapeIndices) {
      a.second = -1;
    }
  }

  bool recursive = cachedfunctions.find(tup)->second.fn->getNumUses() > 0 ||
                   forceAnonymousTape;
  bool noTape = MallocTypes.size() == 0 && !forceAnonymousTape;

  int oldretIdx = -1;
  if (returnMapping.find(AugmentedStruct::Return) != returnMapping.end()) {
    oldretIdx = returnMapping[AugmentedStruct::Return];
  }

  if (noTape) {
    auto tidx = returnMapping.find(AugmentedStruct::Tape)->second;
    returnMapping.erase(AugmentedStruct::Tape);
    cachedfunctions.find(tup)->second.returns.erase(AugmentedStruct::Tape);
    if (returnMapping.find(AugmentedStruct::Return) != returnMapping.end()) {
      cachedfunctions.find(tup)->second.returns[AugmentedStruct::Return] -=
          (returnMapping[AugmentedStruct::Return] > tidx) ? 1 : 0;
      returnMapping[AugmentedStruct::Return] -=
          (returnMapping[AugmentedStruct::Return] > tidx) ? 1 : 0;
    }
    if (returnMapping.find(AugmentedStruct::DifferentialReturn) !=
        returnMapping.end()) {
      cachedfunctions.find(tup)
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
    assert(noTape);

  bool removeStruct = RetTypes.size() == 1;

  if (removeStruct) {
    RetType = RetTypes[0];
    for (auto &a : returnMapping) {
      a.second = -1;
    }
    for (auto &a : cachedfunctions.find(tup)->second.returns) {
      a.second = -1;
    }
  }

  ValueToValueMapTy VMap;
  std::vector<Type *> ArgTypes;
  for (const Argument &I : nf->args()) {
    ArgTypes.push_back(I.getType());
  }

  // Create a new function type...
  FunctionType *FTy =
      FunctionType::get(RetType, ArgTypes, nf->getFunctionType()->isVarArg());

  // Create the new function...
  Function *NewF = Function::Create(
      FTy, nf->getLinkage(), "augmented_" + todiff->getName(), nf->getParent());

  unsigned ii = 0, jj = 0;
  for (auto i = nf->arg_begin(), j = NewF->arg_begin(); i != nf->arg_end();) {
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
  CloneFunctionInto(NewF, nf, VMap, nf->getSubprogram() != nullptr, Returns, "",
                    nullptr);

  IRBuilder<> ib(NewF->getEntryBlock().getFirstNonPHI());

  Value *ret = noReturn ? nullptr : ib.CreateAlloca(RetType);

  if (!noTape) {
    Value *tapeMemory;
    if (recursive) {
      auto i64 = Type::getInt64Ty(NewF->getContext());
      ConstantInt *size;
      tapeMemory = CallInst::CreateMalloc(
          NewF->getEntryBlock().getFirstNonPHI(), i64, tapeType,
          size = ConstantInt::get(
              i64, NewF->getParent()->getDataLayout().getTypeAllocSizeInBits(
                       tapeType) /
                       8),
          nullptr, nullptr, "tapemem");
      CallInst *malloccall = dyn_cast<CallInst>(tapeMemory);
      if (malloccall == nullptr) {
        malloccall =
            cast<CallInst>(cast<Instruction>(tapeMemory)->getOperand(0));
      }
      malloccall->addAttribute(AttributeList::ReturnIndex, Attribute::NoAlias);
      malloccall->addAttribute(AttributeList::ReturnIndex, Attribute::NonNull);
      malloccall->addDereferenceableAttr(llvm::AttributeList::ReturnIndex,
                                         size->getLimitedValue());
      malloccall->addDereferenceableOrNullAttr(llvm::AttributeList::ReturnIndex,
                                               size->getLimitedValue());
      std::vector<Value *> Idxs = {
          ib.getInt32(0),
          ib.getInt32(returnMapping.find(AugmentedStruct::Tape)->second),
      };
      assert(malloccall);
      assert(ret);
      Value *gep = ret;
      if (!removeStruct) {
        gep = ib.CreateGEP(ret, Idxs, "");
        cast<GetElementPtrInst>(gep)->setIsInBounds(true);
      }
      ib.CreateStore(malloccall, gep);
    } else {
      std::vector<Value *> Idxs = {
          ib.getInt32(0),
          ib.getInt32(returnMapping.find(AugmentedStruct::Tape)->second),
      };
      tapeMemory = ret;
      if (!removeStruct) {
        tapeMemory = ib.CreateGEP(ret, Idxs, "");
        cast<GetElementPtrInst>(tapeMemory)->setIsInBounds(true);
      }
    }

    unsigned i = 0;
    for (auto v : gutils->getTapeValues()) {
      if (!isa<UndefValue>(v)) {
        IRBuilder<> ib(cast<Instruction>(VMap[v])->getNextNode());
        std::vector<Value *> Idxs = {ib.getInt32(0), ib.getInt32(i)};
        Value *gep = tapeMemory;
        if (!removeTapeStruct) {
          gep = ib.CreateGEP(tapeMemory, Idxs, "");
          cast<GetElementPtrInst>(gep)->setIsInBounds(true);
        }
        ib.CreateStore(VMap[v], gep);
      }
      ++i;
    }
  }

  for (BasicBlock &BB : *nf) {
    if (auto ri = dyn_cast<ReturnInst>(BB.getTerminator())) {
      ReturnInst *rim = cast<ReturnInst>(VMap[ri]);
      IRBuilder<> ib(rim);
      if (returnUsed) {
        Value *rv = rim->getReturnValue();
        assert(rv);
        Value *actualrv = nullptr;
        if (auto iv = dyn_cast<InsertValueInst>(rv)) {
          if (iv->getNumIndices() == 1 &&
              (int)iv->getIndices()[0] == oldretIdx) {
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

      if (retType == DIFFE_TYPE::DUP_ARG || retType == DIFFE_TYPE::DUP_NONEED) {
        assert(invertedRetPs[ri]);
        if (!isa<UndefValue>(invertedRetPs[ri])) {
          assert(VMap[invertedRetPs[ri]]);
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
          ib.CreateStore(VMap[invertedRetPs[ri]], gep);
        }
      }
      if (noReturn)
        ib.CreateRetVoid();
      else
        ib.CreateRet(ib.CreateLoad(ret));
      gutils->erase(cast<Instruction>(VMap[ri]));
    }
  }

  for (Argument &Arg : NewF->args()) {
    if (Arg.hasAttribute(Attribute::Returned))
      Arg.removeAttr(Attribute::Returned);
    if (Arg.hasAttribute(Attribute::StructRet))
      Arg.removeAttr(Attribute::StructRet);
  }
  if (NewF->hasFnAttribute(Attribute::OptimizeNone))
    NewF->removeFnAttr(Attribute::OptimizeNone);

  if (auto bytes =
          NewF->getDereferenceableBytes(llvm::AttributeList::ReturnIndex)) {
    AttrBuilder ab;
    ab.addDereferenceableAttr(bytes);
    NewF->removeAttributes(llvm::AttributeList::ReturnIndex, ab);
  }
  if (NewF->hasAttribute(llvm::AttributeList::ReturnIndex,
                         llvm::Attribute::NoAlias)) {
    NewF->removeAttribute(llvm::AttributeList::ReturnIndex,
                          llvm::Attribute::NoAlias);
  }
  if (NewF->hasAttribute(llvm::AttributeList::ReturnIndex,
                         llvm::Attribute::ZExt)) {
    NewF->removeAttribute(llvm::AttributeList::ReturnIndex,
                          llvm::Attribute::ZExt);
  }

  if (llvm::verifyFunction(*NewF, &llvm::errs())) {
    llvm::errs() << *gutils->oldFunc << "\n";
    llvm::errs() << *NewF << "\n";
    report_fatal_error("augmented function failed verification (3)");
  }

  SmallVector<CallInst *, 4> fnusers;
  for (auto user : cachedfunctions.find(tup)->second.fn->users()) {
    fnusers.push_back(cast<CallInst>(user));
  }
  for (auto user : fnusers) {
    if (removeStruct) {
      IRBuilder<> B(user);
      auto n = user->getName().str();
      user->setName("");
      std::vector<Value *> args(user->arg_begin(), user->arg_end());
      auto rep = B.CreateCall(NewF, args);
      rep->copyIRFlags(user);
      rep->setAttributes(user->getAttributes());
      rep->setCallingConv(user->getCallingConv());
      rep->setTailCallKind(user->getTailCallKind());
      rep->setDebugLoc(user->getDebugLoc());
      assert(user);
      std::vector<ExtractValueInst*> torep;
      for (auto u : user->users()) {
        assert(u);
        if (auto ei = dyn_cast<ExtractValueInst>(u)) {
          torep.push_back(ei);
        }
      }
      for(auto ei : torep) {
        ei->replaceAllUsesWith(rep);
        ei->eraseFromParent();
      }
      user->eraseFromParent();
    } else {
      cast<CallInst>(user)->setCalledFunction(NewF);
    }
  }
  cachedfunctions.find(tup)->second.fn = NewF;
  if (recursive)
    cachedfunctions.find(tup)->second.tapeType = tapeType;
  insert_or_assign(cachedfinished, tup, true);

  gutils->newFunc->eraseFromParent();

  delete gutils;
  if (enzyme_print)
    llvm::errs() << *NewF << "\n";
  return cachedfunctions.find(tup)->second;
}

void createInvertedTerminator(TypeResults &TR, DiffeGradientUtils *gutils,
                              const std::vector<DIFFE_TYPE> &argTypes,
                              BasicBlock *oBB, AllocaInst *retAlloca,
                              AllocaInst *dretAlloca, unsigned extraArgs) {
  LoopContext loopContext;
  BasicBlock *BB = cast<BasicBlock>(gutils->getNewFromOriginal(oBB));
  bool inLoop = gutils->getContext(BB, loopContext);
  BasicBlock *BB2 = gutils->reverseBlocks[BB];
  assert(BB2);
  IRBuilder<> Builder(BB2);
  Builder.setFastMathFlags(getFast());

  std::map<BasicBlock *, std::vector<BasicBlock *>> targetToPreds;
  for (auto pred : predecessors(BB)) {
    targetToPreds[gutils->getReverseOrLatchMerge(pred, BB)].emplace_back(pred);
  }

  if (targetToPreds.size() == 0) {
    SmallVector<Value *, 4> retargs;

    if (retAlloca) {
      auto result = Builder.CreateLoad(retAlloca, "retreload");
      // TODO reintroduce invariant load/group
      // result->setMetadata(LLVMContext::MD_invariant_load,
      // MDNode::get(retAlloca->getContext(), {}));
      retargs.push_back(result);
    }

    if (dretAlloca) {
      auto result = Builder.CreateLoad(dretAlloca, "dretreload");
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
  std::vector<SelectInst *> selects;

  IRBuilder<> phibuilder(BB2);
  bool setphi = false;

  // Ensure phi values have their derivatives propagated
  for (auto I = oBB->begin(), E = oBB->end(); I != E; ++I) {
    if (PHINode *orig = dyn_cast<PHINode>(&*I)) {
      if (gutils->isConstantValue(orig))
        continue;
      auto PNtype = TR.intType(orig, /*necessary*/ false);

      // TODO remove explicit type check and only use PNtype
      if (PNtype == BaseType::Pointer || orig->getType()->isPointerTy())
        continue;

      auto prediff = gutils->diffe(orig, Builder);
      gutils->setDiffe(orig, Constant::getNullValue(orig->getType()), Builder);

      Type *PNfloatType = PNtype.isFloat();
      if (!PNfloatType)
        llvm::errs() << " for orig " << *orig << " saw "
                     << TR.intType(orig, /*necessary*/ false).str() << "\n";
      TR.intType(orig, /*necessary*/ true);
      assert(PNfloatType);

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
    } else
      break;
  }
  if (!setphi) {
    phibuilder.SetInsertPoint(Builder.GetInsertBlock(),
                              Builder.GetInsertPoint());
  }

  if (inLoop && BB == loopContext.header) {
    std::map<BasicBlock *, std::vector<BasicBlock *>> targetToPreds;
    for (auto pred : predecessors(BB)) {
      if (pred == loopContext.preheader)
        continue;
      targetToPreds[gutils->getReverseOrLatchMerge(pred, BB)].emplace_back(
          pred);
    }

    assert(targetToPreds.size() &&
           "only loops with one backedge are presently supported");

    Value *av = phibuilder.CreateLoad(loopContext.antivaralloc);
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

Function *CreatePrimalAndGradient(
    Function *todiff, DIFFE_TYPE retType,
    const std::vector<DIFFE_TYPE> &constant_args, TargetLibraryInfo &TLI,
    TypeAnalysis &TA, AAResults &global_AA, bool returnUsed, bool dretPtr,
    bool topLevel, llvm::Type *additionalArg, const FnTypeInfo &oldTypeInfo_,
    const std::map<Argument *, bool> _uncacheable_args,
    const AugmentedReturn *augmenteddata) {

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
      if (recursiveUse)
        pair.second.clear();
    }
  }

  if (retType != DIFFE_TYPE::CONSTANT)
    assert(!todiff->getReturnType()->isVoidTy());

  using CacheKey = std::tuple<Function *, DIFFE_TYPE /*retType*/,
                 std::vector<DIFFE_TYPE> /*constant_args*/,
                 std::map<Argument *, bool> /*uncacheable_args*/,
                 bool /*retval*/, bool /*dretPtr*/, bool /*topLevel*/,
                 llvm::Type *, const FnTypeInfo>;
  static std::map<CacheKey, Function *> cachedfunctions;
  CacheKey tup = std::make_tuple(
      todiff, retType, constant_args,
      std::map<Argument *, bool>(_uncacheable_args.begin(),
                                 _uncacheable_args.end()),
      returnUsed, dretPtr, topLevel, additionalArg, oldTypeInfo);
  if (cachedfunctions.find(tup) != cachedfunctions.end()) {
    return cachedfunctions.find(tup)->second;
  }

  // Whether we shuold actually return the value
  bool returnValue = returnUsed && topLevel;

  bool hasTape = false;

  // TODO change this to go by default function type assumptions
  bool hasconstant = false;
  for (auto v : constant_args) {
    if (v == DIFFE_TYPE::CONSTANT) {
      hasconstant = true;
      break;
    }
  }

  if (!hasconstant && !topLevel && !returnValue &&
      hasMetadata(todiff, "enzyme_gradient")) {

    auto md = todiff->getMetadata("enzyme_gradient");
    if (!isa<MDTuple>(md)) {
      llvm::errs() << *todiff << "\n";
      llvm::errs() << *md << "\n";
      report_fatal_error(
          "unknown gradient for noninvertible function -- metadata incorrect");
    }
    auto md2 = cast<MDTuple>(md);
    assert(md2->getNumOperands() == 1);
    auto gvemd = cast<ConstantAsMetadata>(md2->getOperand(0));
    auto foundcalled = cast<Function>(gvemd->getValue());

    DIFFE_TYPE subretType = todiff->getReturnType()->isFPOrFPVectorTy()
                                ? DIFFE_TYPE::OUT_DIFF
                                : DIFFE_TYPE::DUP_ARG;
    if (todiff->getReturnType()->isVoidTy() ||
        todiff->getReturnType()->isEmptyTy())
      subretType = DIFFE_TYPE::CONSTANT;
    auto res = getDefaultFunctionTypeForGradient(todiff->getFunctionType(),
                                                 /*retType*/ subretType);

    if (foundcalled->arg_size() == res.first.size() + 1 /*tape*/) {
      auto lastarg = foundcalled->arg_end();
      lastarg--;
      res.first.push_back(lastarg->getType());
      hasTape = true;
    } else if (foundcalled->arg_size() == res.first.size()) {
      res.first.push_back(StructType::get(todiff->getContext(), {}));
    } else {
      llvm::errs() << "expected args: [";
      for (auto a : res.first) {
        llvm::errs() << *a << " ";
      }
      llvm::errs() << "]\n";
      llvm::errs() << *foundcalled << "\n";
      assert(0 && "bad type for custom gradient");
    }

    auto st = dyn_cast<StructType>(foundcalled->getReturnType());
    bool wrongRet = st == nullptr;
    if (wrongRet || !hasTape) {
      FunctionType *FTy =
          FunctionType::get(StructType::get(todiff->getContext(), {res.second}),
                            res.first, todiff->getFunctionType()->isVarArg());
      Function *NewF = Function::Create(
          FTy, Function::LinkageTypes::InternalLinkage,
          "fixgradient_" + todiff->getName(), todiff->getParent());
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
      if (!hasTape) {
        args.pop_back();
      }
      llvm::errs() << *NewF << "\n";
      llvm::errs() << *foundcalled << "\n";
      auto cal = bb.CreateCall(foundcalled, args);
      cal->setCallingConv(foundcalled->getCallingConv());
      Value *val = cal;
      if (wrongRet) {
        auto ut = UndefValue::get(NewF->getReturnType());
        if (val->getType()->isEmptyTy() && res.second.size() == 0) {
          val = ut;
        } else if (res.second.size() == 1 && res.second[0] == val->getType()) {
          val = bb.CreateInsertValue(ut, cal, {0u});
        } else {
          llvm::errs() << *foundcalled << "\n";
          assert(0 && "illegal type for reverse");
        }
      }
      bb.CreateRet(val);
      foundcalled = NewF;
    }
    return insert_or_assign2<CacheKey, Function*>(cachedfunctions, tup, foundcalled)->second;
  }

  assert(!todiff->empty());
  auto M = todiff->getParent();

  AAResults AA(TLI);
  // AA.addAAResult(global_AA);
  DiffeGradientUtils *gutils = DiffeGradientUtils::CreateFromClone(
      topLevel, todiff, TLI, TA, AA, retType, constant_args,
      returnValue ? (dretPtr ? ReturnType::ArgsWithTwoReturns
                             : ReturnType::ArgsWithReturn)
                  : (dretPtr ? ReturnType::ArgsWithReturn
                             : ReturnType::Args),
      additionalArg);
  insert_or_assign2<CacheKey, Function*>(cachedfunctions, tup, gutils->newFunc);

  const SmallPtrSet<BasicBlock *, 4> guaranteedUnreachable =
      getGuaranteedUnreachable(gutils->oldFunc);

  SmallPtrSet<Value *, 4> assumeTrue;
  SmallPtrSet<Value *, 4> assumeFalse;

  if (!topLevel) {
    // TODO also can consider switch instance as well
    // TODO can also insert to topLevel as well [note this requires putting the
    // intrinsic at the correct location]
    for (auto &BB : *gutils->oldFunc) {
      std::vector<BasicBlock *> unreachables;
      std::vector<BasicBlock *> reachables;
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
        IRBuilder<> B(&gutils->newFunc->getEntryBlock().front());

        if (auto inst = dyn_cast<Instruction>(bi->getCondition())) {
          B.SetInsertPoint(gutils->getNewFromOriginal(inst)->getNextNode());
        }

        Value *vals[1] = {gutils->getNewFromOriginal(bi->getCondition())};
        if (bi->getSuccessor(0) == unreachables[0]) {
          assumeFalse.insert(vals[0]);
          vals[0] = B.CreateNot(vals[0]);
        } else {
          assumeTrue.insert(vals[0]);
        }
        B.CreateCall(Intrinsic::getDeclaration(M, Intrinsic::assume), vals);
      }
    }
  }

  gutils->forceContexts();

  FnTypeInfo typeInfo(gutils->oldFunc);
  {
    auto toarg = todiff->arg_begin();
    auto olarg = gutils->oldFunc->arg_begin();
    for (; toarg != todiff->arg_end(); ++toarg, ++olarg) {

      {
        auto fd = oldTypeInfo.Arguments.find(toarg);
        assert(fd != oldTypeInfo.Arguments.end());
        typeInfo.Arguments.insert(
            std::pair<Argument *, TypeTree>(olarg, fd->second));
      }

      {
        auto cfd = oldTypeInfo.KnownValues.find(toarg);
        assert(cfd != oldTypeInfo.KnownValues.end());
        typeInfo.KnownValues.insert(
            std::pair<Argument *, std::set<int64_t>>(olarg, cfd->second));
      }
    }
    typeInfo.Return = oldTypeInfo.Return;
  }

  TypeResults TR = TA.analyzeFunction(typeInfo);
  assert(TR.info.Function == gutils->oldFunc);

  gutils->forceActiveDetection(AA, TR);
  gutils->forceAugmentedReturns(TR, guaranteedUnreachable);

  std::map<std::pair<Instruction *, CacheType>, int> mapping;
  if (augmenteddata)
    mapping = augmenteddata->tapeIndices;

  auto getIndex = [&](Instruction *I, CacheType u) -> unsigned {
    return gutils->getIndex(std::make_pair(I, u), mapping);
  };

  // Convert uncacheable args from the input function to the preprocessed
  // function
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

  SmallPtrSet<const Value *, 4> unnecessaryValues;
  SmallPtrSet<const Instruction *, 4> unnecessaryInstructions;
  calculateUnusedValuesInFunction(*gutils->oldFunc, unnecessaryValues, unnecessaryInstructions,
                                    returnValue, topLevel ? DerivativeMode::Both : DerivativeMode::Reverse, TR, gutils, TLI, constant_args);


  SmallPtrSet<const Instruction *, 4> unnecessaryStores;
  calculateUnusedStoresInFunction(*gutils->oldFunc, unnecessaryStores, unnecessaryInstructions, gutils);

  const std::map<CallInst *, const std::map<Argument *, bool>>
      uncacheable_args_map =
          (augmenteddata)
              ? augmenteddata->uncacheable_args_map
              : compute_uncacheable_args_for_callsites(
                    gutils->oldFunc, gutils->DT, TLI, unnecessaryInstructions,
                    AA, gutils, _uncacheable_argsPP);

  const std::map<Instruction *, bool> can_modref_map =
      augmenteddata ? augmenteddata->can_modref_map
                    : compute_uncacheable_load_map(gutils, AA, TLI,
                                                   unnecessaryInstructions,
                                                   _uncacheable_argsPP);

  gutils->can_modref_map = &can_modref_map;

  Value *additionalValue = nullptr;
  if (additionalArg) {
    auto v = gutils->newFunc->arg_end();
    v--;
    additionalValue = v;
    assert(!topLevel);
    assert(augmenteddata);

    // TODO VERIFY THIS
    if (augmenteddata->tapeType &&
        augmenteddata->tapeType != additionalValue->getType()) {
      IRBuilder<> BuilderZ(gutils->inversionAllocs);
      // assert(PointerType::getUnqual(augmenteddata->tapeType) ==
      // additionalValue->getType()); auto tapep = additionalValue;
      auto tapep = BuilderZ.CreatePointerCast(
          additionalValue, PointerType::getUnqual(augmenteddata->tapeType));
      LoadInst *truetape = BuilderZ.CreateLoad(tapep);
      truetape->setMetadata("enzyme_mustcache",
                            MDNode::get(truetape->getContext(), {}));

      CallInst *ci = cast<CallInst>(CallInst::CreateFree(
          additionalValue, truetape)); //&*BuilderZ.GetInsertPoint()));
      ci->moveAfter(truetape);
      ci->addAttribute(AttributeList::FirstArgIndex, Attribute::NonNull);
      additionalValue = truetape;
    }

    // TODO here finish up making recursive structs simply pass in i8*
    gutils->setTape(additionalValue);
  }

  Argument *differetval = nullptr;
  if (retType == DIFFE_TYPE::OUT_DIFF) {
    auto endarg = gutils->newFunc->arg_end();
    endarg--;
    if (additionalArg)
      endarg--;
    differetval = endarg;
    if (differetval->getType() != todiff->getReturnType()) {
      llvm::errs() << *gutils->oldFunc << "\n";
      llvm::errs() << *gutils->newFunc << "\n";
    }
    assert(differetval->getType() == todiff->getReturnType());
  }

  // Explicitly handle all returns first to ensure that return instructions know
  // if they are used or not before
  //   processessing instructions
  std::map<ReturnInst *, StoreInst *> replacedReturns;
  llvm::AllocaInst *retAlloca = nullptr;
  llvm::AllocaInst *dretAlloca = nullptr;
  if (returnValue) {
    retAlloca = IRBuilder<>(&gutils->newFunc->getEntryBlock().front())
                    .CreateAlloca(todiff->getReturnType(), nullptr, "toreturn");
  }
  if (dretPtr) {
    assert(retType == DIFFE_TYPE::DUP_ARG || retType == DIFFE_TYPE::DUP_NONEED);
    assert(topLevel);
    dretAlloca =
        IRBuilder<>(&gutils->newFunc->getEntryBlock().front())
            .CreateAlloca(todiff->getReturnType(), nullptr, "dtoreturn");
  }
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

      if (dretAlloca && !gutils->isConstantValue(orig->getReturnValue())) {
        rb.CreateStore(gutils->invertPointerM(orig->getReturnValue(), rb),
                        dretAlloca);
      }

      if (retType == DIFFE_TYPE::OUT_DIFF) {
        assert(orig->getReturnValue());
        assert(differetval);
        if (!gutils->isConstantValue(orig->getReturnValue())) {
          IRBuilder<> reverseB(gutils->reverseBlocks[BB]);
          gutils->setDiffe(orig->getReturnValue(), differetval, reverseB);
        }
      } else {
        assert(retAlloca == nullptr);
      }

      rb.CreateBr(gutils->reverseBlocks[BB]);
      gutils->erase(op);
    }
  }

  AdjointGenerator<const AugmentedReturn *> maker(
      topLevel ? DerivativeMode::Both : DerivativeMode::Reverse, gutils,
      constant_args, TR, getIndex, uncacheable_args_map, /*returnuses*/ nullptr,
      augmenteddata, &replacedReturns, unnecessaryValues,
      unnecessaryInstructions, unnecessaryStores, dretAlloca);

  for (BasicBlock &oBB : *gutils->oldFunc) {
    // Don't create derivatives for code that results in termination
    if (guaranteedUnreachable.find(&oBB) != guaranteedUnreachable.end()) {
      auto newBB = cast<BasicBlock>(gutils->getNewFromOriginal(&oBB));
      std::vector<BasicBlock*> toRemove;
      for (auto next : successors(&oBB)) {
        auto sucBB = cast<BasicBlock>(gutils->getNewFromOriginal(next));
        toRemove.push_back(sucBB);
      }
      for(auto sucBB : toRemove) {
        sucBB->removePredecessor(newBB);
      }

      std::vector<Instruction *> toerase;
      for (auto &I : oBB) {
        toerase.push_back(&I);
      }
      for (auto I : toerase) {
        maker.eraseIfUnused(*I, /*erase*/ true, /*check*/ topLevel == true);
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
    createInvertedTerminator(TR, gutils, constant_args, &oBB, retAlloca,
                             dretAlloca,
                             0 + (additionalArg ? 1 : 0) +
                                 ((retType == DIFFE_TYPE::DUP_ARG ||
                                   retType == DIFFE_TYPE::DUP_NONEED)
                                      ? 1
                                      : 0));
  }

  gutils->eraseFictiousPHIs();

  for (auto val : assumeTrue) {
    bool changed;
    do {
      changed = false;
      for (auto &use : val->uses()) {
        assert(use.getUser());
        if (auto user = dyn_cast<IntrinsicInst>(use.getUser())) {
          if (user->getIntrinsicID() == Intrinsic::assume)
            continue;
        }
        use.set(ConstantInt::getTrue(val->getContext()));
        changed = true;
        break;
      }
    } while (!changed);
  }

  for (auto val : assumeFalse) {
    bool changed;
    do {
      changed = false;
      for (auto &use : val->uses()) {
        assert(use.getUser());
        if (auto notu = dyn_cast<BinaryOperator>(use.getUser())) {
          if (notu->getNumUses() == 1 &&
              notu->getOpcode() == BinaryOperator::Xor &&
              notu->getOperand(0) == val &&
              isa<ConstantInt>(notu->getOperand(1)) &&
              cast<ConstantInt>(notu->getOperand(1))->isOne()) {
            if (auto user = dyn_cast<IntrinsicInst>(*notu->user_begin())) {
              if (user->getIntrinsicID() == Intrinsic::assume) {
                continue;
              }
            }
          }
        }
        use.set(ConstantInt::getFalse(val->getContext()));
        changed = true;
        break;
      }
    } while (!changed);
  }

  while (gutils->inversionAllocs->size() > 0) {
    gutils->inversionAllocs->back().moveBefore(
        gutils->newFunc->getEntryBlock().getFirstNonPHIOrDbgOrLifetime());
  }

  (IRBuilder<>(gutils->inversionAllocs)).CreateUnreachable();
  DeleteDeadBlock(gutils->inversionAllocs);
  for (auto BBs : gutils->reverseBlocks) {
    if (pred_begin(BBs.second) == pred_end(BBs.second)) {
      (IRBuilder<>(BBs.second)).CreateUnreachable();
      DeleteDeadBlock(BBs.second);
    }
  }

  for (Argument &Arg : gutils->newFunc->args()) {
    if (Arg.hasAttribute(Attribute::Returned))
      Arg.removeAttr(Attribute::Returned);
    if (Arg.hasAttribute(Attribute::StructRet))
      Arg.removeAttr(Attribute::StructRet);
  }
  if (gutils->newFunc->hasFnAttribute(Attribute::OptimizeNone))
    gutils->newFunc->removeFnAttr(Attribute::OptimizeNone);

  if (auto bytes = gutils->newFunc->getDereferenceableBytes(
          llvm::AttributeList::ReturnIndex)) {
    AttrBuilder ab;
    ab.addDereferenceableAttr(bytes);
    gutils->newFunc->removeAttributes(llvm::AttributeList::ReturnIndex, ab);
  }

  if (gutils->newFunc->getAttributes().getRetAlignment()) {
    AttrBuilder ab;
    ab.addAlignmentAttr(gutils->newFunc->getAttributes().getRetAlignment());
    gutils->newFunc->removeAttributes(llvm::AttributeList::ReturnIndex, ab);
  }
  if (gutils->newFunc->hasAttribute(llvm::AttributeList::ReturnIndex,
                                    llvm::Attribute::NoAlias)) {
    gutils->newFunc->removeAttribute(llvm::AttributeList::ReturnIndex,
                                     llvm::Attribute::NoAlias);
  }
  if (gutils->newFunc->hasAttribute(llvm::AttributeList::ReturnIndex,
                                    llvm::Attribute::NonNull)) {
    gutils->newFunc->removeAttribute(llvm::AttributeList::ReturnIndex,
                                     llvm::Attribute::NonNull);
  }
  if (gutils->newFunc->hasAttribute(llvm::AttributeList::ReturnIndex,
                                    llvm::Attribute::ZExt)) {
    gutils->newFunc->removeAttribute(llvm::AttributeList::ReturnIndex,
                                     llvm::Attribute::ZExt);
  }

  if (llvm::verifyFunction(*gutils->newFunc, &llvm::errs())) {
    llvm::errs() << *gutils->oldFunc << "\n";
    llvm::errs() << *gutils->newFunc << "\n";
    report_fatal_error("function failed verification (4)");
  }

  optimizeIntermediate(gutils, topLevel, gutils->newFunc);

  auto nf = gutils->newFunc;
  delete gutils;

  return nf;
}
