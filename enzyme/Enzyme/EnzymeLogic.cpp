/*
 * EnzymeLogic.cpp
 *
 * Copyright (C) 2020 William S. Moses (enzyme@wsmoses.com) - All Rights Reserved
 *
 * For commercial use of this code please contact the author(s) above.
 */

#include "EnzymeLogic.h"

#include "SCEV/ScalarEvolutionExpander.h"

#include <deque>

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Verifier.h"

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueTracking.h"

#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"

#include "Utils.h"
#include "GradientUtils.h"
#include "FunctionUtils.h"
#include "LibraryFuncs.h"

using namespace llvm;

llvm::cl::opt<bool> enzyme_print("enzyme_print", cl::init(false), cl::Hidden,
                cl::desc("Print before and after fns for autodiff"));

cl::opt<bool> looseTypeAnalysis(
            "enzyme_loosetypes", cl::init(false), cl::Hidden,
            cl::desc("Allow looser use of types"));

cl::opt<bool> cache_reads_always(
            "enzyme_always_cache_reads", cl::init(false), cl::Hidden,
            cl::desc("Force always caching of all reads"));

cl::opt<bool> cache_reads_never(
            "enzyme_never_cache_reads", cl::init(false), cl::Hidden,
            cl::desc("Force never caching of all reads"));

cl::opt<bool> nonmarkedglobals_inactiveloads(
            "enzyme_nonmarkedglobals_inactiveloads", cl::init(true), cl::Hidden,
            cl::desc("Consider loads of nonmarked globals to be inactive"));

bool is_load_uncacheable(LoadInst& li, AAResults& AA, GradientUtils* gutils, TargetLibraryInfo& TLI, const SmallPtrSetImpl<const Instruction*> &unnecessaryInstructions, const std::map<Argument*, bool>& uncacheable_args);


bool is_value_mustcache_from_origin(Value* obj, AAResults& AA, GradientUtils* gutils, TargetLibraryInfo& TLI, const SmallPtrSetImpl<const Instruction*> &unnecessaryInstructions, const std::map<Argument*, bool>& uncacheable_args) {
  bool mustcache = false;

  // If the pointer operand is from an argument to the function, we need to check if the argument
  //   received from the caller is uncacheable.
  if (isa<UndefValue>(obj)) {
    //llvm::errs() << " + ocs undef (safe=" << init_safe << ") " << *callsite_op << " object: " << *obj << "\n";
  } else if (auto arg = dyn_cast<Argument>(obj)) {
    auto found = uncacheable_args.find(arg);
    if (found == uncacheable_args.end()) {
        llvm::errs() << "uncacheable_args:\n";
        for(auto& pair : uncacheable_args) {
            llvm::errs() << " + " << *pair.first << ": " << pair.second << " of func " << pair.first->getParent()->getName() << "\n";
        }
        llvm::errs() << "could not find " << *arg << " of func " << arg->getParent()->getName() << " in args_map\n";
    }
    assert(found != uncacheable_args.end());
    if (found->second) {
      //llvm::errs() << "OP is uncacheable arg: " << li << "\n";
      mustcache = true;
    }
    //llvm::errs() << " + argument (mustcache=" << mustcache << ") " << " object: " << *obj << " arg: " << *arg << "e\n";
  } else {

    // Pointer operands originating from call instructions that are not malloc/free are conservatively considered uncacheable.
    if (auto obj_op = dyn_cast<CallInst>(obj)) {
      Function* called = obj_op->getCalledFunction();
      if (auto castinst = dyn_cast<ConstantExpr>(obj_op->getCalledValue())) {
        if (castinst->isCast()) {
          if (auto fn = dyn_cast<Function>(castinst->getOperand(0))) {
            if (isAllocationFunction(*fn, TLI) || isDeallocationFunction(*fn, TLI)) {
              called = fn;
            }
          }
        }
      }
      if (called && isCertainMallocOrFree(called)) {
        //llvm::errs() << "OP is certain malloc or free: " << *op << "\n";
      } else {
        //llvm::errs() << "OP is a non malloc/free call so we need to cache " << *op << "\n";
        mustcache = true;
      }
    } else if (isa<AllocaInst>(obj)) {
      //No change to modref if alloca
    } else if (auto sli = dyn_cast<LoadInst>(obj)) {
      // If obj is from a load instruction conservatively consider it uncacheable if that load itself cannot be cached
      //llvm::errs() << "OP is from a load, needing to cache " << *op << "\n";
      mustcache = is_load_uncacheable(*sli, AA, gutils, TLI, unnecessaryInstructions, uncacheable_args);
    } else {
      // In absence of more information, assume that the underlying object for pointer operand is uncacheable in caller.
      //llvm::errs() << "OP is an unknown instruction, needing to cache obj:" << *obj << "\n";
      mustcache = true;
    }
  }
  return mustcache;
}

bool is_load_uncacheable(LoadInst& li, AAResults& AA, GradientUtils* gutils, TargetLibraryInfo& TLI, const SmallPtrSetImpl<const Instruction*> &unnecessaryInstructions, const std::map<Argument*, bool>& uncacheable_args) {
  assert(li.getParent()->getParent() == gutils->oldFunc);

  // Find the underlying object for the pointer operand of the load instruction.
  auto obj = GetUnderlyingObject(li.getPointerOperand(), gutils->oldFunc->getParent()->getDataLayout(), 100);


  bool can_modref = is_value_mustcache_from_origin(obj, AA, gutils, TLI, unnecessaryInstructions, uncacheable_args);

  //llvm::errs() << "underlying object for load " << li << " is " << *obj << " fromorigin: " << can_modref << "\n";

  if (!can_modref) {
    allFollowersOf(&li, [&](Instruction* inst2) {
      // Don't consider modref from malloc/free as a need to cache
      if (auto obj_op = dyn_cast<CallInst>(inst2)) {
        Function* called = obj_op->getCalledFunction();
        if (auto castinst = dyn_cast<ConstantExpr>(obj_op->getCalledValue())) {
          if (castinst->isCast()) {
            if (auto fn = dyn_cast<Function>(castinst->getOperand(0))) {
              if (isAllocationFunction(*fn, TLI) || isDeallocationFunction(*fn, TLI)) {
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

  //llvm::errs() << "F - " << li << " can_modref" << can_modref << "\n";
  return can_modref;
}

// Computes a map of LoadInst -> boolean for a function indicating whether that load is "uncacheable".
//   A load is considered "uncacheable" if the data at the loaded memory location can be modified after
//   the load instruction.
std::map<Instruction*, bool> compute_uncacheable_load_map(GradientUtils* gutils, AAResults& AA, TargetLibraryInfo& TLI, const SmallPtrSetImpl<const Instruction*> &unnecessaryInstructions,
    const std::map<Argument*, bool> uncacheable_args) {
  std::map<Instruction*, bool> can_modref_map;
  for (inst_iterator I = inst_begin(*gutils->oldFunc), E = inst_end(*gutils->oldFunc); I != E; ++I) {
    Instruction* inst = &*I;
      // For each load instruction, determine if it is uncacheable.
      if (auto op = dyn_cast<LoadInst>(inst)) {
        can_modref_map[inst] = is_load_uncacheable(*op, AA, gutils, TLI, unnecessaryInstructions, uncacheable_args);
      }
  }
  return can_modref_map;
}

std::map<Argument*, bool> compute_uncacheable_args_for_one_callsite(CallInst* callsite_op, DominatorTree &DT,
    TargetLibraryInfo &TLI, const SmallPtrSetImpl<const Instruction*> &unnecessaryInstructions, AAResults& AA, GradientUtils* gutils, const std::map<Argument*, bool> parent_uncacheable_args) {

  if (!callsite_op->getCalledFunction()) return {};

  std::vector<Value*> args;
  std::vector<bool> args_safe;

  //llvm::errs() << "CallInst: " << *callsite_op<< "CALL ARGUMENT INFO: \n";

  // First, we need to propagate the uncacheable status from the parent function to the callee.
  //   because memory location x modified after parent returns => x modified after callee returns.
  for (unsigned i = 0; i < callsite_op->getNumArgOperands(); i++) {
      args.push_back(callsite_op->getArgOperand(i));

      // If the UnderlyingObject is from one of this function's arguments, then we need to propagate the volatility.
      Value* obj = GetUnderlyingObject(callsite_op->getArgOperand(i),
                                       callsite_op->getParent()->getModule()->getDataLayout(),
                                       100);
      //llvm::errs() << "ocs underlying object for callsite " << *callsite_op << " idx: " << i << " is " << *obj << "\n";

      bool init_safe = !is_value_mustcache_from_origin(obj, AA, gutils, TLI, unnecessaryInstructions, parent_uncacheable_args);
      //llvm::errs() << " +++ safety " << init_safe << " of underlying object for callsite " << *callsite_op << " idx: " << i << " is " << *obj << "\n";
      args_safe.push_back(init_safe);
  }

  // Second, we check for memory modifications that can occur in the continuation of the
  //   callee inside the parent function.
  allFollowersOf(callsite_op, [&](Instruction* inst2) {
      // Don't consider modref from malloc/free as a need to cache
      if (auto obj_op = dyn_cast<CallInst>(inst2)) {
        Function* called = obj_op->getCalledFunction();
        if (auto castinst = dyn_cast<ConstantExpr>(obj_op->getCalledValue())) {
          if (castinst->isCast()) {
            if (auto fn = dyn_cast<Function>(castinst->getOperand(0))) {
              if (isAllocationFunction(*fn, TLI) || isDeallocationFunction(*fn, TLI)) {
                called = fn;
              }
            }
          }
        }
        if (called && isCertainMallocOrFree(called)) {
          return false;
        }
      }

      if (unnecessaryInstructions.count(inst2)) return false;

      for (unsigned i = 0; i < args.size(); i++) {
        if (llvm::isModSet(AA.getModRefInfo(inst2, MemoryLocation::getForArgument(callsite_op, i, TLI)))) {
          args_safe[i] = false;
          //llvm::errs() << "Instruction " << *inst2 << " is maybe ModRef with call argument " << *args[i] << "\n";
        }
      }
      return false;
  });

  std::map<Argument*, bool> uncacheable_args;

  auto arg = callsite_op->getCalledFunction()->arg_begin();
  for (unsigned i = 0; i < args.size(); i++) {
    uncacheable_args[arg] = !args_safe[i];
    //llvm::errs() << "callArg: " << *args[i] << " arg:" << *arg << " uncacheable: " << uncacheable_args[arg] << "\n";
    arg++;
    if (arg ==callsite_op->getCalledFunction()->arg_end()) {
      break;
    }
  }

  return uncacheable_args;
}

// Given a function and the arguments passed to it by its caller that are uncacheable (_uncacheable_args) compute
//   the set of uncacheable arguments for each callsite inside the function. A pointer argument is uncacheable at
//   a callsite if the memory pointed to might be modified after that callsite.
std::map<CallInst*, const std::map<Argument*, bool> > compute_uncacheable_args_for_callsites(
    Function* F, DominatorTree &DT, TargetLibraryInfo &TLI, const SmallPtrSetImpl<const Instruction*> &unnecessaryInstructions, AAResults& AA, GradientUtils* gutils,
    const std::map<Argument*, bool> uncacheable_args) {
  std::map<CallInst*, const std::map<Argument*, bool> > uncacheable_args_map;

  for (inst_iterator I = inst_begin(*gutils->oldFunc), E = inst_end(*gutils->oldFunc); I != E; ++I) {
    Instruction& inst = *I;
      if (auto op = dyn_cast<CallInst>(&inst)) {

        // We do not need uncacheable args for intrinsic functions. So skip such callsites.
        if(isa<IntrinsicInst>(&inst)) {
          continue;
        }

        /*
        // We do not need uncacheable args for memory allocation functions. So skip such callsites.
        Function* called = op->getCalledFunction();
        if (auto castinst = dyn_cast<ConstantExpr>(op->getCalledValue())) {
          if (castinst->isCast()) {
            if (auto fn = dyn_cast<Function>(castinst->getOperand(0))) {
              if (isAllocationFunction(*fn, TLI) || isDeallocationFunction(*fn, TLI)) {
                called = fn;
              }
            }
          }
        }
        if (isCertainMallocOrFree(called)) {
          continue;
        }*/

        // For all other calls, we compute the uncacheable args for this callsite.
        uncacheable_args_map.insert(std::pair<CallInst*, const std::map<Argument*, bool>>(op, compute_uncacheable_args_for_one_callsite(op,
            DT, TLI, unnecessaryInstructions, AA, gutils, uncacheable_args)));
      }
  }
  return uncacheable_args_map;
}

std::string to_string(const std::map<Argument*, bool>& us) {
    std::string s = "{";
    for(auto y : us) s += y.first->getName().str() + "@" + y.first->getParent()->getName().str() + ":" + std::to_string(y.second) + ",";
    return s + "}";
}

// Determine if a value is needed in the reverse pass. We only use this logic in the top level function right now.
bool is_value_needed_in_reverse(TypeResults &TR, const GradientUtils* gutils, const Value* inst, bool topLevel, std::map<std::pair<const Value*, bool>, bool> seen = {}) {
  auto idx = std::make_pair(inst, topLevel);
  if (seen.find(idx) != seen.end()) return seen[idx];
  if (auto ainst = dyn_cast<Instruction>(inst)) {
    assert(ainst->getParent()->getParent() == gutils->oldFunc);
  }

  //Inductively claim we aren't needed (and try to find contradiction)
  seen[idx] = false;

  //Consider all users of this value, do any of them need this in the reverse?
  for (auto use : inst->users()) {
    if (use == inst) continue;

    const Instruction* user = dyn_cast<Instruction>(use);

    // One may need to this value in the computation of loop bounds/comparisons/etc (which even though not active -- will be used for the reverse pass)
    //   We only need this if we're not doing the combined forward/reverse since otherwise it will use the local cache (rather than save for a separate backwards cache)
    if (!topLevel) {
        //Proving that none of the uses (or uses' uses) are used in control flow allows us to safely not do this load

        //TODO save loop bounds for dynamic loop

        //TODO make this more aggressive and dont need to save loop latch
        if (isa<BranchInst>(use) || isa<SwitchInst>(use)) {
            //llvm::errs() << " had to use in reverse since used in branch/switch " << *inst << " use: " << *use << "\n";
            return seen[idx] = true;
        }

        if (is_value_needed_in_reverse(TR, gutils, user, topLevel, seen)) {
            //llvm::errs() << " had to use in reverse since used in " << *inst << " use: " << *use << "\n";
            return seen[idx] = true;
        }
    }
    //llvm::errs() << " considering use : " << *user << " of " <<  *inst << "\n";

    //The following are types we know we don't need to compute adjoints

    // A pointer is only needed in the reverse pass if its non-store uses are needed in the reverse pass
    //   Moreover, we only need this pointer in the reverse pass if all of its non-store users are not already cached for the reverse pass
    if (!inst->getType()->isFPOrFPVectorTy() && TR.query(const_cast<Value*>(inst)).Data0()[{}].isPossiblePointer()) {
        //continue;
        bool unknown = true;
        for (auto zu : inst->users()) {
            // Stores to a pointer are not needed for the reverse pass
            if (auto si = dyn_cast<StoreInst>(zu)) {
                if (si->getPointerOperand() == inst) {
                    continue;
                }
            }

            if (isa<LoadInst>(zu) || isa<CastInst>(zu) || isa<PHINode>(zu)) {
                if (is_value_needed_in_reverse(TR, gutils, zu, topLevel, seen)) {
                    //llvm::errs() << " had to use in reverse since sub use " << *zu << " of " << *inst << "\n";
                    return seen[idx] = true;
                }
                continue;
            }

            if (auto II = dyn_cast<IntrinsicInst>(zu)) {
              if (II->getIntrinsicID() == Intrinsic::lifetime_start || II->getIntrinsicID() == Intrinsic::lifetime_end ||
                  II->getIntrinsicID() == Intrinsic::stacksave || II->getIntrinsicID() == Intrinsic::stackrestore) {
                continue;
              }
            }

            if (auto ci = dyn_cast<CallInst>(zu)) {
              // If this instruction isn't constant (and thus we need the argument to propagate to its adjoint)
              //   it may write memory and is topLevel (and thus we need to do the write in reverse)
              //   or we need this value for the reverse pass (we conservatively assume that if legal it is recomputed and not stored)
              if (!gutils->isConstantInstruction(ci) || (ci->mayWriteToMemory() && topLevel) || (gutils->legalRecompute(ci, ValueToValueMapTy()) && is_value_needed_in_reverse(TR, gutils, ci, topLevel, seen))) {
                  return seen[idx] = true;
              }
              continue;
            }


            /*
            if (auto gep = dyn_cast<GetElementPtrInst>(zu)) {
                for(auto &idx : gep->indices()) {
                    if (idx == inst) {
                        return seen[inst] = true;
                    }
                }
                if (gep->getPointerOperand() == inst && is_value_needed_in_reverse(gutils, gep, topLevel, seen)) {
                    //llvm::errs() << " had to use in reverse since sub gep use " << *zu << " of " << *inst << "\n";
                    return seen[inst] = true;
                }
                continue;
            }
            */

            //TODO add handling of call and allow interprocedural
            //llvm::errs() << " unknown pointer use " << *zu << " of " << *inst << "\n";
            unknown = true;
        }
        if (!unknown)
          continue;
          //return seen[inst] = false;
    }

    if (isa<LoadInst>(user) || isa<CastInst>(user) || isa<PHINode>(user)) {
        if (!is_value_needed_in_reverse(TR, gutils, user, topLevel, seen)) {
            continue;
        }
    }

    if (auto II = dyn_cast<IntrinsicInst>(user)) {
      if (II->getIntrinsicID() == Intrinsic::lifetime_start || II->getIntrinsicID() == Intrinsic::lifetime_end ||
          II->getIntrinsicID() == Intrinsic::stacksave || II->getIntrinsicID() == Intrinsic::stackrestore) {
        continue;
      }
    }

    if (auto op = dyn_cast<BinaryOperator>(user)) {
      if (op->getOpcode() == Instruction::FAdd || op->getOpcode() == Instruction::FSub) {
        continue;
      } else if (op->getOpcode() == Instruction::FMul) {
        bool needed = false;
        if (op->getOperand(0) == inst && !gutils->isConstantValue(op->getOperand(1))) needed = true;
        if (op->getOperand(1) == inst && !gutils->isConstantValue(op->getOperand(0))) needed = true;
        //llvm::errs() << "needed " << *inst << " in mul " << *op << " - needed:" << needed << "\n";
        if (!needed) continue;
      } else if (op->getOpcode() == Instruction::FDiv) {
        bool needed = false;
        if (op->getOperand(1) == inst && !gutils->isConstantValue(op->getOperand(1))) needed = true;
        if (op->getOperand(1) == inst && !gutils->isConstantValue(op->getOperand(0))) needed = true;
        if (op->getOperand(0) == inst && !gutils->isConstantValue(op->getOperand(1))) needed = true;
        //llvm::errs() << "needed " << *inst << " in div " << *op << " - needed:" << needed << "\n";
        if (!needed) continue;
      } else continue;
    }

    //We don't need only the indices of a GEP to compute the adjoint of a GEP
    if (auto gep = dyn_cast<GetElementPtrInst>(user)) {
        bool indexuse = false;
        for(auto &idx : gep->indices()) {
            if (idx == inst) {
                indexuse = true;
            }
        }
        if (!indexuse) continue;
    }

    if (auto si = dyn_cast<SelectInst>(use)) {
        // only need the condition if select is active
        if (gutils->isConstantValue(const_cast<SelectInst*>(si))) continue;
        //   none of the other operands are needed otherwise
        if (si->getCondition() != inst) {
            continue;
        }
    }

    //We don't need any of the input operands to compute the adjoint of a store instance
    if (isa<StoreInst>(use)) {
        continue;
    }

    if (isa<CmpInst>(use) || isa<BranchInst>(use) || isa<CastInst>(use) || isa<PHINode>(use) || isa<ReturnInst>(use) || isa<FPExtInst>(use) ||
        (isa<InsertElementInst>(use) && cast<InsertElementInst>(use)->getOperand(2) != inst) ||
        (isa<ExtractElementInst>(use) && cast<ExtractElementInst>(use)->getIndexOperand() != inst)
        //isa<LoadInst>(use) || (isa<SelectInst>(use) && cast<SelectInst>(use)->getCondition() != inst) //TODO remove load?
        //|| isa<SwitchInst>(use) || isa<ExtractElement>(use) || isa<InsertElementInst>(use) || isa<ShuffleVectorInst>(use) ||
        //isa<ExtractValueInst>(use) || isa<AllocaInst>(use)
        /*|| isa<StoreInst>(use)*/){
      continue;
    }

    //! Note it is important that return check comes before this as it may not have a new instruction

    if (auto ci = dyn_cast<CallInst>(use)) {
      // If this instruction isn't constant (and thus we need the argument to propagate to its adjoint)
      //   it may write memory and is topLevel (and thus we need to do the write in reverse)
      //   or we need this value for the reverse pass (we conservatively assume that if legal it is recomputed and not stored)
      if (!gutils->isConstantInstruction(ci) || (ci->mayWriteToMemory() && topLevel) || (gutils->legalRecompute(ci, ValueToValueMapTy()) && is_value_needed_in_reverse(TR, gutils, ci, topLevel, seen))) {
        return seen[idx] = true;
      }
      continue;
    }

    if (auto inst = dyn_cast<Instruction>(use))
        if (gutils->isConstantInstruction(const_cast<Instruction*>(inst))) continue;

    //llvm::errs() << " + must have in reverse from considering use : " << *user << " of " <<  *inst << "\n";
    return seen[idx] = true;
  }
  return false;
}

//! assuming not top level
std::pair<SmallVector<Type*,4>,SmallVector<Type*,4>> getDefaultFunctionTypeForAugmentation(FunctionType* called, bool returnUsed, DIFFE_TYPE retType) {
    SmallVector<Type*, 4> args;
    SmallVector<Type*, 4> outs;
    for(auto &argType : called->params()) {
        args.push_back(argType);

        if (!argType->isFPOrFPVectorTy()) {
            args.push_back(argType);
        }
    }

    auto ret = called->getReturnType();
    outs.push_back(Type::getInt8PtrTy(called->getContext()));
    if (!ret->isVoidTy() && !ret->isEmptyTy()) {
        if (returnUsed) {
            outs.push_back(ret);
        }
        if (retType == DIFFE_TYPE::DUP_ARG || retType == DIFFE_TYPE::DUP_NONEED) {
            outs.push_back(ret);
        }
    }

    return std::pair<SmallVector<Type*,4>,SmallVector<Type*,4>>(args, outs);
}

//! assuming not top level
std::pair<SmallVector<Type*,4>,SmallVector<Type*,4>> getDefaultFunctionTypeForGradient(FunctionType* called, DIFFE_TYPE retType) {
    SmallVector<Type*, 4> args;
    SmallVector<Type*, 4> outs;
    for(auto &argType : called->params()) {
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

    return std::pair<SmallVector<Type*,4>,SmallVector<Type*,4>>(args, outs);
}

static inline bool shouldAugmentCall(CallInst* op, const GradientUtils* gutils, TypeResults &TR) {
  assert(op->getParent()->getParent() == gutils->oldFunc);

  Function *called = op->getCalledFunction();

  bool modifyPrimal = !called || !called->hasFnAttribute(Attribute::ReadNone);

  if (modifyPrimal) {
     #ifdef PRINT_AUGCALL
     if (called)
       llvm::errs() << "primal modified " << called->getName() << " modified via reading from memory" << "\n";
     else
       llvm::errs() << "primal modified " << *op->getCalledValue() << " modified via reading from memory" << "\n";
     #endif
  }

  if ( !op->getType()->isFPOrFPVectorTy() && !gutils->isConstantValue(op) && TR.query(op).Data0()[{}].isPossiblePointer()) {
     modifyPrimal = true;

     #ifdef PRINT_AUGCALL
     if (called)
       llvm::errs() << "primal modified " << called->getName() << " modified via return" << "\n";
     else
       llvm::errs() << "primal modified " << *op->getCalledValue() << " modified via return" << "\n";
     #endif
  }

  if (!called || called->empty()) modifyPrimal = true;

  for(unsigned i=0;i<op->getNumArgOperands(); i++) {
    if (gutils->isConstantValue(op->getArgOperand(i)) && called && !called->empty()) {
        continue;
    }

    auto argType = op->getArgOperand(i)->getType();

    if (!argType->isFPOrFPVectorTy() && !gutils->isConstantValue(op->getArgOperand(i)) && TR.query(op->getArgOperand(i)).Data0()[{}].isPossiblePointer()) {
        if (called && ! ( called->hasParamAttribute(i, Attribute::ReadOnly) || called->hasParamAttribute(i, Attribute::ReadNone)) ) {
            modifyPrimal = true;
            #ifdef PRINT_AUGCALL
            if (called)
              llvm::errs() << "primal modified " << called->getName() << " modified via arg " << i << "\n";
            else
              llvm::errs() << "primal modified " << *op->getCalledValue() << " modified via arg " << i << "\n";
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
  llvm::errs() << "PM: " << *op << " modifyPrimal: " << modifyPrimal << " cv: " << gutils->isConstantValue(op) << "\n";
  #endif
  return modifyPrimal;
}


static inline llvm::raw_ostream& operator<<(llvm::raw_ostream& os, ModRefInfo mri) {
  if (mri == ModRefInfo::NoModRef) return os << "nomodref";
  else if (mri == ModRefInfo::ModRef) return os << "modref";
  else if (mri == ModRefInfo::Mod) return os << "mod";
  else if (mri == ModRefInfo::Ref) return os << "ref";
  else if (mri == ModRefInfo::MustModRef) return os << "mustmodref";
  else if (mri == ModRefInfo::MustMod) return os << "mustmod";
  else if (mri == ModRefInfo::MustRef) return os << "mustref";
  else llvm_unreachable("unknown modref");
  return os;
}

bool legalCombinedForwardReverse(CallInst *origop, const std::map<ReturnInst*,StoreInst*> &replacedReturns, std::vector<Instruction*> &postCreate, std::vector<Instruction*> &userReplace, GradientUtils* gutils, TypeResults &TR, const SmallPtrSetImpl<const Instruction*> &unnecessaryInstructions, const bool subretused) {
  Function* called =  origop->getCalledFunction();

  if (origop->getNumUses() != 0 && isa<PointerType>(origop->getType())) {
    if (called)
      llvm::errs() << " [not implemented] pointer return for combined forward/reverse " << called->getName() << "\n";
    else
      llvm::errs() << " [not implemented] pointer return for combined forward/reverse " << *origop->getCalledValue() << "\n";
    return false;
  }

  // Check any users of the returned value and determine all values that would be needed to be moved to reverse pass
  //  to ensure the forward pass would remain correct and everything computable
  SmallPtrSet<Instruction*,4> usetree;
  std::deque<Instruction*> todo { origop };


  bool legal = true;

  //Given a function I we know must be moved to the reverse for legality reasons
  auto propagate = [&](Instruction* I) {
    // if only used in unneeded return, don't need to move this to reverse (unless this is the original function)
    if (usetree.count(I)) return;
    //llvm::errs() << " propating: " << *I << "\n";
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
        llvm::errs() << " [bi] failed to replace function " << (called->getName()) << " due to " << *I << "\n";
      else
        llvm::errs() << " [bi] ailed to replace function " << (*origop->getCalledValue()) << " due to " << *I << "\n";
      return;
    }

    // Even though there is a dependency on this value here, we can ignore it if it isn't going to be used
    // Unless this is a call that could have a combined forward-reverse
    if (I != origop && unnecessaryInstructions.count(I) ) {
      if (gutils->isConstantInstruction(I) || !isa<CallInst>(I)) {
        userReplace.push_back(I);
        return;
      }
    }

    if (auto op = dyn_cast<CallInst>(I)) {
      Function *called = op->getCalledFunction();

      if (auto castinst = dyn_cast<ConstantExpr>(op->getCalledValue())) {
        if (castinst->isCast()) {
          if (auto fn = dyn_cast<Function>(castinst->getOperand(0))) {
            if (isAllocationFunction(*fn, gutils->TLI) || isDeallocationFunction(*fn, gutils->TLI)) {
              return;
            }
          }
        }
      }
      if (called && isDeallocationFunction(*called, gutils->TLI)) return;
    }


    if (isa<BranchInst>(I)) {
      legal = false;

      return;
    }
    if (isa<PHINode>(I)) {
      legal = false;
      if (called)
        llvm::errs() << " [phi] failed to replace function " << (called->getName()) << " due to " << *I << "\n";
      else
        llvm::errs() << " [phi] ailed to replace function " << (*origop->getCalledValue()) << " due to " << *I << "\n";
      return;
    }
    if (is_value_needed_in_reverse(TR, gutils, I, /*topLevel*/true)) {
      legal = false;
      if (called)
        llvm::errs() << " [nv] failed to replace function " << (called->getName()) << " due to " << *I << "\n";
      else
        llvm::errs() << " [nv] ailed to replace function " << (*origop->getCalledValue()) << " due to " << *I << "\n";
      return;
    }
    if (I != origop && !isa<IntrinsicInst>(I) && isa<CallInst>(I)) {
      legal = false;
      if (called)
        llvm::errs() << " [ci] failed to replace function " << (called->getName()) << " due to " << *I << "\n";
      else
        llvm::errs() << " [ci] ailed to replace function " << (*origop->getCalledValue()) << " due to " << *I << "\n";
      return;
    }
    // Do not try moving an instruction that modifies memory, if we already moved it
    if (!isa<StoreInst>(I) || unnecessaryInstructions.count(I) == 0)
    if (I->mayReadOrWriteMemory() && gutils->getNewFromOriginal(I)->getParent() != gutils->getNewFromOriginal(I->getParent())) {
      legal = false;
      if (called)
        llvm::errs() << " [am] failed to replace function " << (called->getName()) << " due to " << *I << "\n";
      else
        llvm::errs() << " [am] ailed to replace function " << (*origop->getCalledValue()) << " due to " << *I << "\n";
      return;
    }

    //llvm::errs() << " inserting: " << *I << "\n";
    usetree.insert(I);
    for(auto use : I->users()) {
      //llvm::errs() << "I: " << *I << " use: " << *use << "\n";
      todo.push_back(cast<Instruction>(use));
    }
  };

  while (!todo.empty()) {
    auto inst = todo.front();
    todo.pop_front();

    if (inst->mayWriteToMemory()) {
      auto consider = [&](Instruction* user) {
        if (!user->mayReadFromMemory()) return false;
        if (writesToMemoryReadBy(gutils->AA, /*maybeReader*/user, /*maybeWriter*/inst)) {
          //llvm::errs() << " memory deduced need follower of " << *inst << " - " << *user << "\n";
          propagate(user);
          // Fast return if not legal
          if (!legal) return true;
        }
        return false;
      };
      allFollowersOf(inst, consider);
      if (!legal) return false;
      }

    propagate(inst);
    if (!legal) return false;
  }

  //llvm::errs() << " found usetree for: " << ci << "\n";
  //for(auto u : usetree)
  //  llvm::errs() << " + " << *u << "\n";

  // Check if any of the unmoved operations will make it illegal to move the instruction

  for (auto inst : usetree) {
    if (!inst->mayReadFromMemory()) continue;
    allFollowersOf(inst, [&](Instruction* post) {
      if (unnecessaryInstructions.count(post)) return false;
      if (!post->mayWriteToMemory()) return false;
      //llvm::errs() << " checking if illegal move of " << *inst << " due to " << *post << "\n";
      if (writesToMemoryReadBy(gutils->AA, /*maybeReader*/inst, /*maybeWriter*/post)) {
        if (called)
          llvm::errs() << " failed to replace function " << (called->getName()) << " due to " << *post << " usetree: " << *inst << "\n";
        else
          llvm::errs() << " failed to replace function " << (*origop->getCalledValue()) << " due to " << *post << " usetree: " << *inst << "\n";
        legal = false;
        return true;
      }
      return false;
    });
    if (!legal) break;
  }

  if (!legal) return false;


  allFollowersOf(origop, [&](Instruction* inst) {
    if (auto ri = dyn_cast<ReturnInst>(inst)) {
      auto find = replacedReturns.find(ri);
      if (find != replacedReturns.end()) {
        postCreate.push_back(find->second);
        return false;
      }
    }

    if (usetree.count(inst) == 0) return false;
    if (inst->getParent() != origop->getParent()) {
      // Don't move a writing instruction (may change speculatable/etc things)
      if (inst->mayWriteToMemory()) {
        if (called)
          llvm::errs() << " [nonspec] failed to replace function " << (called->getName()) << " due to " << *inst << "\n";
        else
          llvm::errs() << " [nonspec] ailed to replace function " << (*origop->getCalledValue()) << " due to " << *inst << "\n";
        legal = false;
        // Early exit
        return true;
      }
    }
    if (isa<CallInst>(inst) && gutils->originalToNewFn.find(inst) == gutils->originalToNewFn.end()) {
      legal = false;
      if (called)
        llvm::errs() << " [premove] failed to replace function " << (called->getName()) << " due to " << *inst << "\n";
      else
        llvm::errs() << " [premove] ailed to replace function " << (*origop->getCalledValue()) << " due to " << *inst << "\n";
      // Early exit
      return true;
    }
    postCreate.push_back(gutils->getNewFromOriginal(inst));
    return false;
  });

  if (!legal) return false;

  if (called)
    llvm::errs() << " choosing to replace function " << (called->getName()) << " and do both forward/reverse\n";
  else
    llvm::errs() << " choosing to replace function " << (*origop->getCalledValue()) << " and do both forward/reverse\n";

  return true;
}

enum class DerivativeMode {
    Forward,
    Reverse,
    Both
};

std::string to_string(DerivativeMode mode) {
  switch(mode) {
    case DerivativeMode::Forward: return "Forward";
    case DerivativeMode::Reverse: return "Reverse";
    case DerivativeMode::Both: return "Both";
  }
  llvm_unreachable("illegal derivative mode");
}

template<class AugmentedReturnType = AugmentedReturn*>
class DerivativeMaker : public llvm::InstVisitor<DerivativeMaker<AugmentedReturnType>> {
public:
  DerivativeMode mode;
  GradientUtils *gutils;
  const std::vector<DIFFE_TYPE>& constant_args;
  TypeResults &TR;
  std::function<unsigned(Instruction*, CacheType)> getIndex;
  const std::map<CallInst*, const std::map<Argument*, bool> > uncacheable_args_map;
  const SmallPtrSetImpl<Instruction*> *returnuses;
  AugmentedReturnType augmentedReturn;
  std::vector<Instruction*> *fakeTBAA;
  const std::map<ReturnInst*,StoreInst*> *replacedReturns;

  const SmallPtrSetImpl<const Value*> &unnecessaryValues;
  const SmallPtrSetImpl<const Instruction*> &unnecessaryInstructions;
  const SmallPtrSetImpl<const Instruction*> &unnecessaryStores;

  AllocaInst* dretAlloca;
  DerivativeMaker(DerivativeMode mode, GradientUtils *gutils, const std::vector<DIFFE_TYPE> &constant_args, TypeResults &TR, std::function<unsigned(Instruction*, CacheType)> getIndex,
    const std::map<CallInst*, const std::map<Argument*, bool> > uncacheable_args_map, const SmallPtrSetImpl<Instruction*> *returnuses, AugmentedReturnType augmentedReturn,
    std::vector<Instruction*>* fakeTBAA, const std::map<ReturnInst*,StoreInst*>* replacedReturns,
    const SmallPtrSetImpl<const Value*> &unnecessaryValues,
    const SmallPtrSetImpl<const Instruction*> &unnecessaryInstructions,
    const SmallPtrSetImpl<const Instruction*> &unnecessaryStores,
    AllocaInst* dretAlloca
    ) : mode(mode), gutils(gutils), constant_args(constant_args), TR(TR),
        getIndex(getIndex), uncacheable_args_map(uncacheable_args_map),
        returnuses(returnuses), augmentedReturn(augmentedReturn), fakeTBAA(fakeTBAA), replacedReturns(replacedReturns),
        unnecessaryValues(unnecessaryValues), unnecessaryInstructions(unnecessaryInstructions), unnecessaryStores(unnecessaryStores), dretAlloca(dretAlloca) {

    assert(TR.info.function == gutils->oldFunc);
    for(auto &pair : TR.analysis.analyzedFunctions.find(TR.info)->second.analysis) {
        if (auto in = dyn_cast<Instruction>(pair.first)) {
          if (in->getParent()->getParent() != gutils->oldFunc) {
            llvm::errs() << "inf: " << *in->getParent()->getParent() << "\n";
            llvm::errs() << "gutils->oldFunc: " << *gutils->oldFunc << "\n";
            llvm::errs() << "in: " << *in << "\n";
          }
          assert(in->getParent()->getParent() == gutils->oldFunc);
        }
    }


  }

  SmallPtrSet<Instruction*, 4> erased;

  void eraseIfUnused(llvm::Instruction &I, bool erase=true, bool check=true) {
    bool used = unnecessaryInstructions.find(&I) == unnecessaryInstructions.end();

    auto iload = gutils->getNewFromOriginal(&I);

    // We still need this value if it is the increment/induction variable for a loop
    for(auto& context : gutils->loopContexts) {
      if (context.second.var == iload || context.second.incvar == iload) {
        used = true;
        break;
      }
    }

    //llvm::errs() << " eraseIfUnused:" << I << " used: " << used << " erase:" << erase << " check:" << check << "\n";

    if (used && check) return;

    PHINode* pn = nullptr;
    if (!I.getType()->isVoidTy()) {
      IRBuilder<> BuilderZ(iload);
      pn = BuilderZ.CreatePHI(I.getType(), 1, (I.getName()+"_replacementA").str());
      gutils->fictiousPHIs.push_back(pn);


      for(auto inst_orig : unnecessaryInstructions) {
        if (isa<ReturnInst>(inst_orig)) continue;
        if (erased.count(inst_orig)) continue;
        auto inst = gutils->getNewFromOriginal(inst_orig);
        for(unsigned i=0; i<inst->getNumOperands(); i++) {
          if (inst->getOperand(i) == iload) {
            inst->setOperand(i, pn);
            //inst->setOperand(i, UndefValue::get(iload->getType()));
          }
        }
      }


    }

    erased.insert(&I);
    if (erase) {
      if (pn) gutils->replaceAWithB(iload, pn);
      gutils->erase(iload);
    }
  }

  void visitInstruction(llvm::Instruction& inst) {
    //TODO explicitly handle all instructions rather than using the catch all below
    if (mode == DerivativeMode::Forward) return;

    llvm::errs() << *gutils->oldFunc << "\n";
    llvm::errs() << *gutils->newFunc << "\n";
    llvm::errs() << "in mode: " << to_string(mode) << "\n";
    llvm::errs() << "cannot handle unknown instruction\n" << inst;
    report_fatal_error("unknown value");
  }

  void visitAllocaInst(llvm::AllocaInst &I) {
    eraseIfUnused(I);
  }
  void visitICmpInst(llvm::ICmpInst &I) {
    eraseIfUnused(I);
  }

  void visitFCmpInst(llvm::FCmpInst &I) {
    eraseIfUnused(I);
  }

  void visitLoadInst(llvm::LoadInst &LI) {
    bool constantval = gutils->isConstantValue(&LI);
    auto alignment = LI.getAlignment();
    BasicBlock* parent = LI.getParent();
    Type*  type = LI.getType();

    LoadInst* newi = dyn_cast<LoadInst>(gutils->getNewFromOriginal(&LI));

    //! Store inverted pointer loads that need to be cached for use in reverse pass
    if (!type->isEmptyTy() && !type->isFPOrFPVectorTy() && TR.query(&LI).Data0()[{}].isPossiblePointer()) {
      PHINode* placeholder = cast<PHINode>(gutils->invertedPointers[&LI]);
      assert(placeholder->getType() == type);
      gutils->invertedPointers.erase(&LI);

      //TODO consider optimizing when you know it isnt a pointer and thus don't need to store
      if (!constantval) {
        IRBuilder<> BuilderZ(placeholder);
        Value* newip = nullptr;

        switch(mode) {

          case DerivativeMode::Forward:
          case DerivativeMode::Both:{
            newip = gutils->invertPointerM(&LI, BuilderZ);
            assert(newip->getType() == type);

            if (mode == DerivativeMode::Forward && gutils->can_modref_map->find(&LI)->second) {
              gutils->addMalloc(BuilderZ, newip, getIndex(&LI, CacheType::Shadow));
            }
            placeholder->replaceAllUsesWith(newip);
            gutils->erase(placeholder);
            gutils->invertedPointers[&LI] = newip;
            break;
          }

          case DerivativeMode::Reverse:{
            //only make shadow where caching needed
            if (gutils->can_modref_map->find(&LI)->second) {
              newip = gutils->addMalloc(BuilderZ, placeholder, getIndex(&LI, CacheType::Shadow));
              assert(newip->getType() == type);
              gutils->invertedPointers[&LI] = newip;
            } else {
              newip = gutils->invertPointerM(&LI, BuilderZ);
              assert(newip->getType() == type);
              placeholder->replaceAllUsesWith(newip);
              gutils->erase(placeholder);
              gutils->invertedPointers[&LI] = newip;
            }
            break;
          }
        }

      } else {
        gutils->erase(placeholder);
      }
    }

    eraseIfUnused(LI);

    // Allow forcing cache reads to be on or off using flags.
    assert(!(cache_reads_always && cache_reads_never) && "Both cache_reads_always and cache_reads_never are true. This doesn't make sense.");

    Value* inst = newi;

    //! Store loads that need to be cached for use in reverse pass
    if (cache_reads_always || (!cache_reads_never && gutils->can_modref_map->find(&LI)->second && is_value_needed_in_reverse(TR, gutils, &LI, /*toplevel*/mode == DerivativeMode::Both))) {
      IRBuilder<> BuilderZ(gutils->getNewFromOriginal(&LI)->getNextNode());
      //auto tbaa = inst->getMetadata(LLVMContext::MD_tbaa);

      inst = gutils->addMalloc(BuilderZ, newi, getIndex(&LI, CacheType::Self));
      assert(inst->getType() == type);

      if (mode == DerivativeMode::Reverse) {
        assert(inst != newi);
        //if (tbaa) {
          //inst->setMetadata(LLVMContext::MD_tbaa, tbaa);
        //  assert(fakeTBAA);
        //  fakeTBAA->push_back(inst);
        //}
      } else {
        assert(inst == newi);
      }
    }

    if (mode == DerivativeMode::Forward) return;

    if (constantval) return;

    if (nonmarkedglobals_inactiveloads) {
      //Assume that non enzyme_shadow globals are inactive
      //  If we ever store to a global variable, we will error if it doesn't have a shadow
      //  This allows functions who only read global memory to have their derivative computed
      //  Note that this is too aggressive for general programs as if the global aliases with an argument something that is written to, then we will have a logical error
      if (auto arg = dyn_cast<GlobalVariable>(LI.getPointerOperand())) {
        if (!hasMetadata(arg, "enzyme_shadow")) {
          return;
        }
      }
    }

    bool isfloat = type->isFPOrFPVectorTy();
    if (!isfloat && type->isIntOrIntVectorTy()) {
      auto storeSize = gutils->newFunc->getParent()->getDataLayout().getTypeSizeInBits(type) / 8;
      auto vd = TR.firstPointer(storeSize, LI.getPointerOperand(), /*errifnotfound*/false, /*pointerIntSame*/true);
      if (vd.isKnown()) isfloat = vd.isFloat();
      else isfloat = TR.intType(&LI, /*errIfNotFound*/!looseTypeAnalysis).isFloat();
    }

    if (isfloat) {
      IRBuilder<> Builder2 = getReverseBuilder(parent);
      auto prediff = diffe(&LI, Builder2);
      setDiffe(&LI, Constant::getNullValue(type), Builder2);
      //llvm::errs() << "  + doing load propagation: orig:" << *oorig << " inst:" << *inst << " prediff: " << *prediff << " inverted_operand: " << *inverted_operand << "\n";

      if (!gutils->isConstantValue(LI.getPointerOperand())) {
        Value* inverted_operand = gutils->invertPointerM(LI.getPointerOperand(), Builder2);
        assert(inverted_operand);
        ((DiffeGradientUtils*)gutils)->addToInvertedPtrDiffe(inverted_operand, prediff, Builder2, alignment);
      }
    }
  }

  void visitStoreInst(llvm::StoreInst &SI) {
    Value* orig_ptr = SI.getPointerOperand();
    Value* orig_val = SI.getValueOperand();
    Value* val  = gutils->getNewFromOriginal(orig_val);
    Type* valType = orig_val->getType();

    if (unnecessaryStores.count(&SI)) {
      eraseIfUnused(SI);
      return;
    }


    if (gutils->isConstantValue(orig_ptr)) {
      eraseIfUnused(SI);
      return;
    }

    //TODO allow recognition of other types that could contain pointers [e.g. {void*, void*} or <2 x i64> ]
    StoreInst* ts = nullptr;

    auto storeSize = gutils->newFunc->getParent()->getDataLayout().getTypeSizeInBits(valType) / 8;

    //! Storing a floating point value
    Type* FT = nullptr;
    if (valType->isFPOrFPVectorTy()) {
      FT = valType->getScalarType();
    } else if (!valType->isPointerTy()) {
      if (looseTypeAnalysis) {
        auto fp = TR.firstPointer(storeSize, orig_ptr, /*errifnotfound*/false, /*pointerIntSame*/true);
        if (fp.isKnown()) {
            FT = fp.isFloat();
        } else if (isa<ConstantInt>(orig_val) || valType->isIntOrIntVectorTy()) {
            llvm::errs() << "assuming type as integral for store: " << SI << "\n";
            FT = nullptr;
        } else {
            TR.firstPointer(storeSize, orig_ptr, /*errifnotfound*/true, /*pointerIntSame*/true);
            llvm::errs() << "cannot deduce type of store " << SI << "\n";
            assert(0 && "cannot deduce");
        }
      } else
      FT = TR.firstPointer(storeSize, orig_ptr, /*errifnotfound*/true, /*pointerIntSame*/true).isFloat();
    }

    if (FT) {
      //! Only need to update the reverse function
      if (mode == DerivativeMode::Reverse || mode == DerivativeMode::Both) {
        IRBuilder<> Builder2 = getReverseBuilder(SI.getParent());

        if (gutils->isConstantValue(orig_val)) {
          ts = setPtrDiffe(orig_ptr, Constant::getNullValue(valType), Builder2);
        } else {
          auto dif1 = Builder2.CreateLoad(gutils->invertPointerM(orig_ptr, Builder2));
          dif1->setAlignment(SI.getAlignment());
          ts = setPtrDiffe(orig_ptr, Constant::getNullValue(valType), Builder2);
          addToDiffe(orig_val, dif1, Builder2, FT);
        }
      }

    //! Storing an integer or pointer
    } else {
      //! Only need to update the forward function
      if (mode == DerivativeMode::Forward || mode == DerivativeMode::Both) {
        IRBuilder <> storeBuilder(gutils->getNewFromOriginal(&SI));

        Value* valueop = nullptr;

        //Fallback mechanism, TODO check
        if (gutils->isConstantValue(orig_val)) {
            valueop = val; //Constant::getNullValue(op->getValueOperand()->getType());
        } else {
            valueop = gutils->invertPointerM(orig_val, storeBuilder);
        }
        ts = setPtrDiffe(orig_ptr, valueop, storeBuilder);
      }

    }

    if (ts) {
      ts->setAlignment(SI.getAlignment());
      ts->setVolatile(SI.isVolatile());
      ts->setOrdering(SI.getOrdering());
      ts->setSyncScopeID(SI.getSyncScopeID());
    }
    eraseIfUnused(SI);
  }

  void visitGetElementPtrInst(llvm::GetElementPtrInst &gep) {
    eraseIfUnused(gep);
  }

  void visitPHINode(llvm::PHINode& phi) {
    eraseIfUnused(phi);
  }

  void visitCastInst(llvm::CastInst &I) {
    eraseIfUnused(I);
    if (gutils->isConstantValue(&I)) return;
    if (I.getType()->isPointerTy() || I.getOpcode() == CastInst::CastOps::PtrToInt) return;

    if (mode == DerivativeMode::Forward) return;

    Value* orig_op0 = I.getOperand(0);
    Value* op0 = gutils->getNewFromOriginal(orig_op0);

    IRBuilder<> Builder2 = getReverseBuilder(I.getParent());

    if (!gutils->isConstantValue(orig_op0)) {
      Value* dif = diffe(&I, Builder2);
      if (I.getOpcode()==CastInst::CastOps::FPTrunc || I.getOpcode()==CastInst::CastOps::FPExt) {
        addToDiffe(orig_op0, Builder2.CreateFPCast (dif, op0->getType()), Builder2, TR.intType(orig_op0, false).isFloat());
      } else if (I.getOpcode()==CastInst::CastOps::BitCast) {
        addToDiffe(orig_op0, Builder2.CreateBitCast(dif, op0->getType()), Builder2, TR.intType(orig_op0, false).isFloat());
      } else if (I.getOpcode()==CastInst::CastOps::Trunc) {
        //TODO CHECK THIS
        auto trunced = Builder2.CreateZExt(dif, op0->getType());
        addToDiffe(orig_op0, trunced, Builder2, TR.intType(orig_op0, false).isFloat());
      } else {
        llvm::errs() << *I.getParent()->getParent() << "\n" << *I.getParent() << "\n";
        llvm::errs() << "cannot handle above cast " << I << "\n";
        report_fatal_error("unknown instruction");
      }
    }
    setDiffe(&I, Constant::getNullValue(I.getType()), Builder2);
  }

  void visitSelectInst(llvm::SelectInst &SI) {
    eraseIfUnused(SI);
    if (gutils->isConstantValue(&SI)) return;
    if (SI.getType()->isPointerTy()) return;

    if (mode == DerivativeMode::Forward) return;

    Value* op0 = gutils->getNewFromOriginal(SI.getOperand(0));
    Value* orig_op1 = SI.getOperand(1);
    Value* op1 = gutils->getNewFromOriginal(orig_op1);
    Value* orig_op2 = SI.getOperand(2);
    Value* op2 = gutils->getNewFromOriginal(orig_op2);

    //TODO fix all the reverse builders
    IRBuilder<> Builder2 = getReverseBuilder(SI.getParent());

    Value* dif1 = nullptr;
    Value* dif2 = nullptr;

    if (!gutils->isConstantValue(orig_op1))
      dif1 = Builder2.CreateSelect(lookup(op0, Builder2), diffe(&SI, Builder2), Constant::getNullValue(op1->getType()), "diffe"+op1->getName());
    if (!gutils->isConstantValue(orig_op2))
      dif2 = Builder2.CreateSelect(lookup(op0, Builder2), Constant::getNullValue(op2->getType()), diffe(&SI, Builder2), "diffe"+op2->getName());

    setDiffe(&SI, Constant::getNullValue(SI.getType()), Builder2);
    if (dif1) addToDiffe(orig_op1, dif1, Builder2, TR.intType(orig_op1, false).isFloat());
    if (dif2) addToDiffe(orig_op2, dif2, Builder2, TR.intType(orig_op2, false).isFloat());
  }

  void visitExtractElementInst(llvm::ExtractElementInst &EEI) {
    eraseIfUnused(EEI);
    if (gutils->isConstantValue(&EEI)) return;
    if (mode == DerivativeMode::Forward) return;

    IRBuilder<> Builder2 = getReverseBuilder(EEI.getParent());

    Value* orig_vec = EEI.getVectorOperand();

    if (!gutils->isConstantValue(orig_vec)) {
      SmallVector<Value*,4> sv;
      sv.push_back(gutils->getNewFromOriginal(EEI.getIndexOperand()));
      ((DiffeGradientUtils*)gutils)->addToDiffeIndexed(orig_vec, diffe(&EEI, Builder2), sv, Builder2);
    }
    setDiffe(&EEI, Constant::getNullValue(EEI.getType()), Builder2);
  }

  void visitInsertElementInst(llvm::InsertElementInst &IEI) {
    eraseIfUnused(IEI);
    if (gutils->isConstantValue(&IEI)) return;
    if (mode == DerivativeMode::Forward) return;

    IRBuilder<> Builder2 = getReverseBuilder(IEI.getParent());

    Value* dif1 = diffe(&IEI, Builder2);


    Value* orig_op0 = IEI.getOperand(0);
    Value* orig_op1 = IEI.getOperand(1);
    Value* op1 = gutils->getNewFromOriginal(orig_op1);
    Value* op2 = gutils->getNewFromOriginal(IEI.getOperand(2));

    if (!gutils->isConstantValue(orig_op0))
      addToDiffe(orig_op0, Builder2.CreateInsertElement(dif1, Constant::getNullValue(op1->getType()), lookup(op2, Builder2)), Builder2, TR.intType(orig_op0, false).isFloat());

    if (!gutils->isConstantValue(orig_op1))
      addToDiffe(orig_op1, Builder2.CreateExtractElement(dif1, lookup(op2, Builder2)), Builder2, TR.intType(orig_op1, false).isFloat());

    setDiffe(&IEI, Constant::getNullValue(IEI.getType()), Builder2);
  }

  void visitShuffleVectorInst(llvm::ShuffleVectorInst &SVI) {
    eraseIfUnused(SVI);
    if (gutils->isConstantValue(&SVI)) return;
    if (mode == DerivativeMode::Forward) return;


    IRBuilder<> Builder2 = getReverseBuilder(SVI.getParent());

    auto loaded = diffe(&SVI, Builder2);
    size_t l1 = cast<VectorType>(SVI.getOperand(0)->getType())->getNumElements();
    uint64_t instidx = 0;

    for( size_t idx : SVI.getShuffleMask()) {
      auto opnum = (idx < l1) ? 0 : 1;
      auto opidx = (idx < l1) ? idx : (idx-l1);
      SmallVector<Value*,4> sv;
      sv.push_back(ConstantInt::get(Type::getInt32Ty(SVI.getContext()), opidx));
      if (!gutils->isConstantValue(SVI.getOperand(opnum)))
        ((DiffeGradientUtils*)gutils)->addToDiffeIndexed(SVI.getOperand(opnum), Builder2.CreateExtractElement(loaded, instidx), sv, Builder2);
      instidx++;
    }
    setDiffe(&SVI, Constant::getNullValue(SVI.getType()), Builder2);
  }

  void visitExtractValueInst(llvm::ExtractValueInst &EVI) {
    eraseIfUnused(EVI);
    if (gutils->isConstantValue(&EVI)) return;
    if (EVI.getType()->isPointerTy()) return;

    if (mode == DerivativeMode::Forward) return;


    Value* orig_op0 = EVI.getOperand(0);

    IRBuilder<> Builder2 = getReverseBuilder(EVI.getParent());

    auto prediff = diffe(&EVI, Builder2);

    //todo const
    if (!gutils->isConstantValue(orig_op0)) {
      SmallVector<Value*,4> sv;
      for(auto i : EVI.getIndices())
        sv.push_back(ConstantInt::get(Type::getInt32Ty(EVI.getContext()), i));
      ((DiffeGradientUtils*)gutils)->addToDiffeIndexed(orig_op0, prediff, sv, Builder2);
    }

    setDiffe(&EVI, Constant::getNullValue(EVI.getType()), Builder2);
  }

  void visitInsertValueInst(llvm::InsertValueInst &IVI) {
    eraseIfUnused(IVI);
    if (gutils->isConstantValue(&IVI)) return;

    if (mode == DerivativeMode::Forward) return;

    auto st = cast<StructType>(IVI.getType());
    bool hasNonPointer = false;
    for(unsigned i=0; i<st->getNumElements(); i++) {
      if (!st->getElementType(i)->isPointerTy()) {
         hasNonPointer = true;
      }
    }
    if (!hasNonPointer) return;

    bool floatingInsertion = false;
    for(InsertValueInst* iv = &IVI; ;) {
      auto it = TR.intType(iv->getInsertedValueOperand(), false);
      if (it.isFloat() || !it.isKnown()) {
        floatingInsertion = true;
        break;
      }
      Value* val = iv->getAggregateOperand();
      if (gutils->isConstantValue(val)) break;
      if (auto dc = dyn_cast<InsertValueInst>(val)) {
        iv = dc;
      } else {
        // unsure where this came from, conservatively assume contains float
        floatingInsertion = true;
        break;
      }
    }

    if (!floatingInsertion) return;

    // TODO handle pointers
    // TODO type analysis handle structs

    IRBuilder<> Builder2 = getReverseBuilder(IVI.getParent());

    Value* orig_inserted = IVI.getInsertedValueOperand();
    Value* orig_agg = IVI.getAggregateOperand();

    Type* flt = nullptr;
    if (!gutils->isConstantValue(orig_inserted) && (flt = TR.intType(orig_inserted).isFloat())) {
      auto prediff = diffe(&IVI, Builder2);
      auto dindex = Builder2.CreateExtractValue(prediff, IVI.getIndices());
      addToDiffe(orig_inserted, dindex, Builder2, flt);
    }

    if (!gutils->isConstantValue(orig_agg)) {
      auto prediff = diffe(&IVI, Builder2);
      auto dindex = Builder2.CreateInsertValue(prediff, Constant::getNullValue(orig_inserted->getType()), IVI.getIndices());
      llvm::errs() << "orig:" << IVI << " query(orig_agg):" << TR.query(orig_agg).str() << "\n";
      addToDiffe(orig_agg, dindex, Builder2, TR.intType(orig_agg, false).isFloat());
    }

    setDiffe(&IVI, Constant::getNullValue(IVI.getType()), Builder2);
  }

  inline IRBuilder<> getReverseBuilder(BasicBlock* oBB) {
    BasicBlock* BB = cast<BasicBlock>(gutils->getNewFromOriginal(oBB));
    BasicBlock* BB2 = gutils->reverseBlocks[BB];
    if (!BB2) {
      llvm::errs() << "oldFunc: " << *gutils->oldFunc << "\n";
      llvm::errs() << "newFunc: " << *gutils->newFunc << "\n";
      llvm::errs() << "could not invert " << *BB;
    }
    assert(BB2);

    IRBuilder<> Builder2(BB2);
    //if (BB2->size() > 0) {
    //    Builder2.SetInsertPoint(BB2->getFirstNonPHI());
    //}
    Builder2.setFastMathFlags(getFast());
    return Builder2;
  }

  Value* diffe(Value* val, IRBuilder<> &Builder) {
    assert(mode == DerivativeMode::Reverse || mode == DerivativeMode::Both);
    return ((DiffeGradientUtils*)gutils)->diffe(val, Builder);
  }

  void setDiffe(Value* val, Value* dif, IRBuilder<> &Builder) {
    assert(mode == DerivativeMode::Reverse || mode == DerivativeMode::Both);
    ((DiffeGradientUtils*)gutils)->setDiffe(val, dif, Builder);
  }

  StoreInst* setPtrDiffe(Value* val, Value* dif, IRBuilder<> &Builder) {
    return gutils->setPtrDiffe(val, dif, Builder);
  }

  std::vector<SelectInst*> addToDiffe(Value* val, Value* dif, IRBuilder<> &Builder, Type* T) {
    assert(mode == DerivativeMode::Reverse || mode == DerivativeMode::Both);
    return ((DiffeGradientUtils*)gutils)->addToDiffe(val, dif, Builder, T);
  }

  Value* lookup(Value* val, IRBuilder<> &Builder) {
    return gutils->lookupM(val, Builder);
  }

  void visitBinaryOperator(llvm::BinaryOperator &BO) {
    eraseIfUnused(BO);
    if (gutils->isConstantValue(&BO)) return;
    if (mode != DerivativeMode::Reverse && mode != DerivativeMode::Both) return;

    Value* orig_op0 = BO.getOperand(0);
    Value* orig_op1 = BO.getOperand(1);
    bool constantval0 = gutils->isConstantValue(orig_op0);
    bool constantval1 = gutils->isConstantValue(orig_op1);


    if (BO.getType()->isIntOrIntVectorTy() && TR.intType(&BO, /*errifnotfound*/false) == IntType::Pointer) {
      return;
    }

    IRBuilder<> Builder2 = getReverseBuilder(BO.getParent());

    Value* dif0 = nullptr;
    Value* dif1 = nullptr;
    Value* idiff = diffe(&BO, Builder2);

    Type* addingType = BO.getType();

    switch(BO.getOpcode()) {
      case Instruction::FMul:{
        if (!constantval0)
          dif0 = Builder2.CreateFMul(idiff, lookup(gutils->getNewFromOriginal(orig_op1), Builder2), "m0diffe"+orig_op0->getName());
        if (!constantval1)
          dif1 = Builder2.CreateFMul(idiff, lookup(gutils->getNewFromOriginal(orig_op0), Builder2), "m1diffe"+orig_op1->getName());
        break;
      }
      case Instruction::FAdd:{
        if (!constantval0)
          dif0 = idiff;
        if (!constantval1)
          dif1 = idiff;
        break;
      }
      case Instruction::FSub:{
        if (!constantval0)
          dif0 = idiff;
        if (!constantval1)
          dif1 = Builder2.CreateFNeg(idiff);
        break;
      }
      case Instruction::FDiv:{
        if (!constantval0)
          dif0 = Builder2.CreateFDiv(idiff, lookup(gutils->getNewFromOriginal(orig_op1), Builder2), "d0diffe"+orig_op0->getName());
        if (!constantval1) {
          Value* lop0 = lookup(gutils->getNewFromOriginal(orig_op0), Builder2);
          Value* lop1 = lookup(gutils->getNewFromOriginal(orig_op1), Builder2);
          Value* lastdiv = Builder2.CreateFDiv(lop0, lop1);
          if (auto newi = dyn_cast<Instruction>(lastdiv)) newi->copyIRFlags(&BO);

          dif1 = Builder2.CreateFNeg(
            Builder2.CreateFMul(lastdiv,
              Builder2.CreateFDiv(idiff, lop1)
            )
          );
        }
        break;
      }
      case Instruction::LShr:{
        if (!constantval0) {
          if (auto ci = dyn_cast<ConstantInt>(orig_op1)) {
            if (Type* flt = TR.intType(orig_op0, /*necessary*/false).isFloat()) {
              auto bits = gutils->newFunc->getParent()->getDataLayout().getTypeAllocSizeInBits(flt);
              if (ci->getSExtValue() >= (int64_t)bits && ci->getSExtValue() % bits == 0) {
                dif0 = Builder2.CreateShl(idiff, ci);
                addingType = flt;
                goto done;
              }
            }
          }
        }
        goto def;
      }
      case Instruction::Add:{
        if (looseTypeAnalysis) {
          // if loose type analysis, assume this integer add is constant
          return;
        }
        goto def;
      }
      default:def:;
        llvm::errs() << *gutils->oldFunc << "\n";
        for(auto & pair : gutils->internal_isConstantInstruction) {
          llvm::errs() << " constantinst[" << *pair.first << "] = " << pair.second << " val:" << gutils->internal_isConstantValue[const_cast<Instruction*>(pair.first)] << " type: " << TR.query(const_cast<Instruction*>(pair.first)).str() << "\n";
        }
        llvm::errs() << "cannot handle unknown binary operator: " << BO << "\n";
        report_fatal_error("unknown binary operator");
    }

    done:;
    if (dif0 || dif1) setDiffe(&BO, Constant::getNullValue(BO.getType()), Builder2);
    if (dif0) addToDiffe(orig_op0, dif0, Builder2, addingType);
    if (dif1) addToDiffe(orig_op1, dif1, Builder2, addingType);
  }

  void visitMemSetInst(llvm::MemSetInst &MS) {
    // Don't duplicate set in reverse pass
    if (mode == DerivativeMode::Reverse) {
      erased.insert(&MS);
      gutils->erase(gutils->getNewFromOriginal(&MS));
    }

    if (gutils->isConstantInstruction(&MS)) return;

    Value* orig_op0 = MS.getOperand(0);
    Value* orig_op1 = MS.getOperand(1);
    Value* op1 = gutils->getNewFromOriginal(orig_op1);
    Value* op2 = gutils->getNewFromOriginal(MS.getOperand(2));
    Value* op3 = gutils->getNewFromOriginal(MS.getOperand(3));

    //TODO this should 1) assert that the value being meset is constant
    //                 2) duplicate the memset for the inverted pointer

    if (!gutils->isConstantValue(orig_op1)) {
        llvm::errs() << "couldn't handle non constant inst in memset to propagate differential to\n" << MS;
        report_fatal_error("non constant in memset");
    }

    if (mode == DerivativeMode::Forward || mode == DerivativeMode::Both) {
      IRBuilder <>BuilderZ(gutils->getNewFromOriginal(&MS));

      SmallVector<Value*, 4> args;
      if (!gutils->isConstantValue(orig_op0)) {
        args.push_back(gutils->invertPointerM(orig_op0, BuilderZ));
      } else {
        //If constant destination then no operation needs doing
        return;
        //args.push_back(gutils->lookupM(MS.getOperand(0), BuilderZ));
      }

      args.push_back(gutils->lookupM(op1, BuilderZ));
      args.push_back(gutils->lookupM(op2, BuilderZ));
      args.push_back(gutils->lookupM(op3, BuilderZ));

      Type *tys[] = {args[0]->getType(), args[2]->getType()};
      auto cal = BuilderZ.CreateCall(Intrinsic::getDeclaration(MS.getParent()->getParent()->getParent(), Intrinsic::memset, tys), args);
      cal->setAttributes(MS.getAttributes());
      cal->setCallingConv(MS.getCallingConv());
      cal->setTailCallKind(MS.getTailCallKind());
    }

    if (mode == DerivativeMode::Reverse || mode == DerivativeMode::Both) {
      //TODO consider what reverse pass memset should be
    }
  }

  void subTransferHelper(Type* secretty, BasicBlock* parent, Intrinsic::ID intrinsic, unsigned dstalign, unsigned srcalign, unsigned offset, Value* orig_dst, Value* orig_src, Value* length, Value* isVolatile, llvm::MemTransferInst& MTI) {
    // TODO offset

    if (secretty) {
      //no change to forward pass if represents floats
      if (mode == DerivativeMode::Reverse || mode == DerivativeMode::Both) {
        IRBuilder<> Builder2 = getReverseBuilder(parent);

        // If the src is context simply zero d_dst and don't propagate to d_src (which thus == src and may be illegal)
        if (gutils->isConstantValue(orig_src)) {
          SmallVector<Value*, 4> args;
          args.push_back(gutils->invertPointerM(orig_dst, Builder2));
          args.push_back(ConstantInt::get(Type::getInt8Ty(parent->getContext()), 0));
          args.push_back(lookup(length, Builder2));
          args.push_back(ConstantInt::getFalse(parent->getContext()));

          Type *tys[] = {args[0]->getType(), args[2]->getType()};
          auto memsetIntr = Intrinsic::getDeclaration(parent->getParent()->getParent(), Intrinsic::memset, tys);
          auto cal = Builder2.CreateCall(memsetIntr, args);
          cal->setCallingConv(memsetIntr->getCallingConv());
          if (dstalign != 0) {
            cal->addParamAttr(0, Attribute::getWithAlignment(parent->getContext(), dstalign));
          }

        } else {
          SmallVector<Value*, 4> args;
          auto secretpt = PointerType::getUnqual(secretty);
          auto dsto = gutils->invertPointerM(orig_dst, Builder2);
          if (offset != 0) dsto = Builder2.CreateConstInBoundsGEP1_64(dsto, offset);
          args.push_back(Builder2.CreatePointerCast(dsto, secretpt));
          auto srco = gutils->invertPointerM(orig_src, Builder2);
          if (offset != 0) srco = Builder2.CreateConstInBoundsGEP1_64(srco, offset);
          args.push_back(Builder2.CreatePointerCast(srco, secretpt));
          args.push_back(Builder2.CreateUDiv(lookup(length, Builder2),

              ConstantInt::get(length->getType(), Builder2.GetInsertBlock()->getParent()->getParent()->getDataLayout().getTypeAllocSizeInBits(secretty)/8)
          ));

          auto dmemcpy = ( (intrinsic == Intrinsic::memcpy) ? getOrInsertDifferentialFloatMemcpy : getOrInsertDifferentialFloatMemmove)(*parent->getParent()->getParent(), secretpt, dstalign, srcalign);
          Builder2.CreateCall(dmemcpy, args);
        }
      }
    } else {

      //if represents pointer or integer type then only need to modify forward pass with the copy
      if (mode == DerivativeMode::Forward || mode == DerivativeMode::Both) {

        //It is questionable how the following case would even occur, but if the dst is constant, we shouldn't do anything extra
        if (gutils->isConstantValue(orig_dst)) {
          return;
        }

        SmallVector<Value*, 4> args;
        IRBuilder <>BuilderZ(gutils->getNewFromOriginal(&MTI));

        //If src is inactive, then we should copy from the regular pointer (i.e. suppose we are copying constant memory representing dimensions into a tensor)
        //  to ensure that the differential tensor is well formed for use OUTSIDE the derivative generation (as enzyme doesn't need this), we should also perform the copy
        //  onto the differential. Future Optimization (not implemented): If dst can never escape Enzyme code, we may omit this copy.
        //no need to update pointers, even if dst is active
        auto dsto = gutils->invertPointerM(orig_dst, BuilderZ);
        if (offset != 0) dsto = BuilderZ.CreateConstInBoundsGEP1_64(dsto, offset);
        args.push_back(dsto);
        auto srco = gutils->invertPointerM(orig_src, BuilderZ);
        if (offset != 0) srco = BuilderZ.CreateConstInBoundsGEP1_64(srco, offset);
        args.push_back(srco);

        args.push_back(length);
        args.push_back(isVolatile);

        Type *tys[] = {args[0]->getType(), args[1]->getType(), args[2]->getType()};
        auto memtransIntr = Intrinsic::getDeclaration(gutils->newFunc->getParent(), intrinsic, tys);
        auto cal = BuilderZ.CreateCall(memtransIntr, args);
        cal->setAttributes(MTI.getAttributes());
        cal->setCallingConv(memtransIntr->getCallingConv());
        cal->setTailCallKind(MTI.getTailCallKind());

        if (dstalign != 0) {
          cal->addParamAttr(0, Attribute::getWithAlignment(parent->getContext(), dstalign));
        }
        if (srcalign != 0) {
          cal->addParamAttr(1, Attribute::getWithAlignment(parent->getContext(), srcalign));
        }
      }
    }
  }

  void visitMemTransferInst(llvm::MemTransferInst& MTI) {
    if (gutils->isConstantInstruction(&MTI)) {
      eraseIfUnused(MTI);
      return;
    }

    if (unnecessaryStores.count(&MTI)) {
      eraseIfUnused(MTI);
      return;
    }


    Value* orig_op0 = MTI.getOperand(0);
    Value* orig_op1 = MTI.getOperand(1);
    Value* op2 = gutils->getNewFromOriginal(MTI.getOperand(2));
    Value* op3 = gutils->getNewFromOriginal(MTI.getOperand(3));

    // copying into nullptr is invalid (not sure why it exists here), but we shouldn't do it in reverse pass or shadow
    if (isa<ConstantPointerNull>(orig_op0) || TR.query(orig_op0).Data0()[{}] == IntType::Anything) {
      eraseIfUnused(MTI);
      return;
    }

    size_t size = 1;
    if (auto ci = dyn_cast<ConstantInt>(op2)) {
      size = ci->getLimitedValue();
    }

    //TODO note that we only handle memcpy/etc of ONE type (aka memcpy of {int, double} not allowed)

    //llvm::errs() << *gutils->oldFunc << "\n";
    //TR.dump();


    auto vd = TR.query(orig_op0).Data0().AtMost(size);
    vd |= TR.query(orig_op1).Data0().AtMost(size);

    //llvm::errs() << "MIT: " << MTI << "|size: " << size << " vd: " << vd.str() << "\n";

    if (!vd.isKnownPastPointer()) {
      if (looseTypeAnalysis) {
        if (isa<CastInst>(orig_op0) && cast<CastInst>(orig_op0)->getSrcTy()->isPointerTy() && cast<PointerType>(cast<CastInst>(orig_op0)->getSrcTy())->getElementType()->isFPOrFPVectorTy()) {
          vd = ValueData(DataType(cast<PointerType>(cast<CastInst>(orig_op0)->getSrcTy())->getElementType()->getScalarType())).Only({0});
          goto known;
        }
      }
      llvm::errs() << "cannot deduce type for mti: " << MTI << " " << *orig_op0 << "\n";
      TR.firstPointer(size, orig_op0, /*errifnotfound*/true, /*pointerIntSame*/true);
      assert(0 && "bad mti");
    }
    known:;

    unsigned dstalign = 0;
    if (MTI.paramHasAttr(0, Attribute::Alignment)) {
        dstalign = MTI.getParamAttr(0, Attribute::Alignment).getValueAsInt();
    }
    unsigned srcalign = 0;
    if (MTI.paramHasAttr(1, Attribute::Alignment)) {
        srcalign = MTI.getParamAttr(1, Attribute::Alignment).getValueAsInt();
    }

    unsigned start = 0;

    IRBuilder <>BuilderZ(gutils->getNewFromOriginal(&MTI));

    while(1) {
      unsigned nextStart = size;

      auto dt = vd[{-1}];
      for(size_t i=start; i<size; i++) {
          bool legal = true;
          dt.legalMergeIn(vd[{(int)i}], /*pointerIntSame*/true, legal);
          if (!legal) {
            nextStart = i;
            break;
          }
      }
      if (!dt.isKnown()) {
        TR.dump();
        llvm::errs() << " vd:" << vd.str() << " start:" << start << " size: " << size << " dt:" << dt.str() << "\n";
      }
      assert(dt.isKnown());

      Value* length = op2;
      if (nextStart != size) {
        length = ConstantInt::get(op2->getType(), nextStart);
      }
      if (start != 0)
        length = BuilderZ.CreateSub(length, ConstantInt::get(op2->getType(), start));

      unsigned subdstalign = dstalign;
      // todo make better alignment calculation
      if (dstalign != 0) {
        if (start % dstalign != 0) {
          dstalign = 1;
        }
      }
      unsigned subsrcalign = srcalign;
      // todo make better alignment calculation
      if (srcalign != 0) {
        if (start % srcalign != 0) {
          srcalign = 1;
        }
      }

      subTransferHelper(dt.isFloat(), MTI.getParent(), MTI.getIntrinsicID(), subdstalign, subsrcalign, /*offset*/start, orig_op0, orig_op1, /*length*/length, /*volatile*/op3, MTI);

      if (nextStart == size) break;
      start = nextStart;
    }

    eraseIfUnused(MTI);
  }

  void visitIntrinsicInst(llvm::IntrinsicInst &II) {
    if (II.getIntrinsicID() == Intrinsic::stacksave) {
      eraseIfUnused(II, /*erase*/true, /*check*/false);
      return;
    }
    if (II.getIntrinsicID() == Intrinsic::stackrestore || II.getIntrinsicID() == Intrinsic::lifetime_end) {
      eraseIfUnused(II, /*erase*/true, /*check*/false);
      return;
    }

    eraseIfUnused(II);
    Value* orig_ops[II.getNumOperands()];

    for(unsigned i=0; i<II.getNumOperands(); i++) {
      orig_ops[i] = II.getOperand(i);
    }

    if (mode == DerivativeMode::Forward) {
      switch(II.getIntrinsicID()) {
        case Intrinsic::prefetch:
        case Intrinsic::dbg_declare:
        case Intrinsic::dbg_value:
        #if LLVM_VERSION_MAJOR > 6
        case Intrinsic::dbg_label:
        #endif
        case Intrinsic::dbg_addr:
        case Intrinsic::lifetime_start:
        case Intrinsic::assume:
        case Intrinsic::fabs:
        case Intrinsic::x86_sse_max_ss:
        case Intrinsic::x86_sse_max_ps:
        case Intrinsic::maxnum:
        case Intrinsic::x86_sse_min_ss:
        case Intrinsic::x86_sse_min_ps:
        case Intrinsic::minnum:
        case Intrinsic::log:
        case Intrinsic::log2:
        case Intrinsic::log10:
        case Intrinsic::exp:
        case Intrinsic::exp2:
        case Intrinsic::pow:
        case Intrinsic::sin:
        case Intrinsic::cos:
        case Intrinsic::floor:
        case Intrinsic::ceil:
        case Intrinsic::trunc:
        case Intrinsic::rint:
        case Intrinsic::nearbyint:
        case Intrinsic::round:
        case Intrinsic::sqrt:
          return;
        default:
          if (gutils->isConstantInstruction(&II)) return;
          llvm::errs() << *gutils->oldFunc << "\n";
          llvm::errs() << *gutils->newFunc << "\n";
          llvm::errs() << "cannot handle (augmented) unknown intrinsic\n" << II;
          report_fatal_error("(augmented) unknown intrinsic");
      }
    }

    if (mode == DerivativeMode::Both || mode == DerivativeMode::Reverse) {

      IRBuilder<> Builder2 = getReverseBuilder(II.getParent());
      Module* M = II.getParent()->getParent()->getParent();

      Value* vdiff = nullptr;
      if (!gutils->isConstantValue(&II)) {
        vdiff = diffe(&II, Builder2);
        setDiffe(&II, Constant::getNullValue(II.getType()), Builder2);
      }

      switch(II.getIntrinsicID()) {
        case Intrinsic::assume:
        case Intrinsic::prefetch:
        case Intrinsic::dbg_declare:
        case Intrinsic::dbg_value:
        #if LLVM_VERSION_MAJOR > 6
        case Intrinsic::dbg_label:
        #endif
        case Intrinsic::dbg_addr:
        case Intrinsic::floor:
        case Intrinsic::ceil:
        case Intrinsic::trunc:
        case Intrinsic::rint:
        case Intrinsic::nearbyint:
        case Intrinsic::round:
          //Derivative of these is zero and requires no modification
          return;

        case Intrinsic::lifetime_start:{
          if (gutils->isConstantInstruction(&II)) return;
          SmallVector<Value*, 2> args = {lookup(gutils->getNewFromOriginal(orig_ops[0]), Builder2), lookup(gutils->getNewFromOriginal(orig_ops[1]), Builder2)};
          Type *tys[] = {args[1]->getType()};
          auto cal = Builder2.CreateCall(Intrinsic::getDeclaration(M, Intrinsic::lifetime_end, tys), args);
          cal->setAttributes(II.getAttributes());
          cal->setCallingConv(II.getCallingConv());
          cal->setTailCallKind(II.getTailCallKind());
          return;
        }

        case Intrinsic::sqrt: {
          if (vdiff && !gutils->isConstantValue(orig_ops[0])) {
            SmallVector<Value*, 2> args = {lookup(gutils->getNewFromOriginal(orig_ops[0]), Builder2)};
            Type *tys[] = {orig_ops[0]->getType()};
            auto cal = cast<CallInst>(Builder2.CreateCall(Intrinsic::getDeclaration(M, II.getIntrinsicID(), tys), args));
            cal->copyIRFlags(&II);
            cal->setAttributes(II.getAttributes());
            cal->setCallingConv(II.getCallingConv());
            cal->setTailCallKind(II.getTailCallKind());
            cal->setDebugLoc(II.getDebugLoc());

            Value* dif0 = Builder2.CreateBinOp(Instruction::FDiv,
              Builder2.CreateFMul(ConstantFP::get(II.getType(), 0.5), vdiff),
              cal
            );

            Value* cmp  = Builder2.CreateFCmpOEQ(args[0], ConstantFP::get(orig_ops[0]->getType(), 0));
            dif0 = Builder2.CreateSelect(cmp, ConstantFP::get(orig_ops[0]->getType(), 0), dif0);

            addToDiffe(orig_ops[0], dif0, Builder2, II.getType());
          }
          return;
        }

        case Intrinsic::fabs: {
          if (vdiff && !gutils->isConstantValue(orig_ops[0])) {
            Value* cmp  = Builder2.CreateFCmpOLT(lookup(gutils->getNewFromOriginal(orig_ops[0]), Builder2), ConstantFP::get(orig_ops[0]->getType(), 0));
            Value* dif0 = Builder2.CreateFMul(Builder2.CreateSelect(cmp, ConstantFP::get(orig_ops[0]->getType(), -1), ConstantFP::get(orig_ops[0]->getType(), 1)), vdiff);
            addToDiffe(orig_ops[0], dif0, Builder2, II.getType());
          }
          return;
        }

        case Intrinsic::x86_sse_max_ss:
        case Intrinsic::x86_sse_max_ps:
        case Intrinsic::maxnum: {
          if (vdiff && !gutils->isConstantValue(orig_ops[0])) {
            Value* cmp  = Builder2.CreateFCmpOLT(lookup(gutils->getNewFromOriginal(orig_ops[0]), Builder2), lookup(gutils->getNewFromOriginal(orig_ops[1]), Builder2));
            Value* dif0 = Builder2.CreateSelect(cmp, ConstantFP::get(orig_ops[0]->getType(), 0), vdiff);
            addToDiffe(orig_ops[0], dif0, Builder2, II.getType());
          }
          if (vdiff && !gutils->isConstantValue(orig_ops[1])) {
            Value* cmp  = Builder2.CreateFCmpOLT(lookup(gutils->getNewFromOriginal(orig_ops[0]), Builder2), lookup(gutils->getNewFromOriginal(orig_ops[1]), Builder2));
            Value* dif1 = Builder2.CreateSelect(cmp, vdiff, ConstantFP::get(orig_ops[0]->getType(), 0));
            addToDiffe(orig_ops[1], dif1, Builder2, II.getType());
          }
          return;
        }

        case Intrinsic::x86_sse_min_ss:
        case Intrinsic::x86_sse_min_ps:
        case Intrinsic::minnum: {
          if (vdiff && !gutils->isConstantValue(orig_ops[0])) {
            Value* cmp = Builder2.CreateFCmpOLT(lookup(gutils->getNewFromOriginal(orig_ops[0]), Builder2), lookup(gutils->getNewFromOriginal(orig_ops[1]), Builder2));
            Value* dif0 = Builder2.CreateSelect(cmp, vdiff, ConstantFP::get(orig_ops[0]->getType(), 0));
            addToDiffe(orig_ops[0], dif0, Builder2, II.getType());
          }
          if (vdiff && !gutils->isConstantValue(orig_ops[1])) {
            Value* cmp = Builder2.CreateFCmpOLT(lookup(gutils->getNewFromOriginal(orig_ops[0]), Builder2), lookup(gutils->getNewFromOriginal(orig_ops[1]), Builder2));
            Value* dif1 = Builder2.CreateSelect(cmp, ConstantFP::get(orig_ops[0]->getType(), 0), vdiff);
            addToDiffe(orig_ops[1], dif1, Builder2, II.getType());
          }
          return;
        }

        case Intrinsic::log: {
          if (vdiff && !gutils->isConstantValue(orig_ops[0])) {
            Value* dif0 = Builder2.CreateFDiv(vdiff, lookup(gutils->getNewFromOriginal(orig_ops[0]), Builder2));
            addToDiffe(orig_ops[0], dif0, Builder2, II.getType());
          }
          return;
        }

        case Intrinsic::log2: {
          if (vdiff && !gutils->isConstantValue(orig_ops[0])) {
            Value* dif0 = Builder2.CreateFDiv(vdiff,
              Builder2.CreateFMul(ConstantFP::get(II.getType(), 0.6931471805599453), lookup(gutils->getNewFromOriginal(orig_ops[0]), Builder2))
            );
            addToDiffe(orig_ops[0], dif0, Builder2, II.getType());
          }
          return;
        }
        case Intrinsic::log10: {
          if (vdiff && !gutils->isConstantValue(orig_ops[0])) {
            Value* dif0 = Builder2.CreateFDiv(vdiff,
              Builder2.CreateFMul(ConstantFP::get(II.getType(), 2.302585092994046), lookup(gutils->getNewFromOriginal(orig_ops[0]), Builder2))
            );
            addToDiffe(orig_ops[0], dif0, Builder2, II.getType());
          }
          return;
        }

        case Intrinsic::exp: {
          if (vdiff && !gutils->isConstantValue(orig_ops[0])) {
            SmallVector<Value*, 2> args = {lookup(gutils->getNewFromOriginal(orig_ops[0]), Builder2)};
            Type *tys[] = {orig_ops[0]->getType()};
            auto cal = cast<CallInst>(Builder2.CreateCall(Intrinsic::getDeclaration(M, II.getIntrinsicID(), tys), args));
            cal->copyIRFlags(&II);
            cal->setAttributes(II.getAttributes());
            cal->setCallingConv(II.getCallingConv());
            cal->setTailCallKind(II.getTailCallKind());
            cal->setDebugLoc(II.getDebugLoc());

            Value* dif0 = Builder2.CreateFMul(vdiff, lookup(cal, Builder2));
            addToDiffe(orig_ops[0], dif0, Builder2, II.getType());
          }
          return;
        }
        case Intrinsic::exp2: {
          if (vdiff && !gutils->isConstantValue(orig_ops[0])) {
            SmallVector<Value*, 2> args = {lookup(gutils->getNewFromOriginal(orig_ops[0]), Builder2)};
            Type *tys[] = {orig_ops[0]->getType()};
            auto cal = cast<CallInst>(Builder2.CreateCall(Intrinsic::getDeclaration(M, II.getIntrinsicID(), tys), args));
            cal->copyIRFlags(&II);
            cal->setAttributes(II.getAttributes());
            cal->setCallingConv(II.getCallingConv());
            cal->setTailCallKind(II.getTailCallKind());
            cal->setDebugLoc(II.getDebugLoc());

            Value* dif0 = Builder2.CreateFMul(
              Builder2.CreateFMul(vdiff, lookup(cal, Builder2)), ConstantFP::get(II.getType(), 0.6931471805599453)
            );
            addToDiffe(orig_ops[0], dif0, Builder2, II.getType());
          }
          return;
        }
        case Intrinsic::pow: {
          if (vdiff && !gutils->isConstantValue(orig_ops[0])) {

            Value* op0 = gutils->getNewFromOriginal(orig_ops[0]);
            Value* op1 = gutils->getNewFromOriginal(orig_ops[1]);
            /*
            dif0 = Builder2.CreateFMul(
              Builder2.CreateFMul(vdiff,
                Builder2.CreateFDiv(lookup(&II), lookup(II.getOperand(0)))), lookup(II.getOperand(1))
            );
            */
            SmallVector<Value*, 2> args = {lookup(op0, Builder2), Builder2.CreateFSub(lookup(op1, Builder2), ConstantFP::get(II.getType(), 1.0))};
            Type *tys[] = {orig_ops[0]->getType()};
            auto cal = cast<CallInst>(Builder2.CreateCall(Intrinsic::getDeclaration(M, Intrinsic::pow, tys), args));
            cal->copyIRFlags(&II);
            cal->setAttributes(II.getAttributes());
            cal->setCallingConv(II.getCallingConv());
            cal->setTailCallKind(II.getTailCallKind());
            cal->setDebugLoc(II.getDebugLoc());

            Value* dif0 = Builder2.CreateFMul(
              Builder2.CreateFMul(vdiff, cal)
              , lookup(op1, Builder2)
            );
            addToDiffe(orig_ops[0], dif0, Builder2, II.getType());
          }

          if (vdiff && !gutils->isConstantValue(orig_ops[1])) {

            CallInst* cal;
            {
            SmallVector<Value*, 2> args = {lookup(gutils->getNewFromOriginal(orig_ops[0]), Builder2), lookup(gutils->getNewFromOriginal(orig_ops[1]), Builder2)};

            Type *tys[] = {orig_ops[0]->getType()};
            cal = cast<CallInst>(Builder2.CreateCall(Intrinsic::getDeclaration(M, II.getIntrinsicID(), tys), args));
            cal->copyIRFlags(&II);
            cal->setAttributes(II.getAttributes());
            cal->setCallingConv(II.getCallingConv());
            cal->setTailCallKind(II.getTailCallKind());
            cal->setDebugLoc(II.getDebugLoc());
            }

            Value *args[] = {lookup(gutils->getNewFromOriginal(orig_ops[0]), Builder2)};
            Type *tys[] = {orig_ops[0]->getType()};

            Value* dif1 = Builder2.CreateFMul(
              Builder2.CreateFMul(vdiff, cal),
              Builder2.CreateCall(Intrinsic::getDeclaration(M, Intrinsic::log, tys), args)
            );
            addToDiffe(orig_ops[1], dif1, Builder2, II.getType());
          }

          return;
        }
        case Intrinsic::sin: {
          if (vdiff && !gutils->isConstantValue(orig_ops[0])) {
            Value *args[] = {lookup(gutils->getNewFromOriginal(orig_ops[0]), Builder2)};
            Type *tys[] = {orig_ops[0]->getType()};
            CallInst* cal = cast<CallInst>(Builder2.CreateCall(Intrinsic::getDeclaration(M, Intrinsic::cos, tys), args));
            cal->setTailCallKind(II.getTailCallKind());
            Value* dif0 = Builder2.CreateFMul(vdiff, cal);
            addToDiffe(orig_ops[0], dif0, Builder2, II.getType());
          }
          return;
        }
        case Intrinsic::cos: {
          if (vdiff && !gutils->isConstantValue(orig_ops[0])) {
            Value *args[] = {lookup(gutils->getNewFromOriginal(orig_ops[0]), Builder2)};
            Type *tys[] = {orig_ops[0]->getType()};
            CallInst* cal = cast<CallInst>(Builder2.CreateCall(Intrinsic::getDeclaration(M, Intrinsic::sin, tys), args));
            cal->setTailCallKind(II.getTailCallKind());
            Value* dif0 = Builder2.CreateFMul(vdiff,
              Builder2.CreateFNeg(cal)
            );
            addToDiffe(orig_ops[0], dif0, Builder2, II.getType());
          }
          return;
        }

        default:
          if (gutils->isConstantInstruction(&II)) return;
          llvm::errs() << *gutils->oldFunc << "\n";
          llvm::errs() << *gutils->newFunc << "\n";
          llvm::errs() << "cannot handle (augmented) unknown intrinsic\n" << II;
          report_fatal_error("(augmented) unknown intrinsic");
      }
    }

    llvm::InstVisitor<DerivativeMaker<AugmentedReturnType>>::visitIntrinsicInst(II);
  }

  // Return
  void visitCallInst(llvm::CallInst &call) {

    IRBuilder<> BuilderZ(gutils->getNewFromOriginal(&call));
    BuilderZ.setFastMathFlags(getFast());

    if(uncacheable_args_map.find(&call) == uncacheable_args_map.end()) {
      llvm::errs() << " call: " << call << "\n";
      for(auto & pair: uncacheable_args_map) {
        llvm::errs() << " + " << *pair.first << "\n";
      }
    }

    assert(uncacheable_args_map.find(&call) != uncacheable_args_map.end());
    const std::map<Argument*, bool> &uncacheable_args = uncacheable_args_map.find(&call)->second;

    CallInst* orig = &call;

    Function *called = orig->getCalledFunction();

    if (auto castinst = dyn_cast<ConstantExpr>(orig->getCalledValue())) {
        if (castinst->isCast())
        if (auto fn = dyn_cast<Function>(castinst->getOperand(0))) {
            if (isAllocationFunction(*called, gutils->TLI) || isDeallocationFunction(*called, gutils->TLI)) {
                called = fn;
            }
        }
    }

    if (called && (called->getName() == "printf" || called->getName() == "puts")) {
      if (mode == DerivativeMode::Reverse) {
        eraseIfUnused(*orig, /*erase*/true, /*check*/false);
      }
      return;
    }

    // Handle lgamma, safe to recompute so no store/change to forward
    if (called) {
      auto n = called->getName();

      if (called && (called->getName() == "tanhf" || called->getName() == "tanh")) {
        if (mode == DerivativeMode::Forward || gutils->isConstantValue(orig)) return;

        IRBuilder<> Builder2 = getReverseBuilder(call.getParent());
        Value* x  = lookup(gutils->getNewFromOriginal(orig->getArgOperand(0)), Builder2);

        SmallVector<Value*, 1> args = { x };
        auto coshf = gutils->oldFunc->getParent()->getOrInsertFunction( (called->getName() == "tanh") ? "cosh" : "coshf", called->getFunctionType(), called->getAttributes());
        auto cal = cast<CallInst>(Builder2.CreateCall(coshf, args));
        Value* dif0 = Builder2.CreateFDiv(diffe(orig, Builder2), Builder2.CreateFMul(cal, cal));
        setDiffe(orig, Constant::getNullValue(orig->getType()), Builder2);
        addToDiffe(orig->getArgOperand(0), dif0, Builder2, x->getType());
        return;
      }

      if (n == "lgamma" || n == "lgammaf" || n == "lgammal" || n == "lgamma_r" || n == "lgammaf_r" || n == "lgammal_r"
        || n == "__lgamma_r_finite" || n == "__lgammaf_r_finite" || n == "__lgammal_r_finite") {
        if (mode == DerivativeMode::Forward || gutils->isConstantValue(orig)) {
          return;
        }
      }
    }

    if (called && isAllocationFunction(*called, gutils->TLI)) {

      bool constval = gutils->isConstantValue(orig);

      if (!constval) {
        auto anti = gutils->createAntiMalloc(orig, getIndex(orig, CacheType::Shadow));
        if (mode == DerivativeMode::Both || mode == DerivativeMode::Reverse) {
          IRBuilder<> Builder2 = getReverseBuilder(call.getParent());
          Value* tofree = lookup(anti, Builder2);
          assert(tofree);
          assert(tofree->getType());
          assert(Type::getInt8Ty(tofree->getContext()));
          assert(PointerType::getUnqual(Type::getInt8Ty(tofree->getContext())));
          assert(Type::getInt8PtrTy(tofree->getContext()));
          freeKnownAllocation(Builder2, tofree, *called, gutils->TLI)->addAttribute(AttributeList::FirstArgIndex, Attribute::NonNull);
        }
      }
      
      CallInst* const op = cast<CallInst>(gutils->getNewFromOriginal(&call));
      //TODO enable this if we need to free the memory
      // NOTE THAT TOPLEVEL IS THERE SIMPLY BECAUSE THAT WAS PREVIOUS ATTITUTE TO FREE'ing
      if (mode != DerivativeMode::Both) {
        if (is_value_needed_in_reverse(TR, gutils, orig, /*topLevel*/mode == DerivativeMode::Both)) {
          
          gutils->addMalloc(BuilderZ, op, getIndex(orig, CacheType::Self) );
        } else if (mode != DerivativeMode::Forward) {
          //Note that here we cannot simply replace with null as users who try to find the shadow pointer will use the shadow of null rather than the true shadow of this
          auto pn = BuilderZ.CreatePHI(orig->getType(), 1, (orig->getName()+"_replacementB").str());
          gutils->fictiousPHIs.push_back(pn);
          gutils->replaceAWithB(op, pn);
          gutils->erase(op);
        }
      } else {
        IRBuilder<> Builder2 = getReverseBuilder(call.getParent());
        freeKnownAllocation(Builder2, lookup(op, Builder2), *called, gutils->TLI);
      }

      return;
    }

    //Remove free's in forward pass so the memory can be used in the reverse pass
    if (called && isDeallocationFunction(*called, gutils->TLI)) {
      if( gutils->invertedPointers.count(orig) ) {
          auto placeholder = cast<PHINode>(gutils->invertedPointers[orig]);
          gutils->invertedPointers.erase(orig);
          gutils->erase(placeholder);
      }

      llvm::Value* val = orig->getArgOperand(0);
      while(auto cast = dyn_cast<CastInst>(val)) val = cast->getOperand(0);

      if (auto dc = dyn_cast<CallInst>(val)) {
        if (dc->getCalledFunction() && isAllocationFunction(*dc->getCalledFunction(), gutils->TLI)) {
          //llvm::errs() << "erasing free(orig): " << *orig << "\n";
          eraseIfUnused(*orig, /*erase*/true, /*check*/false);
          return;
        }
      }

      if (isa<ConstantPointerNull>(val)) {
        llvm::errs() << "removing free of null pointer\n";
        eraseIfUnused(*orig, /*erase*/true, /*check*/false);
        return;
      }

      //TODO HANDLE FREE
      llvm::errs() << "freeing without malloc " << *val << "\n";
      eraseIfUnused(*orig, /*erase*/true, /*check*/false);
      return;
    }

    bool subretused = unnecessaryValues.find(orig) == unnecessaryValues.end();

    if (gutils->isConstantInstruction(orig)) {
      // If we need this value and it is illegal to recompute it (it writes or may load uncacheable data)
      //    Store and reload it
      if (mode != DerivativeMode::Both && subretused && !orig->doesNotAccessMemory()) {
        CallInst* const op = cast<CallInst>(gutils->getNewFromOriginal(&call));
        gutils->addMalloc(BuilderZ, op, getIndex(orig, CacheType::Self));
        return;
      }

      // If this call may write to memory and is a copy (in the just reverse pass), erase it
      //  Any uses of it should be handled by the case above so it is safe to RAUW
      if (orig->mayWriteToMemory() && mode == DerivativeMode::Reverse) {
        eraseIfUnused(*orig, /*erase*/true, /*check*/false);
        return;
      }

      // if call does not write memory and isn't used, we can erase it
      if (!orig->mayWriteToMemory() && !subretused) {
        eraseIfUnused(*orig, /*erase*/true, /*check*/false);
        return;
      }

      return;
    }

    bool modifyPrimal = shouldAugmentCall(orig, gutils, TR);

    bool foreignFunction = called == nullptr || called->empty();

    SmallVector<Value*, 8> args;
    SmallVector<Value*, 8> pre_args;
    std::vector<DIFFE_TYPE> argsInverted;
    std::vector<Instruction*> postCreate;
    std::vector<Instruction*> userReplace;

    for(unsigned i=0;i<orig->getNumArgOperands(); i++) {

      auto argi = gutils->getNewFromOriginal(orig->getArgOperand(i));

      pre_args.push_back(argi);

      if (mode != DerivativeMode::Forward) {
        IRBuilder<> Builder2 = getReverseBuilder(call.getParent());
        args.push_back(lookup(argi, Builder2));
      }


      if (gutils->isConstantValue(orig->getArgOperand(i)) && !foreignFunction) {
        argsInverted.push_back(DIFFE_TYPE::CONSTANT);
        continue;
      }

      auto argType = argi->getType();


      if (!argType->isFPOrFPVectorTy() && TR.query(orig->getArgOperand(i)).Data0()[{}].isPossiblePointer()) {
        DIFFE_TYPE ty = DIFFE_TYPE::DUP_ARG;
        if (argType->isPointerTy()) {
          auto at = GetUnderlyingObject(orig->getArgOperand(i), gutils->oldFunc->getParent()->getDataLayout(), 100);
          if (auto arg = dyn_cast<Argument>(at)) {
            if (constant_args[arg->getArgNo()] == DIFFE_TYPE::DUP_NONEED) {
              ty = DIFFE_TYPE::DUP_NONEED;
            }
          }
        }
        argsInverted.push_back(ty);

        if (mode != DerivativeMode::Forward) {
          IRBuilder<> Builder2 = getReverseBuilder(call.getParent());
          args.push_back(gutils->invertPointerM(orig->getArgOperand(i), Builder2));
        }
        pre_args.push_back(gutils->invertPointerM(orig->getArgOperand(i), BuilderZ));

        //Note sometimes whattype mistakenly says something should be constant [because composed of integer pointers alone]
        assert(whatType(argType) == DIFFE_TYPE::DUP_ARG || whatType(argType) == DIFFE_TYPE::CONSTANT);
      } else {
        argsInverted.push_back(DIFFE_TYPE::OUT_DIFF);
        assert(whatType(argType) == DIFFE_TYPE::OUT_DIFF || whatType(argType) == DIFFE_TYPE::CONSTANT);
      }
    }

    DIFFE_TYPE subretType;
    if (gutils->isConstantValue(orig)) {
      subretType = DIFFE_TYPE::CONSTANT;
    } else if (!orig->getType()->isFPOrFPVectorTy() && TR.query(orig).Data0()[{}].isPossiblePointer()) {
      subretType = DIFFE_TYPE::DUP_ARG;
      // TODO interprocedural dup_noneed
    } else {
      subretType = DIFFE_TYPE::OUT_DIFF;
    }

    bool replaceFunction = false;

    if (mode == DerivativeMode::Both && !foreignFunction) {
      replaceFunction = legalCombinedForwardReverse(orig, *replacedReturns, postCreate, userReplace, gutils, TR, unnecessaryInstructions, subretused);
      if (replaceFunction) modifyPrimal = false;
    }

    Value* tape = nullptr;
    CallInst* augmentcall = nullptr;
    Value* cachereplace = nullptr;

    NewFnTypeInfo nextTypeInfo(called);
    int argnum = 0;

    if (called) {
        std::map<Value*, std::set<int64_t>> intseen;

        for(auto &arg : called->args()) {
          nextTypeInfo.first.insert(std::pair<Argument*, ValueData>(&arg, TR.query(orig->getArgOperand(argnum))));
          nextTypeInfo.knownValues.insert(std::pair<Argument*, std::set<int64_t>>(&arg, TR.isConstantInt(orig->getArgOperand(argnum))));

          argnum++;
        }
        nextTypeInfo.second = TR.query(orig);
    }

    //llvm::Optional<std::map<std::pair<Instruction*, std::string>, unsigned>> sub_index_map;
    int tapeIdx = 0xDEADBEEF;
    int returnIdx = 0xDEADBEEF;
    int differetIdx = 0xDEADBEEF;

    const AugmentedReturn* subdata = nullptr;
    if (mode == DerivativeMode::Reverse) {
      assert(augmentedReturn);
      if (augmentedReturn) {
          auto fd = augmentedReturn->subaugmentations.find(&call);
          if (fd != augmentedReturn->subaugmentations.end()) {
              subdata = fd->second;
          }
      }
    }

    if (modifyPrimal) {

      Value* newcalled = nullptr;
      const AugmentedReturn* fnandtapetype = nullptr;

      if (!called) {
          newcalled = gutils->invertPointerM(orig->getCalledValue(), BuilderZ);

          auto ft = cast<FunctionType>(cast<PointerType>(orig->getCalledValue()->getType())->getElementType());

          DIFFE_TYPE subretType = orig->getType()->isFPOrFPVectorTy() ? DIFFE_TYPE::OUT_DIFF : DIFFE_TYPE::DUP_ARG;
          if (orig->getType()->isVoidTy() || orig->getType()->isEmptyTy()) subretType = DIFFE_TYPE::CONSTANT;
          auto res = getDefaultFunctionTypeForAugmentation(ft, /*returnUsed*/true, /*subretType*/subretType);
          auto fptype = PointerType::getUnqual(FunctionType::get(StructType::get(newcalled->getContext(), res.second), res.first, ft->isVarArg()));
          newcalled = BuilderZ.CreatePointerCast(newcalled, PointerType::getUnqual(fptype));
          newcalled = BuilderZ.CreateLoad(newcalled);
          tapeIdx = 0;

          if (subretType == DIFFE_TYPE::DUP_ARG || subretType == DIFFE_TYPE::DUP_NONEED) {
              returnIdx = 1;
              differetIdx = 2;
          }

      } else {
          if (mode == DerivativeMode::Forward || mode == DerivativeMode::Both) {
              subdata = &CreateAugmentedPrimal(cast<Function>(called), subretType, argsInverted, gutils->TLI, TR.analysis, gutils->AA, /*return is used*/subretused, nextTypeInfo, uncacheable_args, false);
              if (mode == DerivativeMode::Forward) {
                assert(augmentedReturn);
                auto subaugmentations = (std::map<const llvm::CallInst*, AugmentedReturn*>*)&augmentedReturn->subaugmentations;
                insert_or_assign(*subaugmentations, orig, (AugmentedReturn*)subdata);
              }
          }
          if (!subdata) {
              llvm::errs() << *gutils->oldFunc->getParent() << "\n";
              llvm::errs() << *gutils->oldFunc << "\n";
              llvm::errs() << *gutils->newFunc << "\n";
              llvm::errs() << *called << "\n";
          }
          assert(subdata);
          fnandtapetype = subdata;
          newcalled = subdata->fn;

          auto found = subdata->returns.find(AugmentedStruct::DifferentialReturn);
          if (found != subdata->returns.end()) {
              differetIdx = found->second;
          }

          found = subdata->returns.find(AugmentedStruct::Return);
          if (found != subdata->returns.end()) {
              returnIdx = found->second;
          }

          found = subdata->returns.find(AugmentedStruct::Tape);
          if (found != subdata->returns.end()) {
              tapeIdx = found->second;
          }

      }
          //sub_index_map = fnandtapetype.tapeIndices;

      assert(newcalled);
      FunctionType* FT = cast<FunctionType>(cast<PointerType>(newcalled->getType())->getElementType());

          //llvm::errs() << "seeing sub_index_map of " << sub_index_map->size() << " in ap " << cast<Function>(called)->getName() << "\n";
          if (mode == DerivativeMode::Both || mode == DerivativeMode::Forward) {

            if (false) {
          badaugmentedfn:;
                  auto NC = dyn_cast<Function>(newcalled);
                  llvm::errs() << *gutils->oldFunc << "\n";
                  llvm::errs() << *gutils->newFunc << "\n";
                  if (NC)
                  llvm::errs() << " trying to call " << NC->getName() << " " << *FT << "\n";
                  else
                  llvm::errs() << " trying to call " << *newcalled << " " << *FT << "\n";

                  for(unsigned i=0; i<pre_args.size(); i++) {
                      assert(pre_args[i]);
                      assert(pre_args[i]->getType());
                      llvm::errs() << "args[" << i << "] = " << *pre_args[i] << " FT:" << *FT->getParamType(i) << "\n";
                  }
                  assert(0 && "calling with wrong number of arguments");
                  exit(1);

            }

            if (pre_args.size() != FT->getNumParams()) goto badaugmentedfn;

            for(unsigned i=0; i<pre_args.size(); i++) {
              if (pre_args[i]->getType() != FT->getParamType(i)) goto badaugmentedfn;
            }

            augmentcall = BuilderZ.CreateCall(newcalled, pre_args);
            augmentcall->setCallingConv(orig->getCallingConv());
            augmentcall->setDebugLoc(orig->getDebugLoc());

            if (!augmentcall->getType()->isVoidTy())
              augmentcall->setName(orig->getName()+"_augmented");

            if (tapeIdx != 0xDEADBEEF) {
              tape = (tapeIdx == -1) ? augmentcall : BuilderZ.CreateExtractValue(augmentcall, {(unsigned)tapeIdx}, "subcache");
              if (tape->getType()->isEmptyTy()) {
                auto tt = tape->getType();
                gutils->erase(cast<Instruction>(tape));
                tape = UndefValue::get(tt);
              }
              tape = gutils->addMalloc(BuilderZ, tape, getIndex(orig, CacheType::Tape) );
            }

            if (subretused) {
              CallInst* const op = cast<CallInst>(gutils->getNewFromOriginal(&call));

              Value* dcall = nullptr;
              dcall = (returnIdx < 0) ? augmentcall : BuilderZ.CreateExtractValue(augmentcall, {(unsigned)returnIdx});
              gutils->originalToNewFn[orig] = dcall;
              assert(dcall->getType() == orig->getType());
              assert(dcall);

              if (!gutils->isConstantValue(orig)) {
                gutils->originalToNewFn[orig] = dcall;
                if (!orig->getType()->isFPOrFPVectorTy() && TR.query(orig).Data0()[{}].isPossiblePointer()) {
                } else if (mode != DerivativeMode::Forward) {
                  ((DiffeGradientUtils*)gutils)->differentials[dcall] = ((DiffeGradientUtils*)gutils)->differentials[op];
                  ((DiffeGradientUtils*)gutils)->differentials.erase(op);
                }
              }
              assert(dcall->getType() == orig->getType());
              gutils->replaceAWithB(op, dcall);

              auto name = op->getName().str();
              op->setName("");
              if (isa<Instruction>(dcall) && !isa<PHINode>(dcall)) {
                cast<Instruction>(dcall)->setName(name);
              }

              if (mode == DerivativeMode::Forward && is_value_needed_in_reverse(TR, gutils, orig, /*topLevel*/mode == DerivativeMode::Both)) {
                gutils->addMalloc(BuilderZ, dcall, getIndex(orig, CacheType::Self) );
              } 

              BuilderZ.SetInsertPoint(op->getNextNode());
              gutils->erase(op);
            } else {
              BuilderZ.SetInsertPoint(BuilderZ.GetInsertPoint()->getNextNode());
              eraseIfUnused(*orig, /*erase*/true, /*check*/false);
              gutils->originalToNewFn[orig] = augmentcall; 
            }

          } else {
            if (subdata && subdata->returns.find(AugmentedStruct::Tape) == subdata->returns.end()) {
            } else {
              //assert(!tape);
              //assert(subdata);
              if (!tape) {
                assert(tapeIdx != 0xDEADBEEF);
                tape = BuilderZ.CreatePHI( (tapeIdx == -1) ? FT->getReturnType() : cast<StructType>(FT->getReturnType())->getElementType(tapeIdx), 1, "tapeArg" );
              }
              tape = gutils->addMalloc(BuilderZ, tape, getIndex(orig, CacheType::Tape) );
            }

            if (subretused) {
              if (is_value_needed_in_reverse(TR, gutils, orig, mode == DerivativeMode::Both)) {
                cachereplace = BuilderZ.CreatePHI(orig->getType(), 1, orig->getName()+"_tmpcacheB");
                cachereplace = gutils->addMalloc(BuilderZ, cachereplace, getIndex(orig, CacheType::Self) );
              } else {
                auto pn = BuilderZ.CreatePHI(orig->getType(), 1, (orig->getName()+"_replacementE").str());
                gutils->fictiousPHIs.push_back(pn);
                cachereplace = pn;
              }
            } else {
              // TODO move right after op for the insertion point of BuilderZ

              BuilderZ.SetInsertPoint(BuilderZ.GetInsertPoint()->getNextNode());
              eraseIfUnused(*orig, /*erase*/true, /*check*/false);
            }
          }

          if( gutils->invertedPointers.count(orig) ) {

              auto placeholder = cast<PHINode>(gutils->invertedPointers[orig]);
              llvm::errs() << " +  considering placeholder: " << *placeholder << " of " << *orig << "\n";

              bool subcheck = (subretType == DIFFE_TYPE::DUP_ARG || subretType == DIFFE_TYPE::DUP_NONEED);

              //! We only need the shadow pointer for non-forward mode if it is used in a non return setting
              bool hasNonReturnUse = false;
              for(auto use : orig->users()) {
                  if (mode == DerivativeMode::Forward || !isa<ReturnInst>(use)) { // || returnuses.find(cast<Instruction>(use)) == returnuses.end()) {
                      hasNonReturnUse = true;
                  }
              }

              if( subcheck && hasNonReturnUse) {

                Value* newip = nullptr;
                  if (mode == DerivativeMode::Both || mode  == DerivativeMode::Forward) {
                    newip = (differetIdx < 0) ? augmentcall : BuilderZ.CreateExtractValue(augmentcall, {(unsigned)differetIdx}, orig->getName()+"'ac");
                    assert(newip->getType() == orig->getType());
                    placeholder->replaceAllUsesWith(newip);
                    gutils->erase(placeholder);
                  } else {
                    newip = placeholder;
                  }

                  newip = gutils->addMalloc(BuilderZ, newip, getIndex(orig, CacheType::Shadow) );

                  gutils->invertedPointers[orig] = newip;
              } else {
                  gutils->invertedPointers.erase(orig);
                  gutils->erase(placeholder);
              }
          }

          if (fnandtapetype && fnandtapetype->tapeType && mode != DerivativeMode::Forward) {
            auto tapep = BuilderZ.CreatePointerCast(tape, PointerType::getUnqual(fnandtapetype->tapeType));
            auto truetape = BuilderZ.CreateLoad(tapep);
            truetape->setMetadata("enzyme_mustcache", MDNode::get(truetape->getContext(), {}));

            CallInst* ci = cast<CallInst>(CallInst::CreateFree(tape, &*BuilderZ.GetInsertPoint()));
            ci->addAttribute(AttributeList::FirstArgIndex, Attribute::NonNull);
            tape = truetape;
          }
    } else {
      if( gutils->invertedPointers.count(orig) ) {
        auto placeholder = cast<PHINode>(gutils->invertedPointers[orig]);
        gutils->invertedPointers.erase(orig);
        gutils->erase(placeholder);
      }
      if (/*!topLevel*/mode != DerivativeMode::Both && subretused && !orig->doesNotAccessMemory()) {
        if (is_value_needed_in_reverse(TR, gutils, orig, mode == DerivativeMode::Both)) {
          assert(!replaceFunction);
          cachereplace = BuilderZ.CreatePHI(orig->getType(), 1, orig->getName()+"_cachereplace2");
          cachereplace = gutils->addMalloc(BuilderZ, cachereplace, getIndex(orig, CacheType::Self) );
        } else {
          auto pn = BuilderZ.CreatePHI(orig->getType(), 1, (orig->getName()+"_replacementC").str());
          gutils->fictiousPHIs.push_back(pn);
          cachereplace = pn; //UndefValue::get(op->getType());
          //cachereplace = UndefValue::get(op->getType());
        }
      }

      if (!subretused && !replaceFunction)
        eraseIfUnused(*orig, /*erase*/true, /*check*/false);
    }

    // Note here down only contains the reverse bits
    if (mode == DerivativeMode::Forward) {
      return;
    }

    IRBuilder<> Builder2 = getReverseBuilder(call.getParent());

    bool retUsed = replaceFunction && subretused;
    Value* newcalled = nullptr;

    bool subdretptr = (subretType == DIFFE_TYPE::DUP_ARG || subretType == DIFFE_TYPE::DUP_NONEED) && replaceFunction;
    bool subtopLevel = replaceFunction || !modifyPrimal;
    if (called) {
      newcalled = CreatePrimalAndGradient(cast<Function>(called), subretType, argsInverted, gutils->TLI, TR.analysis, gutils->AA, /*returnValue*/retUsed, /*subdretptr*/subdretptr, /*topLevel*/subtopLevel, tape ? tape->getType() : nullptr, nextTypeInfo, uncacheable_args, subdata);//, LI, DT);
    } else {

      assert(!subtopLevel);

      newcalled = gutils->invertPointerM(orig->getCalledValue(), Builder2);

      auto ft = cast<FunctionType>(cast<PointerType>(orig->getCalledValue()->getType())->getElementType());

      DIFFE_TYPE subretType = orig->getType()->isFPOrFPVectorTy() ? DIFFE_TYPE::OUT_DIFF : DIFFE_TYPE::DUP_ARG;
      if (orig->getType()->isVoidTy() || orig->getType()->isEmptyTy()) subretType = DIFFE_TYPE::CONSTANT;
      auto res = getDefaultFunctionTypeForGradient(ft, /*subretType*/subretType);
      //TODO Note there is empty tape added here, replace with generic
      res.first.push_back(Type::getInt8PtrTy(newcalled->getContext()));
      auto fptype = PointerType::getUnqual(FunctionType::get(StructType::get(newcalled->getContext(), res.second), res.first, ft->isVarArg()));
      newcalled = Builder2.CreatePointerCast(newcalled, PointerType::getUnqual(fptype));
      newcalled = Builder2.CreateLoad(Builder2.CreateConstGEP1_64(newcalled, 1));
    }

    if (subretType == DIFFE_TYPE::OUT_DIFF) {
      args.push_back(diffe(orig, Builder2));
    }

    if (tape) {
      auto ntape = gutils->lookupM(tape, Builder2);
      assert(ntape);
      assert(ntape->getType());
      args.push_back(ntape);
    }

    assert(newcalled);
    //if (auto NC = dyn_cast<Function>(newcalled)) {
    FunctionType* FT = cast<FunctionType>(cast<PointerType>(newcalled->getType())->getElementType());

    if (false) {
  badfn:;
          auto NC = dyn_cast<Function>(newcalled);
          llvm::errs() << *gutils->oldFunc << "\n";
          llvm::errs() << *gutils->newFunc << "\n";
          if (NC)
          llvm::errs() << " trying to call " << NC->getName() << " " << *FT << "\n";
          else
          llvm::errs() << " trying to call " << *newcalled << " " << *FT << "\n";

          for(unsigned i=0; i<args.size(); i++) {
              assert(args[i]);
              assert(args[i]->getType());
              llvm::errs() << "args[" << i << "] = " << *args[i] << " FT:" << *FT->getParamType(i) << "\n";
          }
          assert(0 && "calling with wrong number of arguments");
          exit(1);

    }

    if (args.size() != FT->getNumParams()) goto badfn;

    for(unsigned i=0; i<args.size(); i++) {
      if (args[i]->getType() != FT->getParamType(i)) goto badfn;
    }

    CallInst* diffes = Builder2.CreateCall(newcalled, args);
    diffes->setCallingConv(orig->getCallingConv());
    diffes->setDebugLoc(orig->getDebugLoc());

    unsigned structidx = retUsed ? 1 : 0;
    if (subdretptr) structidx++;

    for(unsigned i=0;i<orig->getNumArgOperands(); i++) {
      if (argsInverted[i] == DIFFE_TYPE::OUT_DIFF) {
        Value* diffeadd = Builder2.CreateExtractValue(diffes, {structidx});
        structidx++;
        addToDiffe(orig->getArgOperand(i), diffeadd, Builder2, TR.intType(orig->getArgOperand(i), false).isFloat());
      }
    }

    if (diffes->getType()->isVoidTy()) {
      assert(structidx == 0);
    } else {
      assert(cast<StructType>(diffes->getType())->getNumElements() == structidx);
    }

    if (subretType == DIFFE_TYPE::OUT_DIFF)
      setDiffe(orig, Constant::getNullValue(orig->getType()), Builder2);

    if (replaceFunction) {

      //if a function is replaced for joint forward/reverse, handle inverted pointers
      if (gutils->invertedPointers.count(orig)) {
          auto placeholder = cast<PHINode>(gutils->invertedPointers[orig]);
          gutils->invertedPointers.erase(orig);
          if (subdretptr) {
              dumpMap(gutils->invertedPointers);
              auto dretval = cast<Instruction>(Builder2.CreateExtractValue(diffes, {1}));
              /* todo handle this case later */
              assert(!subretused);
              gutils->invertedPointers[orig] = dretval;
          }
          gutils->erase(placeholder);
      }

      Instruction* retval = nullptr;

      CallInst* const op = cast<CallInst>(gutils->getNewFromOriginal(&call));

      ValueToValueMapTy mapp;
      if (subretused) {
        retval = cast<Instruction>(Builder2.CreateExtractValue(diffes, {0}));

        
        if (gutils->scopeMap.find(op) != gutils->scopeMap.end()) {
          AllocaInst* cache = cast<AllocaInst>(gutils->scopeMap[op]);
          for(auto st : gutils->scopeStores[cache])
            cast<StoreInst>(st)->eraseFromParent();
          gutils->scopeStores.clear();
          gutils->storeInstructionInCache(op->getParent(), retval, cache);
        }
        op->replaceAllUsesWith(retval);
        mapp[op] = retval;
      } else {
        eraseIfUnused(*orig, /*erase*/false, /*check*/false);
      }

      for (auto &a : *gutils->reverseBlocks[cast<BasicBlock>(gutils->getNewFromOriginal(orig->getParent()))]) {
        mapp[&a] = &a;
      }

      std::reverse(postCreate.begin(), postCreate.end());
      for(auto a : postCreate) {

        // If is the store to return handle manually since no original inst for
        bool fromStore = false;
        for(auto& pair : *replacedReturns) {
          if (pair.second == a) {
            for(unsigned i=0; i<a->getNumOperands(); i++) {
              a->setOperand(i, gutils->unwrapM(a->getOperand(i), Builder2, mapp, UnwrapMode::LegalFullUnwrap));
            }
            a->moveBefore(*Builder2.GetInsertBlock(), Builder2.GetInsertPoint());
            fromStore = true;
            break;
          }
        }
        if (fromStore) continue;

        auto orig_a = gutils->isOriginal(a);
        if (orig_a) {
          for(unsigned i=0; i<a->getNumOperands(); i++) {
            a->setOperand(i, gutils->unwrapM(gutils->getNewFromOriginal(orig_a->getOperand(i)), Builder2, mapp, UnwrapMode::LegalFullUnwrap));
          }
        }
        a->moveBefore(*Builder2.GetInsertBlock(), Builder2.GetInsertPoint());
        mapp[a] = a;
      }

      gutils->originalToNewFn[orig] = retval ? retval : diffes;


      //llvm::errs() << "newFunc postrep: " << *gutils->newFunc << "\n";

      erased.insert(orig);
      gutils->erase(op);

      return;
    }

    if (cachereplace) {
      if (subretused) {
        Value* dcall = nullptr;
        assert(cachereplace->getType() == orig->getType());
        assert(dcall == nullptr);
        dcall = cachereplace;
        assert(dcall);

        CallInst* const op = cast<CallInst>(gutils->getNewFromOriginal(&call));

        if (!gutils->isConstantValue(orig)) {
          gutils->originalToNewFn[orig] = dcall;
          if (!orig->getType()->isFPOrFPVectorTy() && TR.query(orig).Data0()[{}].isPossiblePointer()) {
          } else {
            ((DiffeGradientUtils*)gutils)->differentials[dcall] = ((DiffeGradientUtils*)gutils)->differentials[op];
            ((DiffeGradientUtils*)gutils)->differentials.erase(op);
          }
        }
        assert(dcall->getType() == orig->getType());
        op->replaceAllUsesWith(dcall);
        auto name = orig->getName();
        op->setName("");
        if (isa<Instruction>(dcall) && !isa<PHINode>(dcall)) {
          cast<Instruction>(dcall)->setName(name);
        }
        gutils->erase(op);
      } else {
        eraseIfUnused(*orig, /*erase*/true, /*check*/false);
        if (augmentcall) {
          gutils->originalToNewFn[orig] = augmentcall;
        }
      }

    }
    return;
  }
};


//! return structtype if recursive function
const AugmentedReturn& CreateAugmentedPrimal(Function* todiff, DIFFE_TYPE retType, const std::vector<DIFFE_TYPE>& constant_args, TargetLibraryInfo &TLI, TypeAnalysis &TA, AAResults &global_AA,
                                             bool returnUsed, const NewFnTypeInfo& oldTypeInfo_,
                                             const std::map<Argument*, bool> _uncacheable_args, bool forceAnonymousTape) {
  if (returnUsed) assert(!todiff->getReturnType()->isEmptyTy() && !todiff->getReturnType()->isVoidTy());
  if (retType != DIFFE_TYPE::CONSTANT) assert(!todiff->getReturnType()->isEmptyTy() && !todiff->getReturnType()->isVoidTy());

  NewFnTypeInfo oldTypeInfo = oldTypeInfo_;
  for(auto &pair : oldTypeInfo.knownValues) {
    if (pair.second.size() != 0) {
      bool recursiveUse = false;
      for(auto user : pair.first->users()) {
        if (auto bi = dyn_cast<BinaryOperator>(user)) {
          for( auto biuser : bi->users()) {
            if (auto ci = dyn_cast<CallInst>(biuser)) {
              if (ci->getCalledFunction() == todiff && ci->getArgOperand(pair.first->getArgNo()) == bi) {
                recursiveUse = true;
                break;
              }
            }
          }
        }
        if (recursiveUse) break;
      }
      if (recursiveUse) pair.second.clear();
    }
  }

  static std::map<std::tuple<Function*,DIFFE_TYPE/*retType*/,std::vector<DIFFE_TYPE>/*constant_args*/, std::map<Argument*, bool>/*uncacheable_args*/, bool/*returnUsed*/, const NewFnTypeInfo>, AugmentedReturn> cachedfunctions;
  static std::map<std::tuple<Function*,DIFFE_TYPE/*retType*/,std::vector<DIFFE_TYPE>/*constant_args*/, std::map<Argument*, bool>/*uncacheable_args*/, bool/*returnUsed*/, const NewFnTypeInfo>, bool> cachedfinished;
  std::tuple<Function*,DIFFE_TYPE/*retType*/,std::vector<DIFFE_TYPE>/*constant_args*/, std::map<Argument*, bool>/*uncacheable_args*/, bool/*returnUsed*/, const NewFnTypeInfo> tup = std::make_tuple(todiff, retType, constant_args, std::map<Argument*, bool>(_uncacheable_args.begin(), _uncacheable_args.end()), returnUsed, oldTypeInfo);
  auto found = cachedfunctions.find(tup);
  //llvm::errs() << "augmenting function " << todiff->getName() << " constant args " << to_string(constant_args) << " uncacheable_args: " << to_string(_uncacheable_args) << " retType:" << retType << " returnUsed: " << returnUsed << " found==" << (found != cachedfunctions.end()) << "\n";
  if (found != cachedfunctions.end()) {
    return found->second;
  }

  // TODO make default typing (not just constant)
  bool hasconstant = false;
  for(auto v : constant_args) {
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
          report_fatal_error("unknown augment for noninvertible function -- metadata incorrect");
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
        FunctionType *FTy = FunctionType::get(StructType::get(todiff->getContext(), {StructType::get(todiff->getContext(), {}), foundcalled->getReturnType()}),
                                   foundcalled->getFunctionType()->params(), foundcalled->getFunctionType()->isVarArg());
        Function *NewF = Function::Create(FTy, Function::LinkageTypes::InternalLinkage, "fixaugmented_"+todiff->getName(), todiff->getParent());
        NewF->setAttributes(foundcalled->getAttributes());
        if (NewF->hasFnAttribute(Attribute::NoInline)) {
            NewF->removeFnAttr(Attribute::NoInline);
        }
        for (auto i=foundcalled->arg_begin(), j=NewF->arg_begin(); i != foundcalled->arg_end(); ) {
             j->setName(i->getName());
             if (j->hasAttribute(Attribute::Returned))
                 j->removeAttr(Attribute::Returned);
             if (j->hasAttribute(Attribute::StructRet))
                 j->removeAttr(Attribute::StructRet);
             i++;
             j++;
         }
        BasicBlock *BB = BasicBlock::Create(NewF->getContext(), "entry", NewF);
        IRBuilder <>bb(BB);
        SmallVector<Value*,4> args;
        for(auto &a : NewF->args()) args.push_back(&a);
        auto cal = bb.CreateCall(foundcalled, args);
        cal->setCallingConv(foundcalled->getCallingConv());
        auto ut = UndefValue::get(NewF->getReturnType());
        auto val = bb.CreateInsertValue(ut, cal, {1u});
        bb.CreateRet(val);
        return insert_or_assign(cachedfunctions, tup, AugmentedReturn(NewF, nullptr, {}, returnMapping, {}, {}))->second;
      }

      //assert(st->getNumElements() > 0);
      return insert_or_assign(cachedfunctions, tup, AugmentedReturn(foundcalled, nullptr, {}, returnMapping, {}, {}))->second; //dyn_cast<StructType>(st->getElementType(0)));
    }

  if (todiff->empty()) {
    llvm::errs() << "mod: " << *todiff->getParent() << "\n";
    llvm::errs() << *todiff << "\n";
  }
  assert(!todiff->empty());
  std::map<AugmentedStruct, int> returnMapping;
  AAResults AA(TLI);
  //AA.addAAResult(global_AA);

  GradientUtils *gutils = GradientUtils::CreateFromClone(todiff, TLI, TA, AA, retType, constant_args, /*returnUsed*/returnUsed, returnMapping);
  const SmallPtrSet<BasicBlock*, 4> guaranteedUnreachable = getGuaranteedUnreachable(gutils->oldFunc);

  gutils->forceContexts();

  NewFnTypeInfo typeInfo(gutils->oldFunc);
  {
    auto toarg = todiff->arg_begin();
    auto olarg = gutils->oldFunc->arg_begin();
    for(; toarg != todiff->arg_end(); toarg++, olarg++) {

        {
        auto fd = oldTypeInfo.first.find(toarg);
        assert(fd != oldTypeInfo.first.end());
        typeInfo.first.insert(std::pair<Argument*, ValueData>(olarg, fd->second));
        }

        {
        auto cfd = oldTypeInfo.knownValues.find(toarg);
        assert(cfd != oldTypeInfo.knownValues.end());
        typeInfo.knownValues.insert(std::pair<Argument*, std::set<int64_t>>(olarg, cfd->second));
        }
    }
    typeInfo.second = oldTypeInfo.second;
  }
  TypeResults TR = TA.analyzeFunction(typeInfo);
  assert(TR.info.function == gutils->oldFunc);
  gutils->forceActiveDetection(AA, TR);


  gutils->forceAugmentedReturns(TR, guaranteedUnreachable);

  // Convert uncacheable args from the input function to the preprocessed function
  std::map<Argument*, bool> _uncacheable_argsPP;
  {
    auto in_arg = todiff->arg_begin();
    auto pp_arg = gutils->oldFunc->arg_begin();
    for(; pp_arg != gutils->oldFunc->arg_end(); ) {
        _uncacheable_argsPP[pp_arg] = _uncacheable_args.find(in_arg)->second;
        pp_arg++;
        in_arg++;
    }
  }

  SmallPtrSet<const Value*,4> unnecessaryValues;
  SmallPtrSet<const Instruction*, 4> unnecessaryInstructions;
  calculateUnusedValues(*gutils->oldFunc, unnecessaryValues, unnecessaryInstructions, returnUsed, [&](const Value* val) {
    return is_value_needed_in_reverse(TR, gutils, val, /*topLevel*/false);
  }, [&](const Instruction* inst) {
    if (auto II = dyn_cast<IntrinsicInst>(inst)) {
      if (II->getIntrinsicID() == Intrinsic::lifetime_start || II->getIntrinsicID() == Intrinsic::lifetime_end ||
          II->getIntrinsicID() == Intrinsic::stacksave || II->getIntrinsicID() == Intrinsic::stackrestore) {
        return false;
      }
    }

    if (auto obj_op = dyn_cast<CallInst>(inst)) {
      Function* called = obj_op->getCalledFunction();
      if (auto castinst = dyn_cast<ConstantExpr>(obj_op->getCalledValue())) {
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
      if (isa<UndefValue>(si->getValueOperand())) return false;
      auto at = GetUnderlyingObject(si->getPointerOperand(), gutils->oldFunc->getParent()->getDataLayout(), 100);
      if (auto arg = dyn_cast<Argument>(at)) {
        if (constant_args[arg->getArgNo()] == DIFFE_TYPE::DUP_NONEED) {
          return false;
        }
      }
    }

    if (auto mti = dyn_cast<MemTransferInst>(inst)) {
      auto at = GetUnderlyingObject(mti->getArgOperand(1), gutils->oldFunc->getParent()->getDataLayout(), 100);
      if (auto arg = dyn_cast<Argument>(at)) {
        if (constant_args[arg->getArgNo()] == DIFFE_TYPE::DUP_NONEED) {
          return false;
        }
      }
      if (auto ai = dyn_cast<AllocaInst>(at)) {
          bool foundStore = false;
          allInstructionsBetween(gutils->OrigLI, ai, const_cast<MemTransferInst*>(mti), [&](Instruction* I) {
            if (!I->mayWriteToMemory()) return /*earlyBreak*/false;
            if (unnecessaryInstructions.count(I)) return /*earlyBreak*/false;

            //if (I == &MTI) return;
            if (writesToMemoryReadBy(gutils->AA, /*maybeReader*/const_cast<MemTransferInst*>(mti), /*maybeWriter*/I)) {
              foundStore = true;
              return /*earlyBreak*/true;
            }
            return /*earlyBreak*/false;
          });
          if (!foundStore) {
            return false;
          }
      }
    }

    return inst->mayWriteToMemory() || is_value_needed_in_reverse(TR, gutils, inst, /*topLevel*/false);
  });

  SmallPtrSet<const Instruction*, 4> unnecessaryStores;
  calculateUnusedStores(*gutils->oldFunc, unnecessaryStores, [&](const Instruction* inst) {

    if (auto si = dyn_cast<StoreInst>(inst)) {
      if (isa<UndefValue>(si->getValueOperand())) return false;
      auto at = GetUnderlyingObject(si->getPointerOperand(), gutils->oldFunc->getParent()->getDataLayout(), 100);
      if (auto arg = dyn_cast<Argument>(at)) {
        if (constant_args[arg->getArgNo()] == DIFFE_TYPE::DUP_NONEED) {
          return false;
        }
      }
    }

    if (auto mti = dyn_cast<MemTransferInst>(inst)) {
      auto at = GetUnderlyingObject(mti->getArgOperand(1), gutils->oldFunc->getParent()->getDataLayout(), 100);
      if (auto arg = dyn_cast<Argument>(at)) {
        if (constant_args[arg->getArgNo()] == DIFFE_TYPE::DUP_NONEED) {
          return false;
        }
      }
      if (auto ai = dyn_cast<AllocaInst>(at)) {
          bool foundStore = false;
          allInstructionsBetween(gutils->OrigLI, ai, const_cast<MemTransferInst*>(mti), [&](Instruction* I) {
            if (!I->mayWriteToMemory()) return /*earlyBreak*/false;
            if (unnecessaryInstructions.count(I)) return /*earlyBreak*/false;

            //if (I == &MTI) return;
            if (writesToMemoryReadBy(gutils->AA, /*maybeReader*/const_cast<MemTransferInst*>(mti), /*maybeWriter*/I)) {
              foundStore = true;
              return /*earlyBreak*/true;
            }
            return /*earlyBreak*/false;
          });
          if (!foundStore) {
            return false;
          }
      }
    }

    return true;
  });

  const std::map<CallInst*, const std::map<Argument*, bool> > uncacheable_args_map =
      compute_uncacheable_args_for_callsites(gutils->oldFunc, gutils->DT, TLI, unnecessaryInstructions, AA, gutils, _uncacheable_argsPP);

  const std::map<Instruction*, bool> can_modref_map = compute_uncacheable_load_map(gutils, AA, TLI, unnecessaryInstructions, _uncacheable_argsPP);

  //for (auto &iter : can_modref_map) {
  //  llvm::errs() << "isneeded: " << iter.second << " augmented can_modref_map: " << *iter.first << " fn: " << iter.first->getParent()->getParent()->getName() << "\n";
  //}

  insert_or_assign(cachedfunctions, tup, AugmentedReturn(gutils->newFunc, nullptr, {}, returnMapping, uncacheable_args_map, can_modref_map));
  cachedfinished[tup] = false;

  auto getIndex = [&](Instruction* I, CacheType u)-> unsigned {
    //std::map<std::pair<Instruction*,std::string>,unsigned>& mapping = cachedfunctions[tup].tapeIndices;
    return gutils->getIndex( std::make_pair(I, u), cachedfunctions.find(tup)->second.tapeIndices);
  };
  gutils->can_modref_map = &can_modref_map;

  //! Explicitly handle all returns first to ensure that all instructions know whether or not they are used
  SmallPtrSet<Instruction*, 4> returnuses;

  for(BasicBlock& BB: *gutils->oldFunc) {
    if(auto orig_ri = dyn_cast<ReturnInst>(BB.getTerminator())) {
      auto ri = gutils->getNewFromOriginal(orig_ri);
        Value* orig_oldval = orig_ri->getReturnValue();
        Value* oldval = orig_oldval ? gutils->getNewFromOriginal(orig_oldval) : nullptr;
        IRBuilder <>ib(ri);
        Value* rt = UndefValue::get(gutils->newFunc->getReturnType());
        if (oldval && returnUsed) {
            assert(returnMapping.find(AugmentedStruct::Return) != returnMapping.end());
            //llvm::errs() << " rt: " << *rt << " oldval:" << *oldval << "\n";
            //llvm::errs() << "    returnIndex: " << returnMapping.find(AugmentedStruct::Return)->second << "\n";
            auto idx = returnMapping.find(AugmentedStruct::Return)->second;
            if (idx < 0)
              rt = oldval;
            else
              rt = ib.CreateInsertValue(rt, oldval, {(unsigned)idx});
            if (Instruction* inst = dyn_cast<Instruction>(rt)) {
                returnuses.insert(inst);
            }
        }

        auto newri = ib.CreateRet(rt);
        gutils->originalToNewFn[orig_ri] = newri;
        gutils->erase(ri);
    }
  }

  DerivativeMaker<AugmentedReturn*> maker(DerivativeMode::Forward, gutils, constant_args, TR, getIndex, uncacheable_args_map, &returnuses, &cachedfunctions.find(tup)->second, nullptr, nullptr, unnecessaryValues, unnecessaryInstructions, unnecessaryStores, nullptr);

  for(BasicBlock& oBB: *gutils->oldFunc) {
      auto term = oBB.getTerminator();
      assert(term);

      // Don't create derivatives for code that results in termination
      if (guaranteedUnreachable.find(&oBB) != guaranteedUnreachable.end()) {
        std::vector<Instruction*> toerase;

        // For having the prints still exist on bugs, check if indeed unused
        for(auto &I: oBB) { toerase.push_back(&I); }
        for(auto I : toerase) { maker.eraseIfUnused(*I, /*erase*/true, /*check*/true); }
        auto newBB = cast<BasicBlock>(gutils->getNewFromOriginal(&oBB));
        if (!newBB->getTerminator()) {
          for(auto next : successors(&oBB)) {
            auto sucBB = cast<BasicBlock>(gutils->getNewFromOriginal(next));
            sucBB->removePredecessor(newBB);
          }
          IRBuilder<> builder(newBB);
          builder.CreateUnreachable();
        }
        continue;
      }

      if (!isa<ReturnInst>(term) && !isa<BranchInst>(term) && !isa<SwitchInst>(term)) {
        llvm::errs() << *oBB.getParent() << "\n";
        llvm::errs() << "unknown terminator instance " << *term << "\n";
        assert(0 && "unknown terminator inst");
      }

      BasicBlock::reverse_iterator I = oBB.rbegin(), E = oBB.rend();
      I++;
      for (; I != E; I++) {
        maker.visit(&*I);
        assert(oBB.rend() == E);
      }
  }

  auto nf = gutils->newFunc;

  while(gutils->inversionAllocs->size() > 0) {
    gutils->inversionAllocs->back().moveBefore(gutils->newFunc->getEntryBlock().getFirstNonPHIOrDbgOrLifetime());
  }

  (IRBuilder <>(gutils->inversionAllocs)).CreateUnreachable();
  DeleteDeadBlock(gutils->inversionAllocs);

  for (Argument &Arg : gutils->newFunc->args()) {
      if (Arg.hasAttribute(Attribute::Returned))
          Arg.removeAttr(Attribute::Returned);
      if (Arg.hasAttribute(Attribute::StructRet))
          Arg.removeAttr(Attribute::StructRet);
  }

  if (gutils->newFunc->hasFnAttribute(Attribute::OptimizeNone))
    gutils->newFunc->removeFnAttr(Attribute::OptimizeNone);

  if (auto bytes = gutils->newFunc->getDereferenceableBytes(llvm::AttributeList::ReturnIndex)) {
    AttrBuilder ab;
    ab.addDereferenceableAttr(bytes);
    gutils->newFunc->removeAttributes(llvm::AttributeList::ReturnIndex, ab);
  }

  if (gutils->newFunc->hasAttribute(llvm::AttributeList::ReturnIndex, llvm::Attribute::NoAlias)) {
    gutils->newFunc->removeAttribute(llvm::AttributeList::ReturnIndex, llvm::Attribute::NoAlias);
  }
  if (gutils->newFunc->hasAttribute(llvm::AttributeList::ReturnIndex, llvm::Attribute::ZExt)) {
    gutils->newFunc->removeAttribute(llvm::AttributeList::ReturnIndex, llvm::Attribute::ZExt);
  }

  gutils->cleanupActiveDetection();
  gutils->eraseFictiousPHIs();


  if (llvm::verifyFunction(*gutils->newFunc, &llvm::errs())) {
      llvm::errs() << *gutils->oldFunc << "\n";
      llvm::errs() << *gutils->newFunc << "\n";
      report_fatal_error("function failed verification (2)");
  }

  std::vector<Type*> RetTypes(cast<StructType>(gutils->newFunc->getReturnType())->elements());

  std::vector<Type*> MallocTypes;

  for(auto a:gutils->getMallocs()) {
      MallocTypes.push_back(a->getType());
  }

  Type* tapeType = StructType::get(nf->getContext(), MallocTypes);

  bool removeTapeStruct = MallocTypes.size() == 1;
  if (removeTapeStruct) {
    tapeType = MallocTypes[0];

    for(auto &a : cachedfunctions.find(tup)->second.tapeIndices) {
      a.second = -1;
    }
  }

  bool recursive = cachedfunctions.find(tup)->second.fn->getNumUses() > 0 || forceAnonymousTape;
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
      cachedfunctions.find(tup)->second.returns[AugmentedStruct::Return] -= ( returnMapping[AugmentedStruct::Return] > tidx ) ? 1 : 0;
      returnMapping[AugmentedStruct::Return] -= ( returnMapping[AugmentedStruct::Return] > tidx ) ? 1 : 0;
    }
    if (returnMapping.find(AugmentedStruct::DifferentialReturn) != returnMapping.end()) {
      cachedfunctions.find(tup)->second.returns[AugmentedStruct::DifferentialReturn] -= ( returnMapping[AugmentedStruct::DifferentialReturn] > tidx ) ? 1 : 0;
      returnMapping[AugmentedStruct::DifferentialReturn] -= ( returnMapping[AugmentedStruct::DifferentialReturn] > tidx ) ? 1 : 0;
    }
    RetTypes.erase(RetTypes.begin() + tidx);
  } else if (recursive) {
    assert(RetTypes[returnMapping.find(AugmentedStruct::Tape)->second] == Type::getInt8PtrTy(nf->getContext()));
  } else {
    RetTypes[returnMapping.find(AugmentedStruct::Tape)->second] = tapeType;
  }



  bool noReturn = RetTypes.size() == 0;
  Type* RetType = StructType::get(nf->getContext(), RetTypes);
  if (noReturn) RetType = Type::getVoidTy(RetType->getContext());
  if (noReturn) assert(noTape);

  bool removeStruct = RetTypes.size() == 1;

  if (removeStruct) {
    RetType = RetTypes[0];
    for(auto & a : returnMapping) {
      a.second = -1;
    }
    for(auto &a : cachedfunctions.find(tup)->second.returns) {
      a.second = -1;
    }
  }

 ValueToValueMapTy VMap;
 std::vector<Type*> ArgTypes;
 for (const Argument &I : nf->args()) {
     ArgTypes.push_back(I.getType());
 }

 // Create a new function type...
 FunctionType *FTy = FunctionType::get(RetType,
                                   ArgTypes, nf->getFunctionType()->isVarArg());

 // Create the new function...
 Function *NewF = Function::Create(FTy, nf->getLinkage(), "augmented_"+todiff->getName(), nf->getParent());

  unsigned ii = 0, jj = 0;
  for (auto i=nf->arg_begin(), j=NewF->arg_begin(); i != nf->arg_end(); ) {
    VMap[i] = j;
    if (nf->hasParamAttribute(ii, Attribute::NoCapture)) {
      NewF->addParamAttr(jj, Attribute::NoCapture);
    }
    if (nf->hasParamAttribute(ii, Attribute::NoAlias)) {
       NewF->addParamAttr(jj, Attribute::NoAlias);
    }

    j->setName(i->getName());
    j++;
    jj++;

    i++;
    ii++;
  }

  //! Keep track of inverted pointers we may need to return
  ValueToValueMapTy invertedRetPs;
  if (retType == DIFFE_TYPE::DUP_ARG || retType == DIFFE_TYPE::DUP_NONEED) {
    for(BasicBlock& BB: *gutils->oldFunc) {
      if(auto ri = dyn_cast<ReturnInst>(BB.getTerminator())) {
        if (Value* orig_oldval = ri->getReturnValue()) {
          auto newri = gutils->getNewFromOriginal(ri);
          IRBuilder<> BuilderZ(newri);
          invertedRetPs[newri] = gutils->invertPointerM(orig_oldval, BuilderZ);
        }
      }
    }
  }

  SmallVector <ReturnInst*,4> Returns;
  CloneFunctionInto(NewF, nf, VMap, nf->getSubprogram() != nullptr, Returns, "",
                   nullptr);

  IRBuilder<> ib(NewF->getEntryBlock().getFirstNonPHI());

  Value* ret = noReturn ? nullptr : ib.CreateAlloca(RetType);

  if (!noTape) {
    Value* tapeMemory;
    if (recursive) {
        auto i64 = Type::getInt64Ty(NewF->getContext());
        tapeMemory = CallInst::CreateMalloc(NewF->getEntryBlock().getFirstNonPHI(),
                      i64,
                      tapeType,
                      ConstantInt::get(i64, NewF->getParent()->getDataLayout().getTypeAllocSizeInBits(tapeType)/8),
                      nullptr,
                      nullptr,
                      "tapemem"
                      );
      CallInst* malloccall = dyn_cast<CallInst>(tapeMemory);
      if (malloccall == nullptr) {
          malloccall = cast<CallInst>(cast<Instruction>(tapeMemory)->getOperand(0));
      }
      malloccall->addAttribute(AttributeList::ReturnIndex, Attribute::NoAlias);
      malloccall->addAttribute(AttributeList::ReturnIndex, Attribute::NonNull);
      std::vector<Value*> Idxs = {
          ib.getInt32(0),
          ib.getInt32(returnMapping.find(AugmentedStruct::Tape)->second),
      };
      assert(malloccall);
      assert(ret);
      Value* gep = ret;
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

    unsigned i=0;
    for (auto v: gutils->getMallocs()) {
        if (!isa<UndefValue>(v)) {
            //llvm::errs() << "v: " << *v << "VMap[v]: " << *VMap[v] << "\n";
            IRBuilder <>ib(cast<Instruction>(VMap[v])->getNextNode());
            std::vector<Value*> Idxs = {
              ib.getInt32(0),
              ib.getInt32(i)
            };
            Value* gep = tapeMemory;
            if (!removeTapeStruct) {
              gep = ib.CreateGEP(tapeMemory, Idxs, "");
              cast<GetElementPtrInst>(gep)->setIsInBounds(true);
            }
            ib.CreateStore(VMap[v], gep);
        }
        i++;
    }
  }
  
  for(BasicBlock& BB: *nf) {
    if(auto ri = dyn_cast<ReturnInst>(BB.getTerminator())) {
      ReturnInst* rim = cast<ReturnInst>(VMap[ri]);
      IRBuilder <>ib(rim);
      if (returnUsed) {
        Value* rv = rim->getReturnValue();
        assert(rv);
        Value* actualrv = nullptr;
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
        Value* gep = removeStruct ? ret : ib.CreateConstGEP2_32(RetType, ret, 0, returnMapping.find(AugmentedStruct::Return)->second, "");
        if (auto ggep = dyn_cast<GetElementPtrInst>(gep)) {
          ggep->setIsInBounds(true);
        }
        ib.CreateStore(actualrv, gep);
      }

      if (retType == DIFFE_TYPE::DUP_ARG || retType == DIFFE_TYPE::DUP_NONEED) {
          assert(invertedRetPs[ri]);
          if (!isa<UndefValue>(invertedRetPs[ri])) {
            assert(VMap[invertedRetPs[ri]]);
            Value* gep = removeStruct ? ret : ib.CreateConstGEP2_32(RetType, ret, 0, returnMapping.find(AugmentedStruct::DifferentialReturn)->second, "");
            if (auto ggep = dyn_cast<GetElementPtrInst>(gep)) {
              ggep->setIsInBounds(true);
            }
            ib.CreateStore( VMap[invertedRetPs[ri]], gep);
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

  if (auto bytes = NewF->getDereferenceableBytes(llvm::AttributeList::ReturnIndex)) {
    AttrBuilder ab;
    ab.addDereferenceableAttr(bytes);
    NewF->removeAttributes(llvm::AttributeList::ReturnIndex, ab);
  }
  if (NewF->hasAttribute(llvm::AttributeList::ReturnIndex, llvm::Attribute::NoAlias)) {
    NewF->removeAttribute(llvm::AttributeList::ReturnIndex, llvm::Attribute::NoAlias);
  }
  if (NewF->hasAttribute(llvm::AttributeList::ReturnIndex, llvm::Attribute::ZExt)) {
    NewF->removeAttribute(llvm::AttributeList::ReturnIndex, llvm::Attribute::ZExt);
  }

  if (llvm::verifyFunction(*NewF, &llvm::errs())) {
      llvm::errs() << *gutils->oldFunc << "\n";
      llvm::errs() << *NewF << "\n";
      report_fatal_error("augmented function failed verification (3)");
  }

  SmallVector<CallInst*,4> fnusers;
  for(auto user : cachedfunctions.find(tup)->second.fn->users()) {
    fnusers.push_back(cast<CallInst>(user));
  }
  for(auto user : fnusers) {
    if (removeStruct) {
      IRBuilder<> B(user);
      auto n = user->getName().str();
      user->setName("");
      std::vector<Value*> args(user->arg_begin(), user->arg_end());
      auto rep = B.CreateCall(NewF, args);
      rep->copyIRFlags(user);
      rep->setAttributes(user->getAttributes());
      rep->setCallingConv(user->getCallingConv());
      rep->setTailCallKind(user->getTailCallKind());
      rep->setDebugLoc(user->getDebugLoc());

      for(auto u : user->users()) {
        if (auto ei = dyn_cast<ExtractValueInst>(u)) {
          ei->replaceAllUsesWith(rep);
          ei->eraseFromParent();
        }
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

  //llvm::errs() << "augmented fn seeing sub_index_map of " << std::get<2>(cachedfunctions[tup]).size() << " in ap " << NewF->getName() << "\n";
  gutils->newFunc->eraseFromParent();

  delete gutils;
  if (enzyme_print)
    llvm::errs() << *NewF << "\n";
  return cachedfunctions.find(tup)->second;
}

void createInvertedTerminator(TypeResults& TR, DiffeGradientUtils* gutils, const std::vector<DIFFE_TYPE> &argTypes, BasicBlock *oBB, AllocaInst* retAlloca, AllocaInst* dretAlloca, unsigned extraArgs) {
    LoopContext loopContext;
    BasicBlock* BB = cast<BasicBlock>(gutils->getNewFromOriginal(oBB));
    bool inLoop = gutils->getContext(BB, loopContext);
    BasicBlock* BB2 = gutils->reverseBlocks[BB];
    assert(BB2);
    IRBuilder<> Builder(BB2);
    Builder.setFastMathFlags(getFast());

    std::map<BasicBlock*, std::vector<BasicBlock*>> targetToPreds;
    for(auto pred : predecessors(BB)) {
        targetToPreds[gutils->getReverseOrLatchMerge(pred, BB)].emplace_back(pred);
    }

    if (targetToPreds.size() == 0) {
        SmallVector<Value *,4> retargs;

        if (retAlloca) {
          auto result = Builder.CreateLoad(retAlloca, "retreload");
          //TODO reintroduce invariant load/group
          //result->setMetadata(LLVMContext::MD_invariant_load, MDNode::get(retAlloca->getContext(), {}));
          retargs.push_back(result);
        }

        if (dretAlloca) {
          auto result = Builder.CreateLoad(dretAlloca, "dretreload");
          //TODO reintroduce invariant load/group
          //result->setMetadata(LLVMContext::MD_invariant_load, MDNode::get(dretAlloca->getContext(), {}));
          retargs.push_back(result);
        }

        for (auto& I: gutils->oldFunc->args()) {
          if (!gutils->isConstantValue(&I) && argTypes[I.getArgNo()] == DIFFE_TYPE::OUT_DIFF ) {
            retargs.push_back(gutils->diffe(&I, Builder));
          }
        }

        if (gutils->newFunc->getReturnType()->isVoidTy()) {
          assert(retargs.size() == 0);
          Builder.CreateRetVoid();
          return;
        }

        Value* toret = UndefValue::get(gutils->newFunc->getReturnType());
        for(unsigned i=0; i<retargs.size(); i++) {
          unsigned idx[] = { i };
          toret = Builder.CreateInsertValue(toret, retargs[i], idx);
        }
        Builder.CreateRet(toret);
        return;
    }

    //PHINodes to replace that will contain true iff the predecessor was given basicblock
    std::map<BasicBlock*, PHINode*> replacePHIs;
    std::vector<SelectInst*> selects;

    IRBuilder <>phibuilder(BB2);
    bool setphi = false;

    // Ensure phi values have their derivatives propagated
    for (auto I = oBB->begin(), E = oBB->end(); I != E; I++) {
      if(PHINode* orig = dyn_cast<PHINode>(&*I)) {
        if (gutils->isConstantValue(orig)) continue;
        auto PNtype = TR.intType(orig, /*necessary*/false);

        //TODO remove explicit type check and only use PNtype
        if (PNtype == IntType::Pointer || orig->getType()->isPointerTy()) continue;

        auto prediff = gutils->diffe(orig, Builder);
        gutils->setDiffe(orig, Constant::getNullValue(orig->getType()), Builder);

        Type* PNfloatType = PNtype.isFloat();
        if (!PNfloatType)
          llvm::errs() << " for orig " << *orig << " saw " << TR.intType(orig, /*necessary*/false).str() << "\n";
        TR.intType(orig, /*necessary*/true);
        assert(PNfloatType);

        for (BasicBlock* opred : predecessors(oBB)) {
          auto oval = orig->getIncomingValueForBlock(opred);
          if (gutils->isConstantValue(oval)) {
            continue;
          }

          if (orig->getNumIncomingValues() == 1) {
            gutils->addToDiffe(oval, prediff, Builder, PNfloatType);
          } else {
            BasicBlock* pred = cast<BasicBlock>(gutils->getNewFromOriginal(opred));
            if (replacePHIs.find(pred) == replacePHIs.end()) {
              replacePHIs[pred] = Builder.CreatePHI(Type::getInt1Ty(pred->getContext()), 1, "replacePHI");
              if (!setphi) {
                phibuilder.SetInsertPoint(replacePHIs[pred]);
                setphi = true;
              }
            }
            SelectInst* dif = cast<SelectInst>(Builder.CreateSelect(replacePHIs[pred], prediff, Constant::getNullValue(prediff->getType())));
            //llvm::errs() << "creating prediff " << *dif << " for value incoming " << PN->getIncomingValueForBlock(pred) << " for " << *PN << "\n";
            auto addedSelects = gutils->addToDiffe(oval, dif, Builder, PNfloatType);
            /*if (dif->getNumUses() != 0) {
              llvm::errs() << "oldFunc: " << *gutils->oldFunc << "\n";
              llvm::errs() << "newFunc: " << *gutils->newFunc << "\n";
              for (auto use : dif->users()) {
                llvm::errs() << "user: " << *use << "\n";
              }
              llvm::errs() << "dif: " << *dif << "\n";
            }
            assert(dif->getNumUses() == 0);
            dif->eraseFromParent();
            */
            for (auto select : addedSelects)
              selects.emplace_back(select);
          }
        }
      } else break;
    }
    if (!setphi) {
        phibuilder.SetInsertPoint(Builder.GetInsertBlock(), Builder.GetInsertPoint());
    }

    if (inLoop && BB == loopContext.header) {
        std::map<BasicBlock*, std::vector<BasicBlock*>> targetToPreds;
        for(auto pred : predecessors(BB)) {
            if (pred == loopContext.preheader) continue;
            targetToPreds[gutils->getReverseOrLatchMerge(pred, BB)].emplace_back(pred);
        }

        assert(targetToPreds.size() && "only loops with one backedge are presently supported");

        Value* av = phibuilder.CreateLoad(loopContext.antivaralloc);
        Value* phi = phibuilder.CreateICmpEQ(av, Constant::getNullValue(av->getType()));
        Value* nphi = phibuilder.CreateNot(phi);

        for (auto pair : replacePHIs) {
            Value* replaceWith = nullptr;

            if (pair.first == loopContext.preheader) {
                replaceWith = phi;
            } else {
                replaceWith = nphi;
            }

            pair.second->replaceAllUsesWith(replaceWith);
            pair.second->eraseFromParent();
        }

        Builder.SetInsertPoint(BB2);

        Builder.CreateCondBr(phi, gutils->getReverseOrLatchMerge(loopContext.preheader, BB), targetToPreds.begin()->first);

    } else {
        std::map<BasicBlock*, std::vector< std::pair<BasicBlock*, BasicBlock*> >> phiTargetToPreds;
        for (auto pair : replacePHIs) {
            phiTargetToPreds[pair.first].emplace_back(std::make_pair(pair.first, BB));
        }
        BasicBlock* fakeTarget = nullptr;
        for (auto pred : predecessors(BB)) {
            if (phiTargetToPreds.find(pred) != phiTargetToPreds.end()) continue;
            if (fakeTarget == nullptr) fakeTarget = pred;
            phiTargetToPreds[fakeTarget].emplace_back(std::make_pair(pred, BB));
        }
        gutils->branchToCorrespondingTarget(BB, phibuilder, phiTargetToPreds, &replacePHIs);

        std::map<BasicBlock*, std::vector< std::pair<BasicBlock*, BasicBlock*> >> targetToPreds;
        for(auto pred : predecessors(BB)) {
            targetToPreds[gutils->getReverseOrLatchMerge(pred, BB)].emplace_back(std::make_pair(pred, BB));
        }
        Builder.SetInsertPoint(BB2);
        gutils->branchToCorrespondingTarget(BB, Builder, targetToPreds);
    }

    // Optimize select of not to just be a select with operands switched
    for (SelectInst* select : selects) {
        if (BinaryOperator* bo = dyn_cast<BinaryOperator>(select->getCondition())) {
            if (bo->getOpcode() == BinaryOperator::Xor) {
                //llvm::errs() << " considering " << *select << " " << *bo << "\n";
                if (isa<ConstantInt>(bo->getOperand(0)) && cast<ConstantInt>(bo->getOperand(0))->isOne()) {
                    select->setCondition(bo->getOperand(1));
                    auto tmp = select->getTrueValue();
                    select->setTrueValue(select->getFalseValue());
                    select->setFalseValue(tmp);
                    if (bo->getNumUses() == 0) bo->eraseFromParent();
                } else if (isa<ConstantInt>(bo->getOperand(1)) && cast<ConstantInt>(bo->getOperand(1))->isOne()) {
                    select->setCondition(bo->getOperand(0));
                    auto tmp = select->getTrueValue();
                    select->setTrueValue(select->getFalseValue());
                    select->setFalseValue(tmp);
                    if (bo->getNumUses() == 0) bo->eraseFromParent();
                }
            }
        }
    }
}

Function* CreatePrimalAndGradient(Function* todiff, DIFFE_TYPE retType, const std::vector<DIFFE_TYPE>& constant_args, TargetLibraryInfo &TLI,
                                  TypeAnalysis &TA, AAResults &global_AA, bool returnUsed, bool dretPtr, bool topLevel, llvm::Type* additionalArg,
                                  const NewFnTypeInfo& oldTypeInfo_, const std::map<Argument*, bool> _uncacheable_args,
                                  const AugmentedReturn* augmenteddata) {

  NewFnTypeInfo oldTypeInfo = oldTypeInfo_;
  for(auto &pair : oldTypeInfo.knownValues) {
    if (pair.second.size() != 0) {
      bool recursiveUse = false;
      for(auto user : pair.first->users()) {
        if (auto bi = dyn_cast<BinaryOperator>(user)) {
          for( auto biuser : bi->users()) {
            if (auto ci = dyn_cast<CallInst>(biuser)) {
              if (ci->getCalledFunction() == todiff && ci->getArgOperand(pair.first->getArgNo()) == bi) {
                recursiveUse = true;
                break;
              }
            }
          }
        }
        if (recursiveUse) break;
      }
      if (recursiveUse) pair.second.clear();
    }
  }
  //if (additionalArg && !additionalArg->isStructTy()) {
  //    llvm::errs() << *todiff << "\n";
  //    llvm::errs() << "addl arg: " << *additionalArg << "\n";
  //}
  if (retType != DIFFE_TYPE::CONSTANT) assert(!todiff->getReturnType()->isVoidTy());
  static std::map<std::tuple<Function*,DIFFE_TYPE/*retType*/,std::vector<DIFFE_TYPE>/*constant_args*/, std::map<Argument*, bool>/*uncacheable_args*/, bool/*retval*/, bool/*dretPtr*/, bool/*topLevel*/, llvm::Type*, const NewFnTypeInfo>, Function*> cachedfunctions;
  auto tup = std::make_tuple(todiff, retType, constant_args, std::map<Argument*, bool>(_uncacheable_args.begin(), _uncacheable_args.end()), returnUsed, dretPtr, topLevel, additionalArg, oldTypeInfo);
  if (cachedfunctions.find(tup) != cachedfunctions.end()) {
    return cachedfunctions.find(tup)->second;
  }

  /*
  llvm::errs() << "taking grad " << todiff->getName() << " retType: " << tostring(retType) << " topLevel:" << topLevel << " [";
  for(auto a : constant_args) {
    llvm::errs() << tostring(a) << ",";
  }
  llvm::errs() << "]\n";
  */

  //Whether we shuold actually return the value
  bool returnValue = returnUsed && topLevel;
  //llvm::errs() << " returnValue: " << returnValue <<  "toplevel: " << topLevel << " func: " << todiff->getName() << "\n";

  bool hasTape = false;

  // TODO change this to go by default function type assumptions
  bool hasconstant = false;
  for(auto v : constant_args) {
    if (v == DIFFE_TYPE::CONSTANT) {
      hasconstant = true;
      break;
    }
  }

  if (!hasconstant && !topLevel && !returnValue && hasMetadata(todiff, "enzyme_gradient")) {

      auto md = todiff->getMetadata("enzyme_gradient");
      if (!isa<MDTuple>(md)) {
          llvm::errs() << *todiff << "\n";
          llvm::errs() << *md << "\n";
          report_fatal_error("unknown gradient for noninvertible function -- metadata incorrect");
      }
      auto md2 = cast<MDTuple>(md);
      assert(md2->getNumOperands() == 1);
      auto gvemd = cast<ConstantAsMetadata>(md2->getOperand(0));
      auto foundcalled = cast<Function>(gvemd->getValue());

      DIFFE_TYPE subretType = todiff->getReturnType()->isFPOrFPVectorTy() ? DIFFE_TYPE::OUT_DIFF : DIFFE_TYPE::DUP_ARG;
      if (todiff->getReturnType()->isVoidTy() || todiff->getReturnType()->isEmptyTy()) subretType = DIFFE_TYPE::CONSTANT;
      auto res = getDefaultFunctionTypeForGradient(todiff->getFunctionType(), /*retType*/subretType);


      if (foundcalled->arg_size() == res.first.size() + 1 /*tape*/) {
        auto lastarg = foundcalled->arg_end();
        lastarg--;
        res.first.push_back(lastarg->getType());
        hasTape = true;
      } else if (foundcalled->arg_size() == res.first.size()) {
        res.first.push_back(StructType::get(todiff->getContext(), {}));
      } else {
          llvm::errs() << "expected args: [";
          for(auto a : res.first) {
                llvm::errs() << *a << " ";
          }
          llvm::errs() << "]\n";
          llvm::errs() << *foundcalled << "\n";
          assert(0 && "bad type for custom gradient");
      }

      auto st = dyn_cast<StructType>(foundcalled->getReturnType());
      bool wrongRet = st == nullptr;
      if (wrongRet || !hasTape) {
        FunctionType *FTy = FunctionType::get(StructType::get(todiff->getContext(), {res.second}), res.first, todiff->getFunctionType()->isVarArg());
        Function *NewF = Function::Create(FTy, Function::LinkageTypes::InternalLinkage, "fixgradient_"+todiff->getName(), todiff->getParent());
        NewF->setAttributes(foundcalled->getAttributes());
        if (NewF->hasFnAttribute(Attribute::NoInline)) {
            NewF->removeFnAttr(Attribute::NoInline);
        }
          for (Argument &Arg : NewF->args()) {
              if (Arg.hasAttribute(Attribute::Returned))
                  Arg.removeAttr(Attribute::Returned);
              if (Arg.hasAttribute(Attribute::StructRet))
                  Arg.removeAttr(Attribute::StructRet);
          }

        BasicBlock *BB = BasicBlock::Create(NewF->getContext(), "entry", NewF);
        IRBuilder <>bb(BB);
        SmallVector<Value*,4> args;
        for(auto &a : NewF->args()) args.push_back(&a);
        if (!hasTape) {
            args.pop_back();
        }
        llvm::errs() << *NewF << "\n";
        llvm::errs() << *foundcalled << "\n";
        auto cal = bb.CreateCall(foundcalled, args);
        cal->setCallingConv(foundcalled->getCallingConv());
        Value* val = cal;
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
      return insert_or_assign(cachedfunctions, tup, foundcalled)->second;
  }

  assert(!todiff->empty());
  auto M = todiff->getParent();

  AAResults AA(TLI);
  //AA.addAAResult(global_AA);
  DiffeGradientUtils *gutils = DiffeGradientUtils::CreateFromClone(topLevel, todiff, TLI, TA, AA, retType, constant_args, returnValue ? ( dretPtr ? ReturnType::ArgsWithTwoReturns: ReturnType::ArgsWithReturn ) : ReturnType::Args, additionalArg);
  insert_or_assign(cachedfunctions, tup, gutils->newFunc);

  const SmallPtrSet<BasicBlock*, 4> guaranteedUnreachable = getGuaranteedUnreachable(gutils->oldFunc);

  SmallPtrSet<Value*, 4> assumeTrue;
  SmallPtrSet<Value*, 4> assumeFalse;

  if (!topLevel) {
    //TODO also can consider switch instance as well
    // TODO can also insert to topLevel as well [note this requires putting the intrinsic at the correct location]
    for(auto& BB : *gutils->oldFunc) {
      std::vector<BasicBlock*> unreachables;
      std::vector<BasicBlock*> reachables;
      for(auto Succ : successors(&BB)) {
        if (guaranteedUnreachable.find(Succ) != guaranteedUnreachable.end()) {
          unreachables.push_back(Succ);
        } else {
          reachables.push_back(Succ);
        }
      }

      if (unreachables.size() == 0 || reachables.size() == 0) continue;

      if (auto bi = dyn_cast<BranchInst>(BB.getTerminator())) {
        IRBuilder<> B(&gutils->newFunc->getEntryBlock().front());

        if (auto inst = dyn_cast<Instruction>(bi->getCondition())) {
          B.SetInsertPoint(gutils->getNewFromOriginal(inst)->getNextNode());
        }

        Value* vals[1] = { gutils->getNewFromOriginal(bi->getCondition()) };
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

  NewFnTypeInfo typeInfo(gutils->oldFunc);
  {
    auto toarg = todiff->arg_begin();
    auto olarg = gutils->oldFunc->arg_begin();
    for(; toarg != todiff->arg_end(); toarg++, olarg++) {

      {
      auto fd = oldTypeInfo.first.find(toarg);
      assert(fd != oldTypeInfo.first.end());
      typeInfo.first.insert(std::pair<Argument*, ValueData>(olarg, fd->second));
      }

      {
      auto cfd = oldTypeInfo.knownValues.find(toarg);
      assert(cfd != oldTypeInfo.knownValues.end());
      typeInfo.knownValues.insert(std::pair<Argument*, std::set<int64_t>>(olarg, cfd->second));
      }
    }
    typeInfo.second = oldTypeInfo.second;
  }

  TypeResults TR = TA.analyzeFunction(typeInfo);
  assert(TR.info.function == gutils->oldFunc);

  gutils->forceActiveDetection(AA, TR);
  gutils->forceAugmentedReturns(TR, guaranteedUnreachable);

  std::map<std::pair<Instruction*, CacheType>, int> mapping;
  if (augmenteddata) mapping = augmenteddata->tapeIndices;

  auto getIndex = [&](Instruction* I, CacheType u)-> unsigned {
    return gutils->getIndex( std::make_pair(I, u), mapping);
  };

  // Convert uncacheable args from the input function to the preprocessed function
  std::map<Argument*, bool> _uncacheable_argsPP;
  {
    auto in_arg = todiff->arg_begin();
    auto pp_arg = gutils->oldFunc->arg_begin();
    for(; pp_arg != gutils->oldFunc->arg_end(); ) {
        _uncacheable_argsPP[pp_arg] = _uncacheable_args.find(in_arg)->second;
        pp_arg++;
        in_arg++;
    }
  }

  SmallPtrSet<const Value*,4> unnecessaryValues;
  SmallPtrSet<const Instruction*, 4> unnecessaryInstructions;
  calculateUnusedValues(*gutils->oldFunc, unnecessaryValues, unnecessaryInstructions, returnValue, [&](const Value* val) {
    return is_value_needed_in_reverse(TR, gutils, val, /*topLevel*/topLevel);
  }, [&](const Instruction* inst) {
    if (auto II = dyn_cast<IntrinsicInst>(inst)) {
      if (II->getIntrinsicID() == Intrinsic::lifetime_start || II->getIntrinsicID() == Intrinsic::lifetime_end ||
          II->getIntrinsicID() == Intrinsic::stacksave || II->getIntrinsicID() == Intrinsic::stackrestore) {
        return false;
      }
    }

    if (auto obj_op = dyn_cast<CallInst>(inst)) {
      Function* called = obj_op->getCalledFunction();
      if (auto castinst = dyn_cast<ConstantExpr>(obj_op->getCalledValue())) {
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
      if (isa<UndefValue>(si->getValueOperand())) return false;
      auto at = GetUnderlyingObject(si->getPointerOperand(), gutils->oldFunc->getParent()->getDataLayout(), 100);
      if (auto arg = dyn_cast<Argument>(at)) {
        if (constant_args[arg->getArgNo()] == DIFFE_TYPE::DUP_NONEED) {
          return false;
        }
      }
    }

    if (auto mti = dyn_cast<MemTransferInst>(inst)) {
      auto at = GetUnderlyingObject(mti->getArgOperand(1), gutils->oldFunc->getParent()->getDataLayout(), 100);
      if (auto arg = dyn_cast<Argument>(at)) {
        if (constant_args[arg->getArgNo()] == DIFFE_TYPE::DUP_NONEED) {
          return false;
        }
      }
      //llvm::errs() << "origop for mti: " << *mti << " at:" << *at << "\n";
      if (auto ai = dyn_cast<AllocaInst>(at)) {
          bool foundStore = false;
          allInstructionsBetween(gutils->OrigLI, ai, const_cast<MemTransferInst*>(mti), [&](Instruction* I) -> bool {
            if (!I->mayWriteToMemory()) return /*earlyBreak*/false;
            if (unnecessaryInstructions.count(I)) return /*earlyBreak*/false;

            //if (I == &MTI) return;
            if (writesToMemoryReadBy(gutils->AA, /*maybeReader*/const_cast<MemTransferInst*>(mti), /*maybeWriter*/I)) {
              //llvm::errs() << " mti: - " << *mti << " stored into by " << *I << "\n";
              foundStore = true;
              return /*earlyBreak*/true;
            }
            return /*earlyBreak*/false;
          });
          if (!foundStore) {
            //llvm::errs() << "warning - performing a memcpy out of unitialized memory: " << *mti << "\n";
            return false;
          }
      }
    }

    return (topLevel && inst->mayWriteToMemory()) || is_value_needed_in_reverse(TR, gutils, inst, /*topLevel*/topLevel);
  });

  SmallPtrSet<const Instruction*, 4> unnecessaryStores;
  calculateUnusedStores(*gutils->oldFunc, unnecessaryStores, [&](const Instruction* inst) {

    if (auto si = dyn_cast<StoreInst>(inst)) {
      if (isa<UndefValue>(si->getValueOperand())) return false;
    }

    if (auto mti = dyn_cast<MemTransferInst>(inst)) {
      auto at = GetUnderlyingObject(mti->getArgOperand(1), gutils->oldFunc->getParent()->getDataLayout(), 100);
      //llvm::errs() << "origop for mti: " << *mti << " at:" << *at << "\n";
      if (auto ai = dyn_cast<AllocaInst>(at)) {
          bool foundStore = false;
          allInstructionsBetween(gutils->OrigLI, ai, const_cast<MemTransferInst*>(mti), [&](Instruction* I) -> bool {
            if (!I->mayWriteToMemory()) return /*earlyBreak*/false;
            if (unnecessaryInstructions.count(I)) return /*earlyBreak*/false;

            //if (I == &MTI) return;
            if (writesToMemoryReadBy(gutils->AA, /*maybeReader*/const_cast<MemTransferInst*>(mti), /*maybeWriter*/I)) {
              //llvm::errs() << " mti: - " << *mti << " stored into by " << *I << "\n";
              foundStore = true;
              return /*earlyBreak*/true;
            }
            return /*earlyBreak*/false;
          });
          if (!foundStore) {
            //llvm::errs() << "warning - performing a memcpy out of unitialized memory: " << *mti << "\n";
            return false;
          }
      }
    }

    return true;
  });

  const std::map<CallInst*, const std::map<Argument*, bool> > uncacheable_args_map = (augmenteddata) ? augmenteddata->uncacheable_args_map :
      compute_uncacheable_args_for_callsites(gutils->oldFunc, gutils->DT, TLI, unnecessaryInstructions, AA, gutils, _uncacheable_argsPP);

  const std::map<Instruction*, bool> can_modref_map =  augmenteddata ? augmenteddata->can_modref_map : compute_uncacheable_load_map(gutils, AA, TLI, unnecessaryInstructions, _uncacheable_argsPP);
  /*
    for (auto &iter : can_modref_map_mutable) {
      if (iter.second) {
        //llvm::errs() << "isneeded: " << is_needed << " gradient can_modref_map: " << *iter.first << " fn: " << gutils->oldFunc->getName() << "\n";
      } else {
        //llvm::errs() << "gradient can_modref_map: " << *iter.first << "\n";
      }
    }
  */

  gutils->can_modref_map = &can_modref_map;

  Value* additionalValue = nullptr;
  if (additionalArg) {
    auto v = gutils->newFunc->arg_end();
    v--;
    additionalValue = v;
    assert(!topLevel);
    assert(augmenteddata);

    // TODO VERIFY THIS
    if (augmenteddata->tapeType && augmenteddata->tapeType != additionalValue->getType() ) {
        IRBuilder<> BuilderZ(gutils->inversionAllocs);
        //assert(PointerType::getUnqual(augmenteddata->tapeType) == additionalValue->getType());
        //auto tapep = additionalValue;
        auto tapep = BuilderZ.CreatePointerCast(additionalValue, PointerType::getUnqual(augmenteddata->tapeType));
        LoadInst* truetape = BuilderZ.CreateLoad(tapep);
        truetape->setMetadata("enzyme_mustcache", MDNode::get(truetape->getContext(), {}));

        CallInst* ci = cast<CallInst>(CallInst::CreateFree(additionalValue, truetape));//&*BuilderZ.GetInsertPoint()));
        ci->moveAfter(truetape);
        ci->addAttribute(AttributeList::FirstArgIndex, Attribute::NonNull);
        additionalValue = truetape;
    }

    //TODO here finish up making recursive structs simply pass in i8*
    //blahblahblah
    gutils->setTape(additionalValue);
  }

  Argument* differetval = nullptr;
  if (retType == DIFFE_TYPE::OUT_DIFF) {
    auto endarg = gutils->newFunc->arg_end();
    endarg--;
    if (additionalArg) endarg--;
    differetval = endarg;
    if (differetval->getType() != todiff->getReturnType()) {
        llvm::errs() << *gutils->oldFunc << "\n";
        llvm::errs() << *gutils->newFunc << "\n";
    }
    assert(differetval->getType() == todiff->getReturnType());
  }

  // Explicitly handle all returns first to ensure that return instructions know if they are used or not before
  //   processessing instructions
  std::map<ReturnInst*,StoreInst*> replacedReturns;
  llvm::AllocaInst* retAlloca = nullptr;
  llvm::AllocaInst* dretAlloca = nullptr;
  if (returnValue) {
    retAlloca = IRBuilder<>(&gutils->newFunc->getEntryBlock().front()).CreateAlloca(todiff->getReturnType(), nullptr, "toreturn");
    if (dretPtr && (retType == DIFFE_TYPE::DUP_ARG || retType == DIFFE_TYPE::DUP_NONEED) && !topLevel) {
        dretAlloca = IRBuilder<>(&gutils->newFunc->getEntryBlock().front()).CreateAlloca(todiff->getReturnType(), nullptr, "dtoreturn");
    }

  }

  //! Ficticious values with TBAA to use for constant detection algos until everything is made fully ahead of time
  //   Note that we need to delete the tbaa tags from these values once we finish / before verification
  std::vector<Instruction*> fakeTBAA;

  for(BasicBlock& oBB: *gutils->oldFunc) {
    if(ReturnInst* orig = dyn_cast<ReturnInst>(oBB.getTerminator())) {
      ReturnInst* op = cast<ReturnInst>(gutils->getNewFromOriginal(orig));
      BasicBlock* BB = op->getParent();
      IRBuilder<> rb(op);
      rb.setFastMathFlags(getFast());

      if (retAlloca) {
        StoreInst* si = rb.CreateStore(gutils->getNewFromOriginal(orig->getReturnValue()), retAlloca);
        replacedReturns[orig] = si;

        if (dretAlloca && !gutils->isConstantValue(orig->getReturnValue())) {
            rb.CreateStore(gutils->invertPointerM(orig->getReturnValue(), rb), dretAlloca);
        }
      }

      if (retType == DIFFE_TYPE::OUT_DIFF) {
        assert(orig->getReturnValue());
        assert(differetval);
        if (!gutils->isConstantValue(orig->getReturnValue())) {
          IRBuilder <>reverseB(gutils->reverseBlocks[BB]);
          gutils->setDiffe(orig->getReturnValue(), differetval, reverseB);
        }
      } else {
        assert (retAlloca == nullptr);
      }

      rb.CreateBr(gutils->reverseBlocks[BB]);
      gutils->erase(op);
    }
  }

  DerivativeMaker<const AugmentedReturn*> maker( topLevel ? DerivativeMode::Both : DerivativeMode::Reverse, gutils, constant_args, TR, getIndex, uncacheable_args_map, /*returnuses*/nullptr, augmenteddata, &fakeTBAA, &replacedReturns, unnecessaryValues, unnecessaryInstructions, unnecessaryStores, dretAlloca);

  for(BasicBlock& oBB: *gutils->oldFunc) {
    // Don't create derivatives for code that results in termination
    if (guaranteedUnreachable.find(&oBB) != guaranteedUnreachable.end()) {
        std::vector<Instruction*> toerase;
        for(auto &I: oBB) { toerase.push_back(&I); }
        for(auto I : toerase) { maker.eraseIfUnused(*I, /*erase*/true, /*check*/topLevel == true); }
        auto newBB = cast<BasicBlock>(gutils->getNewFromOriginal(&oBB));
        for(auto next : successors(&oBB)) {
          auto sucBB = cast<BasicBlock>(gutils->getNewFromOriginal(next));
          sucBB->removePredecessor(newBB);
        }
        if (newBB->getTerminator())
          newBB->getTerminator()->eraseFromParent();
        IRBuilder<> builder(newBB);
        builder.CreateUnreachable();
        continue;
    }

    auto term = oBB.getTerminator();
    assert(term);
    if (!isa<ReturnInst>(term) && !isa<BranchInst>(term) && !isa<SwitchInst>(term)) {
      llvm::errs() << *oBB.getParent() << "\n";
      llvm::errs() << "unknown terminator instance " << *term << "\n";
      assert(0 && "unknown terminator inst");
    }

    BasicBlock::reverse_iterator I = oBB.rbegin(), E = oBB.rend();
    I++;
    for (; I != E; I++) {
      //llvm::errs() << "inst: " << *I << "\n";
      //llvm::errs() << "   + constval:" << gutils->isConstantValue(&*I) << " constinst:" << gutils->isConstantInstruction(&*I) << " is_value_needed_in_reverse:" << is_value_needed_in_reverse(TR, gutils, &*I, topLevel) << "\n";
      maker.visit(&*I);
      assert(oBB.rend() == E);
    }
    createInvertedTerminator(TR, gutils, constant_args, &oBB, retAlloca, dretAlloca, 0 + (additionalArg ? 1 : 0) + ( (retType == DIFFE_TYPE::DUP_ARG || retType == DIFFE_TYPE::DUP_NONEED) ? 1 : 0));
  }

  gutils->eraseFictiousPHIs();

  for(auto inst: fakeTBAA) {
      inst->setMetadata(LLVMContext::MD_tbaa, nullptr);
  }

  for(auto val : assumeTrue) {
    bool changed;
    do {
      changed = false;
      for(auto &use : val->uses()) {
        if (auto user = dyn_cast<IntrinsicInst>(use.getUser())) {
          if (user->getIntrinsicID() == Intrinsic::assume) continue;
        }
        use.set(ConstantInt::getTrue(val->getContext()));
        changed = true;
        break;
      }
    }while (!changed);
  }

  for(auto val : assumeFalse) {
    bool changed;
    do {
      changed = false;
      for(auto &use : val->uses()) {
        if (auto notu = dyn_cast<BinaryOperator>(use.getUser())) {
          if (notu->getNumUses() == 1 && notu->getOpcode() == BinaryOperator::Xor && notu->getOperand(0) == val && isa<ConstantInt>(notu->getOperand(1)) && cast<ConstantInt>(notu->getOperand(1))->isOne()) {
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

  while(gutils->inversionAllocs->size() > 0) {
    gutils->inversionAllocs->back().moveBefore(gutils->newFunc->getEntryBlock().getFirstNonPHIOrDbgOrLifetime());
  }

  (IRBuilder <>(gutils->inversionAllocs)).CreateUnreachable();
  DeleteDeadBlock(gutils->inversionAllocs);
  for(auto BBs : gutils->reverseBlocks) {
    if (pred_begin(BBs.second) == pred_end(BBs.second)) {
        (IRBuilder <>(BBs.second)).CreateUnreachable();
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

  if (auto bytes = gutils->newFunc->getDereferenceableBytes(llvm::AttributeList::ReturnIndex)) {
    AttrBuilder ab;
    ab.addDereferenceableAttr(bytes);
    gutils->newFunc->removeAttributes(llvm::AttributeList::ReturnIndex, ab);
  }
  if (gutils->newFunc->hasAttribute(llvm::AttributeList::ReturnIndex, llvm::Attribute::NoAlias)) {
    gutils->newFunc->removeAttribute(llvm::AttributeList::ReturnIndex, llvm::Attribute::NoAlias);
  }
  if (gutils->newFunc->hasAttribute(llvm::AttributeList::ReturnIndex, llvm::Attribute::ZExt)) {
    gutils->newFunc->removeAttribute(llvm::AttributeList::ReturnIndex, llvm::Attribute::ZExt);
  }

  if (llvm::verifyFunction(*gutils->newFunc, &llvm::errs())) {
      llvm::errs() << *gutils->oldFunc << "\n";
      llvm::errs() << *gutils->newFunc << "\n";
      report_fatal_error("function failed verification (4)");
  }

  gutils->cleanupActiveDetection();

  optimizeIntermediate(gutils, topLevel, gutils->newFunc);

  auto nf = gutils->newFunc;
  delete gutils;

  return nf;
}
