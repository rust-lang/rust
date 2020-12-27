//===- DifferentialUseAnalysis.h - Determine values needed in reverse pass-===//
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
// This file contains the declaration of Differential USe Analysis -- an
// AD-specific analysis that deduces if a given value is needed in the reverse
// pass.
//
//===----------------------------------------------------------------------===//
#include "GradientUtils.h"

enum ValueType { Primal, Shadow };

// Determine if a value is needed in the reverse pass. We only use this logic in
// the top level function right now.
template <ValueType VT>
bool is_value_needed_in_reverse(
    TypeResults &TR, const GradientUtils *gutils, const Value *inst,
    bool topLevel, std::map<std::pair<const Value *, bool>, bool> &seen,
    const SmallPtrSetImpl<BasicBlock *> &oldUnreachable) {
  auto idx = std::make_pair(inst, topLevel);
  if (seen.find(idx) != seen.end())
    return seen[idx];
  if (auto ainst = dyn_cast<Instruction>(inst)) {
    assert(ainst->getParent()->getParent() == gutils->oldFunc);
  }

  // Inductively claim we aren't needed (and try to find contradiction)
  seen[idx] = false;

  // Consider all users of this value, do any of them need this in the reverse?
  for (auto use : inst->users()) {
    if (use == inst)
      continue;

    const Instruction *user = dyn_cast<Instruction>(use);

    // A shadow value is only needed in reverse if it or one of its descendants
    // is used in an active instruction
    if (VT == Shadow) {
      if (user)
        if (!gutils->isConstantInstruction(const_cast<Instruction *>(user)))
          return true;
      if (is_value_needed_in_reverse<Shadow>(TR, gutils, use, topLevel, seen,
                                             oldUnreachable)) {
        return true;
      }
      continue;
    }

    assert(VT == Primal);

    // One may need to this value in the computation of loop
    // bounds/comparisons/etc (which even though not active -- will be used for
    // the reverse pass)
    //   We only need this if we're not doing the combined forward/reverse since
    //   otherwise it will use the local cache (rather than save for a separate
    //   backwards cache)
    //   We also don't need this if looking at the shadow rather than primal
    if (!topLevel) {
      // Proving that none of the uses (or uses' uses) are used in control flow
      // allows us to safely not do this load

      // TODO save loop bounds for dynamic loop

      // TODO make this more aggressive and dont need to save loop latch
      if (isa<BranchInst>(use) || isa<SwitchInst>(use)) {
        size_t num = 0;
        for (auto suc : successors(cast<Instruction>(use)->getParent())) {
          if (!oldUnreachable.count(suc)) {
            num++;
          }
        }
        if (num <= 1)
          continue;
        return seen[idx] = true;
      }

      if (auto CI = dyn_cast<CallInst>(use)) {
        if (auto F = CI->getCalledFunction()) {
          if (F->getName() == "__kmpc_for_static_init_4" ||
              F->getName() == "__kmpc_for_static_init_4u" ||
              F->getName() == "__kmpc_for_static_init_8" ||
              F->getName() == "__kmpc_for_static_init_8u") {
            return seen[idx] = true;
          }
        }
      }

      if (is_value_needed_in_reverse<VT>(TR, gutils, user, topLevel, seen,
                                         oldUnreachable)) {
        return seen[idx] = true;
      }
    }

    // The following are types we know we don't need to compute adjoints

    // A pointer is only needed in the reverse pass if its non-store uses are
    // needed in the reverse pass
    //   Moreover, we only need this pointer in the reverse pass if all of its
    //   non-store users are not already cached for the reverse pass
    if (!inst->getType()->isFPOrFPVectorTy() &&
        TR.query(const_cast<Value *>(inst)).Inner0().isPossiblePointer()) {
      // continue;
      bool unknown = true;
      for (auto zu : inst->users()) {
        // Stores to a pointer are not needed for the reverse pass
        if (auto si = dyn_cast<StoreInst>(zu)) {
          if (si->getPointerOperand() == inst) {
            continue;
          }
        }

        if (isa<LoadInst>(zu) || isa<CastInst>(zu) || isa<PHINode>(zu)) {
          if (is_value_needed_in_reverse<VT>(TR, gutils, zu, topLevel, seen,
                                             oldUnreachable)) {
            return seen[idx] = true;
          }
          continue;
        }

        if (auto II = dyn_cast<IntrinsicInst>(zu)) {
          if (II->getIntrinsicID() == Intrinsic::lifetime_start ||
              II->getIntrinsicID() == Intrinsic::lifetime_end ||
              II->getIntrinsicID() == Intrinsic::stacksave ||
              II->getIntrinsicID() == Intrinsic::stackrestore ||
              II->getIntrinsicID() == Intrinsic::nvvm_barrier0_popc ||
              II->getIntrinsicID() == Intrinsic::nvvm_barrier0_and ||
              II->getIntrinsicID() == Intrinsic::nvvm_barrier0_or) {
            continue;
          }
        }

        if (auto ci = dyn_cast<CallInst>(zu)) {
          // If this instruction isn't constant (and thus we need the argument
          // to propagate to its adjoint)
          //   it may write memory and is topLevel (and thus we need to do the
          //   write in reverse) or we need this value for the reverse pass (we
          //   conservatively assume that if legal it is recomputed and not
          //   stored)
          if (!gutils->isConstantInstruction(ci) ||
              !gutils->isConstantValue(
                  const_cast<Value *>((const Value *)ci)) ||
              (ci->mayWriteToMemory() && topLevel) ||
              (gutils->legalRecompute(ci, ValueToValueMapTy()) &&
               is_value_needed_in_reverse<VT>(TR, gutils, ci, topLevel, seen,
                                              oldUnreachable))) {
            return seen[idx] = true;
          }
          continue;
        }

        // TODO add handling of call and allow interprocedural
        unknown = true;
      }
      if (!unknown)
        continue;
      // return seen[inst] = false;
    }

    if (isa<LoadInst>(user) || isa<CastInst>(user) || isa<PHINode>(user)) {
      if (!is_value_needed_in_reverse<VT>(TR, gutils, user, topLevel, seen,
                                          oldUnreachable)) {
        continue;
      }
    }

    if (auto II = dyn_cast<IntrinsicInst>(user)) {
      if (II->getIntrinsicID() == Intrinsic::lifetime_start ||
          II->getIntrinsicID() == Intrinsic::lifetime_end ||
          II->getIntrinsicID() == Intrinsic::stacksave ||
          II->getIntrinsicID() == Intrinsic::stackrestore) {
        continue;
      }
      if (II->getIntrinsicID() == Intrinsic::fma) {
        bool needed = false;
        if (II->getArgOperand(0) == inst &&
            !gutils->isConstantValue(II->getArgOperand(1)))
          needed = true;
        if (II->getArgOperand(1) == inst &&
            !gutils->isConstantValue(II->getArgOperand(0)))
          needed = true;
        if (!needed)
          continue;
      }
    }

    if (auto op = dyn_cast<BinaryOperator>(user)) {
      if (op->getOpcode() == Instruction::FAdd ||
          op->getOpcode() == Instruction::FSub) {
        continue;
      } else if (op->getOpcode() == Instruction::FMul) {
        bool needed = false;
        if (op->getOperand(0) == inst &&
            !gutils->isConstantValue(op->getOperand(1)))
          needed = true;
        if (op->getOperand(1) == inst &&
            !gutils->isConstantValue(op->getOperand(0)))
          needed = true;
        if (!needed)
          continue;
      } else if (op->getOpcode() == Instruction::FDiv) {
        bool needed = false;
        if (op->getOperand(1) == inst &&
            !gutils->isConstantValue(op->getOperand(1)))
          needed = true;
        if (op->getOperand(1) == inst &&
            !gutils->isConstantValue(op->getOperand(0)))
          needed = true;
        if (op->getOperand(0) == inst &&
            !gutils->isConstantValue(op->getOperand(1)))
          needed = true;
        if (!needed)
          continue;
      } else
        continue;
    }

    // We don't need only the indices of a GEP to compute the adjoint of a GEP
    if (auto gep = dyn_cast<GetElementPtrInst>(user)) {
      bool indexuse = false;
      for (auto &idx : gep->indices()) {
        if (idx == inst) {
          indexuse = true;
        }
      }
      if (!indexuse)
        continue;
    }

    if (auto si = dyn_cast<SelectInst>(use)) {
      // only need the condition if select is active
      if (gutils->isConstantValue(const_cast<SelectInst *>(si)))
        continue;
      //   none of the other operands are needed otherwise
      if (si->getCondition() != inst) {
        continue;
      }
    }

    // We don't need any of the input operands to compute the adjoint of a store
    // instance
    if (isa<StoreInst>(use)) {
      continue;
    }

    if (isa<CmpInst>(use) || isa<BranchInst>(use) || isa<CastInst>(use) ||
        isa<PHINode>(use) || isa<ReturnInst>(use) || isa<FPExtInst>(use) ||
        (isa<InsertElementInst>(use) &&
         cast<InsertElementInst>(use)->getOperand(2) != inst) ||
        (isa<ExtractElementInst>(use) &&
         cast<ExtractElementInst>(use)->getIndexOperand() != inst)
        // isa<LoadInst>(use) || (isa<SelectInst>(use) &&
        // cast<SelectInst>(use)->getCondition() != inst) //TODO remove load?
        //|| isa<SwitchInst>(use) || isa<ExtractElement>(use) ||
        // isa<InsertElementInst>(use) || isa<ShuffleVectorInst>(use) ||
        // isa<ExtractValueInst>(use) || isa<AllocaInst>(use)
        /*|| isa<StoreInst>(use)*/) {
      continue;
    }

    //! Note it is important that return check comes before this as it may not
    //! have a new instruction

    if (auto ci = dyn_cast<CallInst>(use)) {
      // If this instruction isn't constant (and thus we need the argument to
      // propagate to its adjoint)
      //   it may write memory and is topLevel (and thus we need to do the write
      //   in reverse) or we need this value for the reverse pass (we
      //   conservatively assume that if legal it is recomputed and not stored)
      if (!gutils->isConstantInstruction(ci) ||
          !gutils->isConstantValue(const_cast<Value *>((const Value *)ci)) ||
          (ci->mayWriteToMemory() && topLevel) ||
          (gutils->legalRecompute(ci, ValueToValueMapTy()) &&
           is_value_needed_in_reverse<VT>(TR, gutils, ci, topLevel, seen,
                                          oldUnreachable))) {
        return seen[idx] = true;
      }
      continue;
    }

    if (auto inst = dyn_cast<Instruction>(use))
      if (gutils->isConstantInstruction(const_cast<Instruction *>(inst)) &&
          gutils->isConstantValue(const_cast<Instruction *>(inst)))
        continue;

    return seen[idx] = true;
  }
  return false;
}

template <ValueType VT>
bool is_value_needed_in_reverse(
    TypeResults &TR, const GradientUtils *gutils, const Value *inst,
    bool topLevel, const SmallPtrSetImpl<BasicBlock *> &oldUnreachable) {
  std::map<std::pair<const Value *, bool>, bool> seen;
  return is_value_needed_in_reverse<VT>(TR, gutils, inst, topLevel, seen,
                                        oldUnreachable);
}