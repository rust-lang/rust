
//===- MustExitScalarEvolution.cpp - ScalarEvolution assuming loops terminate-=//
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
// This file defines MustExitScalarEvolution, a subclass of ScalarEvolution
// that assumes that all loops terminate (and don't loop forever).
//
//===----------------------------------------------------------------------===//

#include "MustExitScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/LoopInfo.h"

using namespace llvm;

#if LLVM_VERSION_MAJOR >= 7
ScalarEvolution::ExitLimit
MustExitScalarEvolution::computeExitLimitFromCond(const Loop *L, Value *ExitCond,
                                            bool ExitIfTrue, bool ControlsExit,
                                            bool AllowPredicates) {
  ScalarEvolution::ExitLimitCacheTy Cache(L, ExitIfTrue, AllowPredicates);
  return computeExitLimitFromCondCached(Cache, L, ExitCond, ExitIfTrue,
                                        ControlsExit, AllowPredicates);
}
#else
ScalarEvolution::ExitLimit MustExitScalarEvolution::computeExitLimitFromCond(
    const Loop *L, Value *ExitCond, BasicBlock *TBB, BasicBlock *FBB,
    bool ControlsExit, bool AllowPredicates) {
  ScalarEvolution::ExitLimitCacheTy Cache(L, TBB, FBB, AllowPredicates);
  return computeExitLimitFromCondCached(Cache, L, ExitCond, TBB, FBB,
                                        ControlsExit, AllowPredicates);
}
#endif

ScalarEvolution::ExitLimit
MustExitScalarEvolution::computeExitLimit(const Loop *L, BasicBlock *ExitingBlock,
                                    bool AllowPredicates) {
  assert(L->contains(ExitingBlock) && "Exit count for non-loop block?");
  // If our exiting block does not dominate the latch, then its connection with
  // loop's exit limit may be far from trivial.
  const BasicBlock *Latch = L->getLoopLatch();
  if (!Latch || !DT.dominates(ExitingBlock, Latch))
    return getCouldNotCompute();

  bool IsOnlyExit = (L->getExitingBlock() != nullptr);
  auto *Term = ExitingBlock->getTerminator();
  if (BranchInst *BI = dyn_cast<BranchInst>(Term)) {
    assert(BI->isConditional() && "If unconditional, it can't be in loop!");
    bool ExitIfTrue = !L->contains(BI->getSuccessor(0));
    assert(ExitIfTrue == L->contains(BI->getSuccessor(1)) &&
           "It should have one successor in loop and one exit block!");
    // Proceed to the next level to examine the exit condition expression.
    #if LLVM_VERSION_MAJOR >= 7
    return computeExitLimitFromCond(L, BI->getCondition(), ExitIfTrue,
                                    /*ControlsExit=*/IsOnlyExit,
                                    AllowPredicates);
    #else
    return computeExitLimitFromCond(
        L, BI->getCondition(), BI->getSuccessor(0), BI->getSuccessor(1),
        /*ControlsExit=*/IsOnlyExit, AllowPredicates);
    #endif
  }

  if (SwitchInst *SI = dyn_cast<SwitchInst>(Term)) {
    // For switch, make sure that there is a single exit from the loop.
    BasicBlock *Exit = nullptr;
    for (auto *SBB : successors(ExitingBlock))
      if (!L->contains(SBB)) {
        if (Exit) // Multiple exit successors.
          return getCouldNotCompute();
        Exit = SBB;
      }
    assert(Exit && "Exiting block must have at least one exit");
    return computeExitLimitFromSingleExitSwitch(L, SI, Exit,
                                                /*ControlsExit=*/IsOnlyExit);
  }

  return getCouldNotCompute();
}

#if LLVM_VERSION_MAJOR >= 7
ScalarEvolution::ExitLimit MustExitScalarEvolution::computeExitLimitFromCondCached(
    ExitLimitCacheTy &Cache, const Loop *L, Value *ExitCond, bool ExitIfTrue,
    bool ControlsExit, bool AllowPredicates) {

  if (auto MaybeEL =
          Cache.find(L, ExitCond, ExitIfTrue, ControlsExit, AllowPredicates))
    return *MaybeEL;

  ExitLimit EL = computeExitLimitFromCondImpl(Cache, L, ExitCond, ExitIfTrue,
                                              ControlsExit, AllowPredicates);
  Cache.insert(L, ExitCond, ExitIfTrue, ControlsExit, AllowPredicates, EL);
  return EL;
}
#else
ScalarEvolution::ExitLimit MustExitScalarEvolution::computeExitLimitFromCondCached(
    ExitLimitCacheTy &Cache, const Loop *L, Value *ExitCond, BasicBlock *TBB,
    BasicBlock *FBB, bool ControlsExit, bool AllowPredicates) {

  if (auto MaybeEL =
          Cache.find(L, ExitCond, TBB, FBB, ControlsExit, AllowPredicates))
    return *MaybeEL;

  ExitLimit EL = computeExitLimitFromCondImpl(Cache, L, ExitCond, TBB, FBB,
                                              ControlsExit, AllowPredicates);
  Cache.insert(L, ExitCond, TBB, FBB, ControlsExit, AllowPredicates, EL);
  return EL;
}
#endif

#if LLVM_VERSION_MAJOR >= 7
ScalarEvolution::ExitLimit MustExitScalarEvolution::computeExitLimitFromCondImpl(
    ExitLimitCacheTy &Cache, const Loop *L, Value *ExitCond, bool ExitIfTrue,
    bool ControlsExit, bool AllowPredicates) {
  // Check if the controlling expression for this loop is an And or Or.
  if (BinaryOperator *BO = dyn_cast<BinaryOperator>(ExitCond)) {
    if (BO->getOpcode() == Instruction::And) {
      // Recurse on the operands of the and.
      bool EitherMayExit = !ExitIfTrue;
      ExitLimit EL0 = computeExitLimitFromCondCached(
          Cache, L, BO->getOperand(0), ExitIfTrue,
          ControlsExit && !EitherMayExit, AllowPredicates);
      ExitLimit EL1 = computeExitLimitFromCondCached(
          Cache, L, BO->getOperand(1), ExitIfTrue,
          ControlsExit && !EitherMayExit, AllowPredicates);
      const SCEV *BECount = getCouldNotCompute();
      const SCEV *MaxBECount = getCouldNotCompute();
      if (EitherMayExit) {
        // Both conditions must be true for the loop to continue executing.
        // Choose the less conservative count.
        if (EL0.ExactNotTaken == getCouldNotCompute() ||
            EL1.ExactNotTaken == getCouldNotCompute())
          BECount = getCouldNotCompute();
        else
          BECount =
              getUMinFromMismatchedTypes(EL0.ExactNotTaken, EL1.ExactNotTaken);
        if (EL0.MaxNotTaken == getCouldNotCompute())
          MaxBECount = EL1.MaxNotTaken;
        else if (EL1.MaxNotTaken == getCouldNotCompute())
          MaxBECount = EL0.MaxNotTaken;
        else
          MaxBECount =
              getUMinFromMismatchedTypes(EL0.MaxNotTaken, EL1.MaxNotTaken);
      } else {
        // Both conditions must be true at the same time for the loop to exit.
        // For now, be conservative.
        if (EL0.MaxNotTaken == EL1.MaxNotTaken)
          MaxBECount = EL0.MaxNotTaken;
        if (EL0.ExactNotTaken == EL1.ExactNotTaken)
          BECount = EL0.ExactNotTaken;
      }

      // There are cases (e.g. PR26207) where computeExitLimitFromCond is able
      // to be more aggressive when computing BECount than when computing
      // MaxBECount.  In these cases it is possible for EL0.ExactNotTaken and
      // EL1.ExactNotTaken to match, but for EL0.MaxNotTaken and EL1.MaxNotTaken
      // to not.
      if (isa<SCEVCouldNotCompute>(MaxBECount) &&
          !isa<SCEVCouldNotCompute>(BECount))
        MaxBECount = getConstant(getUnsignedRangeMax(BECount));

      return ExitLimit(BECount, MaxBECount, false,
                       {&EL0.Predicates, &EL1.Predicates});
    }
    if (BO->getOpcode() == Instruction::Or) {
      // Recurse on the operands of the or.
      bool EitherMayExit = ExitIfTrue;
      ExitLimit EL0 = computeExitLimitFromCondCached(
          Cache, L, BO->getOperand(0), ExitIfTrue,
          ControlsExit && !EitherMayExit, AllowPredicates);
      ExitLimit EL1 = computeExitLimitFromCondCached(
          Cache, L, BO->getOperand(1), ExitIfTrue,
          ControlsExit && !EitherMayExit, AllowPredicates);
      const SCEV *BECount = getCouldNotCompute();
      const SCEV *MaxBECount = getCouldNotCompute();
      if (EitherMayExit) {
        // Both conditions must be false for the loop to continue executing.
        // Choose the less conservative count.
        if (EL0.ExactNotTaken == getCouldNotCompute() ||
            EL1.ExactNotTaken == getCouldNotCompute())
          BECount = getCouldNotCompute();
        else
          BECount =
              getUMinFromMismatchedTypes(EL0.ExactNotTaken, EL1.ExactNotTaken);
        if (EL0.MaxNotTaken == getCouldNotCompute())
          MaxBECount = EL1.MaxNotTaken;
        else if (EL1.MaxNotTaken == getCouldNotCompute())
          MaxBECount = EL0.MaxNotTaken;
        else
          MaxBECount =
              getUMinFromMismatchedTypes(EL0.MaxNotTaken, EL1.MaxNotTaken);
      } else {
        // Both conditions must be false at the same time for the loop to exit.
        // For now, be conservative.
        if (EL0.MaxNotTaken == EL1.MaxNotTaken)
          MaxBECount = EL0.MaxNotTaken;
        if (EL0.ExactNotTaken == EL1.ExactNotTaken)
          BECount = EL0.ExactNotTaken;
      }

      return ExitLimit(BECount, MaxBECount, false,
                       {&EL0.Predicates, &EL1.Predicates});
    }
  }

  // With an icmp, it may be feasible to compute an exact backedge-taken count.
  // Proceed to the next level to examine the icmp.
  if (ICmpInst *ExitCondICmp = dyn_cast<ICmpInst>(ExitCond)) {
    ExitLimit EL =
        computeExitLimitFromICmp(L, ExitCondICmp, ExitIfTrue, ControlsExit);
    if (EL.hasFullInfo() || !AllowPredicates)
      return EL;

    // Try again, but use SCEV predicates this time.
    return computeExitLimitFromICmp(L, ExitCondICmp, ExitIfTrue, ControlsExit,
                                    /*AllowPredicates=*/true);
  }

  // Check for a constant condition. These are normally stripped out by
  // SimplifyCFG, but ScalarEvolution may be used by a pass which wishes to
  // preserve the CFG and is temporarily leaving constant conditions
  // in place.
  if (ConstantInt *CI = dyn_cast<ConstantInt>(ExitCond)) {
    if (ExitIfTrue == !CI->getZExtValue())
      // The backedge is always taken.
      return getCouldNotCompute();
    else
      // The backedge is never taken.
      return getZero(CI->getType());
  }

  // If it's not an integer or pointer comparison then compute it the hard way.
  return computeExitCountExhaustively(L, ExitCond, ExitIfTrue);
}
#else
ScalarEvolution::ExitLimit MustExitScalarEvolution::computeExitLimitFromCondImpl(
    ExitLimitCacheTy &Cache, const Loop *L, Value *ExitCond, BasicBlock *TBB,
    BasicBlock *FBB, bool ControlsExit, bool AllowPredicates) {
  // Check if the controlling expression for this loop is an And or Or.
  if (BinaryOperator *BO = dyn_cast<BinaryOperator>(ExitCond)) {
    if (BO->getOpcode() == Instruction::And) {
      // Recurse on the operands of the and.
      bool EitherMayExit = L->contains(TBB);
      ExitLimit EL0 = computeExitLimitFromCondCached(
          Cache, L, BO->getOperand(0), TBB, FBB, ControlsExit && !EitherMayExit,
          AllowPredicates);
      ExitLimit EL1 = computeExitLimitFromCondCached(
          Cache, L, BO->getOperand(1), TBB, FBB, ControlsExit && !EitherMayExit,
          AllowPredicates);
      const SCEV *BECount = getCouldNotCompute();
      const SCEV *MaxBECount = getCouldNotCompute();
      if (EitherMayExit) {
        // Both conditions must be true for the loop to continue executing.
        // Choose the less conservative count.
        if (EL0.ExactNotTaken == getCouldNotCompute() ||
            EL1.ExactNotTaken == getCouldNotCompute())
          BECount = getCouldNotCompute();
        else
          BECount =
              getUMinFromMismatchedTypes(EL0.ExactNotTaken, EL1.ExactNotTaken);
        if (EL0.MaxNotTaken == getCouldNotCompute())
          MaxBECount = EL1.MaxNotTaken;
        else if (EL1.MaxNotTaken == getCouldNotCompute())
          MaxBECount = EL0.MaxNotTaken;
        else
          MaxBECount =
              getUMinFromMismatchedTypes(EL0.MaxNotTaken, EL1.MaxNotTaken);
      } else {
        // Both conditions must be true at the same time for the loop to exit.
        // For now, be conservative.
        assert(L->contains(FBB) && "Loop block has no successor in loop!");
        if (EL0.MaxNotTaken == EL1.MaxNotTaken)
          MaxBECount = EL0.MaxNotTaken;
        if (EL0.ExactNotTaken == EL1.ExactNotTaken)
          BECount = EL0.ExactNotTaken;
      }

      // There are cases (e.g. PR26207) where computeExitLimitFromCond is able
      // to be more aggressive when computing BECount than when computing
      // MaxBECount.  In these cases it is possible for EL0.ExactNotTaken and
      // EL1.ExactNotTaken to match, but for EL0.MaxNotTaken and EL1.MaxNotTaken
      // to not.
      if (isa<SCEVCouldNotCompute>(MaxBECount) &&
          !isa<SCEVCouldNotCompute>(BECount))
        MaxBECount = getConstant(getUnsignedRangeMax(BECount));

      return ExitLimit(BECount, MaxBECount, false,
                       {&EL0.Predicates, &EL1.Predicates});
    }
    if (BO->getOpcode() == Instruction::Or) {
      // Recurse on the operands of the or.
      bool EitherMayExit = L->contains(FBB);
      ExitLimit EL0 = computeExitLimitFromCondCached(
          Cache, L, BO->getOperand(0), TBB, FBB, ControlsExit && !EitherMayExit,
          AllowPredicates);
      ExitLimit EL1 = computeExitLimitFromCondCached(
          Cache, L, BO->getOperand(1), TBB, FBB, ControlsExit && !EitherMayExit,
          AllowPredicates);
      const SCEV *BECount = getCouldNotCompute();
      const SCEV *MaxBECount = getCouldNotCompute();
      if (EitherMayExit) {
        // Both conditions must be false for the loop to continue executing.
        // Choose the less conservative count.
        if (EL0.ExactNotTaken == getCouldNotCompute() ||
            EL1.ExactNotTaken == getCouldNotCompute())
          BECount = getCouldNotCompute();
        else
          BECount =
              getUMinFromMismatchedTypes(EL0.ExactNotTaken, EL1.ExactNotTaken);
        if (EL0.MaxNotTaken == getCouldNotCompute())
          MaxBECount = EL1.MaxNotTaken;
        else if (EL1.MaxNotTaken == getCouldNotCompute())
          MaxBECount = EL0.MaxNotTaken;
        else
          MaxBECount =
              getUMinFromMismatchedTypes(EL0.MaxNotTaken, EL1.MaxNotTaken);
      } else {
        // Both conditions must be false at the same time for the loop to exit.
        // For now, be conservative.
        assert(L->contains(TBB) && "Loop block has no successor in loop!");
        if (EL0.MaxNotTaken == EL1.MaxNotTaken)
          MaxBECount = EL0.MaxNotTaken;
        if (EL0.ExactNotTaken == EL1.ExactNotTaken)
          BECount = EL0.ExactNotTaken;
      }

      return ExitLimit(BECount, MaxBECount, false,
                       {&EL0.Predicates, &EL1.Predicates});
    }
  }

  // With an icmp, it may be feasible to compute an exact backedge-taken count.
  // Proceed to the next level to examine the icmp.
  if (ICmpInst *ExitCondICmp = dyn_cast<ICmpInst>(ExitCond)) {
    ExitLimit EL =
        computeExitLimitFromICmp(L, ExitCondICmp, TBB, FBB, ControlsExit);
    if (EL.hasFullInfo() || !AllowPredicates)
      return EL;

    // Try again, but use SCEV predicates this time.
    return computeExitLimitFromICmp(L, ExitCondICmp, TBB, FBB, ControlsExit,
                                    /*AllowPredicates=*/true);
  }

  // Check for a constant condition. These are normally stripped out by
  // SimplifyCFG, but ScalarEvolution may be used by a pass which wishes to
  // preserve the CFG and is temporarily leaving constant conditions
  // in place.
  if (ConstantInt *CI = dyn_cast<ConstantInt>(ExitCond)) {
    if (L->contains(FBB) == !CI->getZExtValue())
      // The backedge is always taken.
      return getCouldNotCompute();
    else
      // The backedge is never taken.
      return getZero(CI->getType());
  }

  // If it's not an integer or pointer comparison then compute it the hard way.
  return computeExitCountExhaustively(L, ExitCond, !L->contains(TBB));
}
#endif

#if LLVM_VERSION_MAJOR >= 7
ScalarEvolution::ExitLimit
MustExitScalarEvolution::computeExitLimitFromICmp(const Loop *L, ICmpInst *ExitCond,
                                            bool ExitIfTrue, bool ControlsExit,
                                            bool AllowPredicates) {
  // If the condition was exit on true, convert the condition to exit on false
  ICmpInst::Predicate Pred;
  if (!ExitIfTrue)
    Pred = ExitCond->getPredicate();
  else
    Pred = ExitCond->getInversePredicate();
  const ICmpInst::Predicate OriginalPred = Pred;

  // Handle common loops like: for (X = "string"; *X; ++X)
  if (LoadInst *LI = dyn_cast<LoadInst>(ExitCond->getOperand(0)))
    if (Constant *RHS = dyn_cast<Constant>(ExitCond->getOperand(1))) {
      ExitLimit ItCnt = computeLoadConstantCompareExitLimit(LI, RHS, L, Pred);
      if (ItCnt.hasAnyInfo())
        return ItCnt;
    }

  const SCEV *LHS = getSCEV(ExitCond->getOperand(0));
  const SCEV *RHS = getSCEV(ExitCond->getOperand(1));

#define PROP_PHI(LHS)                                                          \
  if (auto un = dyn_cast<SCEVUnknown>(LHS)) {                                  \
    if (auto pn = dyn_cast_or_null<PHINode>(un->getValue())) {                 \
      const SCEV *sc = nullptr;                                                \
      bool failed = false;                                                     \
      for (auto &a : pn->incoming_values()) {                                  \
        auto subsc = getSCEV(a);                                               \
        if (sc == nullptr) {                                                   \
          sc = subsc;                                                          \
          continue;                                                            \
        }                                                                      \
        if (subsc != sc) {                                                     \
          failed = true;                                                       \
          break;                                                               \
        }                                                                      \
      }                                                                        \
      if (!failed) {                                                           \
        LHS = sc;                                                              \
      }                                                                        \
    }                                                                          \
  }
  PROP_PHI(LHS)
  PROP_PHI(RHS)

  // Try to evaluate any dependencies out of the loop.
  LHS = getSCEVAtScope(LHS, L);
  RHS = getSCEVAtScope(RHS, L);

  // At this point, we would like to compute how many iterations of the
  // loop the predicate will return true for these inputs.
  if (isLoopInvariant(LHS, L) && !isLoopInvariant(RHS, L)) {
    // If there is a loop-invariant, force it into the RHS.
    std::swap(LHS, RHS);
    Pred = ICmpInst::getSwappedPredicate(Pred);
  }

  // Simplify the operands before analyzing them.
  (void)SimplifyICmpOperands(Pred, LHS, RHS);

  // If we have a comparison of a chrec against a constant, try to use value
  // ranges to answer this query.
  if (const SCEVConstant *RHSC = dyn_cast<SCEVConstant>(RHS))
    if (const SCEVAddRecExpr *AddRec = dyn_cast<SCEVAddRecExpr>(LHS))
      if (AddRec->getLoop() == L) {
        // Form the constant range.
        ConstantRange CompRange =
            ConstantRange::makeExactICmpRegion(Pred, RHSC->getAPInt());

        const SCEV *Ret = AddRec->getNumIterationsInRange(CompRange, *this);
        if (!isa<SCEVCouldNotCompute>(Ret))
          return Ret;
      }

  switch (Pred) {
  case ICmpInst::ICMP_NE: { // while (X != Y)
    // Convert to: while (X-Y != 0)
    ExitLimit EL =
        howFarToZero(getMinusSCEV(LHS, RHS), L, ControlsExit, AllowPredicates);
    if (EL.hasAnyInfo())
      return EL;
    break;
  }
  case ICmpInst::ICMP_EQ: { // while (X == Y)
    // Convert to: while (X-Y == 0)
    ExitLimit EL = howFarToNonZero(getMinusSCEV(LHS, RHS), L);
    if (EL.hasAnyInfo())
      return EL;
    break;
  }
  case ICmpInst::ICMP_SLT:
  case ICmpInst::ICMP_ULT: { // while (X < Y)
    bool IsSigned = Pred == ICmpInst::ICMP_SLT;
    ExitLimit EL =
        howManyLessThans(LHS, RHS, L, IsSigned, ControlsExit, AllowPredicates);
    if (EL.hasAnyInfo())
      return EL;
    break;
  }
  case ICmpInst::ICMP_SGT:
  case ICmpInst::ICMP_UGT: { // while (X > Y)
    bool IsSigned = Pred == ICmpInst::ICMP_SGT;
    ExitLimit EL = howManyGreaterThans(LHS, RHS, L, IsSigned, ControlsExit,
                                       AllowPredicates);
    if (EL.hasAnyInfo())
      return EL;
    break;
  }
  default:
    break;
  }

  auto *ExhaustiveCount = computeExitCountExhaustively(L, ExitCond, ExitIfTrue);

  if (!isa<SCEVCouldNotCompute>(ExhaustiveCount))
    return ExhaustiveCount;

  return computeShiftCompareExitLimit(ExitCond->getOperand(0),
                                      ExitCond->getOperand(1), L, OriginalPred);
}
#else
ScalarEvolution::ExitLimit
MustExitScalarEvolution::computeExitLimitFromICmp(const Loop *L,
                                          ICmpInst *ExitCond,
                                          BasicBlock *TBB,
                                          BasicBlock *FBB,
                                          bool ControlsExit,
                                          bool AllowPredicates) {
  // If the condition was exit on true, convert the condition to exit on false
  ICmpInst::Predicate Pred;
  if (!L->contains(FBB))
    Pred = ExitCond->getPredicate();
  else
    Pred = ExitCond->getInversePredicate();
  const ICmpInst::Predicate OriginalPred = Pred;

  // Handle common loops like: for (X = "string"; *X; ++X)
  if (LoadInst *LI = dyn_cast<LoadInst>(ExitCond->getOperand(0)))
    if (Constant *RHS = dyn_cast<Constant>(ExitCond->getOperand(1))) {
      ExitLimit ItCnt =
        computeLoadConstantCompareExitLimit(LI, RHS, L, Pred);
      if (ItCnt.hasAnyInfo())
        return ItCnt;
    }

  const SCEV *LHS = getSCEV(ExitCond->getOperand(0));
  const SCEV *RHS = getSCEV(ExitCond->getOperand(1));

  // Try to evaluate any dependencies out of the loop.
  LHS = getSCEVAtScope(LHS, L);
  RHS = getSCEVAtScope(RHS, L);

  // At this point, we would like to compute how many iterations of the
  // loop the predicate will return true for these inputs.
  if (isLoopInvariant(LHS, L) && !isLoopInvariant(RHS, L)) {
    // If there is a loop-invariant, force it into the RHS.
    std::swap(LHS, RHS);
    Pred = ICmpInst::getSwappedPredicate(Pred);
  }

  // Simplify the operands before analyzing them.
  (void)SimplifyICmpOperands(Pred, LHS, RHS);

  // If we have a comparison of a chrec against a constant, try to use value
  // ranges to answer this query.
  if (const SCEVConstant *RHSC = dyn_cast<SCEVConstant>(RHS))
    if (const SCEVAddRecExpr *AddRec = dyn_cast<SCEVAddRecExpr>(LHS))
      if (AddRec->getLoop() == L) {
        // Form the constant range.
        ConstantRange CompRange =
            ConstantRange::makeExactICmpRegion(Pred, RHSC->getAPInt());

        const SCEV *Ret = AddRec->getNumIterationsInRange(CompRange, *this);
        if (!isa<SCEVCouldNotCompute>(Ret)) return Ret;
      }

  switch (Pred) {
  case ICmpInst::ICMP_NE: {                     // while (X != Y)
    // Convert to: while (X-Y != 0)
    ExitLimit EL = howFarToZero(getMinusSCEV(LHS, RHS), L, ControlsExit,
                                AllowPredicates);
    if (EL.hasAnyInfo()) return EL;
    break;
  }
  case ICmpInst::ICMP_EQ: {                     // while (X == Y)
    // Convert to: while (X-Y == 0)
    ExitLimit EL = howFarToNonZero(getMinusSCEV(LHS, RHS), L);
    if (EL.hasAnyInfo()) return EL;
    break;
  }
  case ICmpInst::ICMP_SLT:
  case ICmpInst::ICMP_ULT: {                    // while (X < Y)
    bool IsSigned = Pred == ICmpInst::ICMP_SLT;
    ExitLimit EL = howManyLessThans(LHS, RHS, L, IsSigned, ControlsExit,
                                    AllowPredicates);
    if (EL.hasAnyInfo()) return EL;
    break;
  }
  case ICmpInst::ICMP_SGT:
  case ICmpInst::ICMP_UGT: {                    // while (X > Y)
    bool IsSigned = Pred == ICmpInst::ICMP_SGT;
    ExitLimit EL =
        howManyGreaterThans(LHS, RHS, L, IsSigned, ControlsExit,
                            AllowPredicates);
    if (EL.hasAnyInfo()) return EL;
    break;
  }
  default:
    break;
  }

  auto *ExhaustiveCount =
      computeExitCountExhaustively(L, ExitCond, !L->contains(TBB));

  if (!isa<SCEVCouldNotCompute>(ExhaustiveCount))
    return ExhaustiveCount;

  return computeShiftCompareExitLimit(ExitCond->getOperand(0),
                                      ExitCond->getOperand(1), L, OriginalPred);
}
#endif

ScalarEvolution::ExitLimit
MustExitScalarEvolution::howManyLessThans(const SCEV *LHS, const SCEV *RHS,
                                    const Loop *L, bool IsSigned,
                                    bool ControlsExit, bool AllowPredicates) {
  SmallPtrSet<const SCEVPredicate *, 4> Predicates;

  const SCEVAddRecExpr *IV = dyn_cast<SCEVAddRecExpr>(LHS);

  if (!IV && AllowPredicates) {
    // Try to make this an AddRec using runtime tests, in the first X
    // iterations of this loop, where X is the SCEV expression found by the
    // algorithm below.
    IV = convertSCEVToAddRecWithPredicates(LHS, L, Predicates);
  }

  // Avoid weird loops
  if (!IV || IV->getLoop() != L || !IV->isAffine())
    return getCouldNotCompute();

  bool NoWrap = ControlsExit && true; // changed this to assume no wrap for inc
  //              IV->getNoWrapFlags(IsSigned ? SCEV::FlagNSW : SCEV::FlagNUW);

  const SCEV *Stride = IV->getStepRecurrence(*this);

  bool PositiveStride = isKnownPositive(Stride);

  // Avoid negative or zero stride values.
  if (!PositiveStride) {
    // We can compute the correct backedge taken count for loops with unknown
    // strides if we can prove that the loop is not an infinite loop with side
    // effects. Here's the loop structure we are trying to handle -
    //
    // i = start
    // do {
    //   A[i] = i;
    //   i += s;
    // } while (i < end);
    //
    // The backedge taken count for such loops is evaluated as -
    // (max(end, start + stride) - start - 1) /u stride
    //
    // The additional preconditions that we need to check to prove correctness
    // of the above formula is as follows -
    //
    // a) IV is either nuw or nsw depending upon signedness (indicated by the
    //    NoWrap flag).
    // b) loop is single exit with no side effects. // dont need this
    //
    //
    // Precondition a) implies that if the stride is negative, this is a single
    // trip loop. The backedge taken count formula reduces to zero in this case.
    //
    // Precondition b) implies that the unknown stride cannot be zero otherwise
    // we have UB.
    //
    // The positive stride case is the same as isKnownPositive(Stride) returning
    // true (original behavior of the function).
    //
    // We want to make sure that the stride is truly unknown as there are edge
    // cases where ScalarEvolution propagates no wrap flags to the
    // post-increment/decrement IV even though the increment/decrement operation
    // itself is wrapping. The computed backedge taken count may be wrong in
    // such cases. This is prevented by checking that the stride is not known to
    // be either positive or non-positive. For example, no wrap flags are
    // propagated to the post-increment IV of this loop with a trip count of 2 -
    //
    // unsigned char i;
    // for(i=127; i<128; i+=129)
    //   A[i] = i;
    //
    if (!NoWrap) // THIS LINE CHANGED
      return getCouldNotCompute();
  } else if (!Stride->isOne() &&
             doesIVOverflowOnLT(RHS, Stride, IsSigned, NoWrap))
    // Avoid proven overflow cases: this will ensure that the backedge taken
    // count will not generate any unsigned overflow. Relaxed no-overflow
    // conditions exploit NoWrapFlags, allowing to optimize in presence of
    // undefined behaviors like the case of C language.
    return getCouldNotCompute();

  ICmpInst::Predicate Cond = IsSigned ? ICmpInst::ICMP_SLT : ICmpInst::ICMP_ULT;
  const SCEV *Start = IV->getStart();
  const SCEV *End = RHS;
  // When the RHS is not invariant, we do not know the end bound of the loop and
  // cannot calculate the ExactBECount needed by ExitLimit. However, we can
  // calculate the MaxBECount, given the start, stride and max value for the end
  // bound of the loop (RHS), and the fact that IV does not overflow (which is
  // checked above).
  if (!isLoopInvariant(RHS, L)) {
    const SCEV *MaxBECount = computeMaxBECountForLT(
        Start, Stride, RHS, getTypeSizeInBits(LHS->getType()), IsSigned);
    return ExitLimit(getCouldNotCompute() /* ExactNotTaken */, MaxBECount,
                     false /*MaxOrZero*/, Predicates);
  }
  // If the backedge is taken at least once, then it will be taken
  // (End-Start)/Stride times (rounded up to a multiple of Stride), where Start
  // is the LHS value of the less-than comparison the first time it is evaluated
  // and End is the RHS.
  const SCEV *BECountIfBackedgeTaken =
      computeBECount(getMinusSCEV(End, Start), Stride, false);
  // If the loop entry is guarded by the result of the backedge test of the
  // first loop iteration, then we know the backedge will be taken at least
  // once and so the backedge taken count is as above. If not then we use the
  // expression (max(End,Start)-Start)/Stride to describe the backedge count,
  // as if the backedge is taken at least once max(End,Start) is End and so the
  // result is as above, and if not max(End,Start) is Start so we get a backedge
  // count of zero.
  const SCEV *BECount;
  if (isLoopEntryGuardedByCond(L, Cond, getMinusSCEV(Start, Stride), RHS))
    BECount = BECountIfBackedgeTaken;
  else {
    End = IsSigned ? getSMaxExpr(RHS, Start) : getUMaxExpr(RHS, Start);
    BECount = computeBECount(getMinusSCEV(End, Start), Stride, false);
  }

  const SCEV *MaxBECount;
  bool MaxOrZero = false;
  if (isa<SCEVConstant>(BECount))
    MaxBECount = BECount;
  else if (isa<SCEVConstant>(BECountIfBackedgeTaken)) {
    // If we know exactly how many times the backedge will be taken if it's
    // taken at least once, then the backedge count will either be that or
    // zero.
    MaxBECount = BECountIfBackedgeTaken;
    MaxOrZero = true;
  } else {
    MaxBECount = computeMaxBECountForLT(
        Start, Stride, RHS, getTypeSizeInBits(LHS->getType()), IsSigned);
  }

  if (isa<SCEVCouldNotCompute>(MaxBECount) &&
      !isa<SCEVCouldNotCompute>(BECount))
    MaxBECount = getConstant(getUnsignedRangeMax(BECount));

  return ExitLimit(BECount, MaxBECount, MaxOrZero, Predicates);
}