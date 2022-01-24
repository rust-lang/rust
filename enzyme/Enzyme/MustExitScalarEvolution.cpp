
//===- MustExitScalarEvolution.cpp - ScalarEvolution assuming loops
// terminate-=//
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
// This file defines MustExitScalarEvolution, a subclass of ScalarEvolution
// that assumes that all loops terminate (and don't loop forever).
//
//===----------------------------------------------------------------------===//

#include "MustExitScalarEvolution.h"
#include "FunctionUtils.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"

using namespace llvm;

bool MustExitScalarEvolution::loopIsFiniteByAssumption(const Loop *L) {
  return true;
}

#if LLVM_VERSION_MAJOR >= 7
ScalarEvolution::ExitLimit MustExitScalarEvolution::computeExitLimitFromCond(
    const Loop *L, Value *ExitCond, bool ExitIfTrue, bool ControlsExit,
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

MustExitScalarEvolution::MustExitScalarEvolution(llvm::Function &F,
                                                 llvm::TargetLibraryInfo &TLI,
                                                 llvm::AssumptionCache &AC,
                                                 llvm::DominatorTree &DT,
                                                 llvm::LoopInfo &LI)
    : ScalarEvolution(F, TLI, AC, DT, LI),
      GuaranteedUnreachable(getGuaranteedUnreachable(&F)) {}

ScalarEvolution::ExitLimit MustExitScalarEvolution::computeExitLimit(
    const Loop *L, BasicBlock *ExitingBlock, bool AllowPredicates) {

  SmallVector<BasicBlock *, 8> ExitingBlocks;
  L->getExitingBlocks(ExitingBlocks);
  for (auto &ExitingBlock : ExitingBlocks) {
    BasicBlock *Exit = nullptr;
    for (auto *SBB : successors(ExitingBlock)) {
      if (!L->contains(SBB)) {
        if (GuaranteedUnreachable.count(SBB))
          continue;
        Exit = SBB;
        break;
      }
    }
    if (!Exit)
      ExitingBlock = nullptr;
  }
  ExitingBlocks.erase(
      std::remove(ExitingBlocks.begin(), ExitingBlocks.end(), nullptr),
      ExitingBlocks.end());

  assert(L->contains(ExitingBlock) && "Exit count for non-loop block?");
  // If our exiting block does not dominate the latch, then its connection with
  // loop's exit limit may be far from trivial.
  const BasicBlock *Latch = L->getLoopLatch();
  if (!Latch || !DT.dominates(ExitingBlock, Latch))
    return getCouldNotCompute();

  bool IsOnlyExit = ExitingBlocks.size() == 1;
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
        if (GuaranteedUnreachable.count(SBB))
          continue;
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
ScalarEvolution::ExitLimit
MustExitScalarEvolution::computeExitLimitFromCondCached(
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
ScalarEvolution::ExitLimit
MustExitScalarEvolution::computeExitLimitFromCondCached(
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
ScalarEvolution::ExitLimit
MustExitScalarEvolution::computeExitLimitFromCondImpl(
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
ScalarEvolution::ExitLimit
MustExitScalarEvolution::computeExitLimitFromCondImpl(
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
ScalarEvolution::ExitLimit MustExitScalarEvolution::computeExitLimitFromICmp(
    const Loop *L, ICmpInst *ExitCond, bool ExitIfTrue, bool ControlsExit,
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
  case ICmpInst::ICMP_ULT:
  case ICmpInst::ICMP_SLE:
  case ICmpInst::ICMP_ULE: { // while (X < Y)
    bool IsSigned = Pred == ICmpInst::ICMP_SLT || Pred == ICmpInst::ICMP_SLE;

    if (Pred == ICmpInst::ICMP_SLE || Pred == ICmpInst::ICMP_ULE) {
      SmallVector<const SCEV *, 2> sv = {
          RHS,
          getConstant(ConstantInt::get(cast<IntegerType>(RHS->getType()), 1))};
      // Since this is not an infinite loop by induction, RHS cannot be
      // int_max/uint_max Therefore adding 1 does not wrap.
      if (IsSigned)
        RHS = getAddExpr(sv, SCEV::FlagNSW);
      else
        RHS = getAddExpr(sv, SCEV::FlagNUW);
    }
    ExitLimit EL =
        howManyLessThans(LHS, RHS, L, IsSigned, ControlsExit, AllowPredicates);
    if (EL.hasAnyInfo())
      return EL;
    break;
  }
  case ICmpInst::ICMP_SGT:
  case ICmpInst::ICMP_UGT:
  case ICmpInst::ICMP_SGE:
  case ICmpInst::ICMP_UGE: { // while (X > Y)
    bool IsSigned = Pred == ICmpInst::ICMP_SGT || Pred == ICmpInst::ICMP_SLE;
    if (Pred == ICmpInst::ICMP_SGE || Pred == ICmpInst::ICMP_UGE) {
      SmallVector<const SCEV *, 2> sv = {
          RHS,
          getConstant(ConstantInt::get(cast<IntegerType>(RHS->getType()), -1))};
      // Since this is not an infinite loop by induction, RHS cannot be
      // int_min/uint_min Therefore subtracting 1 does not wrap.
      if (IsSigned)
        RHS = getAddExpr(sv, SCEV::FlagNSW);
      else
        RHS = getAddExpr(sv, SCEV::FlagNUW);
    }
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
ScalarEvolution::ExitLimit MustExitScalarEvolution::computeExitLimitFromICmp(
    const Loop *L, ICmpInst *ExitCond, BasicBlock *TBB, BasicBlock *FBB,
    bool ControlsExit, bool AllowPredicates) {
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
      ExitLimit ItCnt = computeLoadConstantCompareExitLimit(LI, RHS, L, Pred);
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
  case ICmpInst::ICMP_ULT:
  case ICmpInst::ICMP_SLE:
  case ICmpInst::ICMP_ULE: { // while (X < Y)
    bool IsSigned = Pred == ICmpInst::ICMP_SLT || Pred == ICmpInst::ICMP_SLE;

    if (Pred == ICmpInst::ICMP_SLE || Pred == ICmpInst::ICMP_ULE) {
      SmallVector<const SCEV *, 2> sv = {
          RHS,
          getConstant(ConstantInt::get(cast<IntegerType>(RHS->getType()), 1))};
      // Since this is not an infinite loop by induction, RHS cannot be
      // int_max/uint_max Therefore adding 1 does not wrap.
      if (IsSigned)
        RHS = getAddExpr(sv, SCEV::FlagNSW);
      else
        RHS = getAddExpr(sv, SCEV::FlagNUW);
    }
    ExitLimit EL =
        howManyLessThans(LHS, RHS, L, IsSigned, ControlsExit, AllowPredicates);
    if (EL.hasAnyInfo())
      return EL;
    break;
  }
  case ICmpInst::ICMP_SGT:
  case ICmpInst::ICMP_UGT:
  case ICmpInst::ICMP_SGE:
  case ICmpInst::ICMP_UGE: { // while (X > Y)
    bool IsSigned = Pred == ICmpInst::ICMP_SGT || Pred == ICmpInst::ICMP_SLE;
    if (Pred == ICmpInst::ICMP_SGE || Pred == ICmpInst::ICMP_UGE) {
      SmallVector<const SCEV *, 2> sv = {
          RHS,
          getConstant(ConstantInt::get(cast<IntegerType>(RHS->getType()), -1))};
      // Since this is not an infinite loop by induction, RHS cannot be
      // int_min/uint_min Therefore subtracting 1 does not wrap.
      if (IsSigned)
        RHS = getAddExpr(sv, SCEV::FlagNSW);
      else
        RHS = getAddExpr(sv, SCEV::FlagNUW);
    }
    ExitLimit EL = howManyGreaterThans(LHS, RHS, L, IsSigned, ControlsExit,
                                       AllowPredicates);
    if (EL.hasAnyInfo())
      return EL;
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

#if LLVM_VERSION_MAJOR >= 13
static const SCEV *getSignedOverflowLimitForStep(const SCEV *Step,
                                                 ICmpInst::Predicate *Pred,
                                                 ScalarEvolution *SE) {
  unsigned BitWidth = SE->getTypeSizeInBits(Step->getType());
  if (SE->isKnownPositive(Step)) {
    *Pred = ICmpInst::ICMP_SLT;
    return SE->getConstant(APInt::getSignedMinValue(BitWidth) -
                           SE->getSignedRangeMax(Step));
  }
  if (SE->isKnownNegative(Step)) {
    *Pred = ICmpInst::ICMP_SGT;
    return SE->getConstant(APInt::getSignedMaxValue(BitWidth) -
                           SE->getSignedRangeMin(Step));
  }
  return nullptr;
}
static const SCEV *getUnsignedOverflowLimitForStep(const SCEV *Step,
                                                   ICmpInst::Predicate *Pred,
                                                   ScalarEvolution *SE) {
  unsigned BitWidth = SE->getTypeSizeInBits(Step->getType());
  *Pred = ICmpInst::ICMP_ULT;

  return SE->getConstant(APInt::getMinValue(BitWidth) -
                         SE->getUnsignedRangeMax(Step));
}

namespace {

struct ExtendOpTraitsBase {
  typedef const SCEV *(ScalarEvolution::*GetExtendExprTy)(const SCEV *, Type *,
                                                          unsigned);
};

// Used to make code generic over signed and unsigned overflow.
template <typename ExtendOp> struct ExtendOpTraits {
  // Members present:
  //
  // static const SCEV::NoWrapFlags WrapType;
  //
  // static const ExtendOpTraitsBase::GetExtendExprTy GetExtendExpr;
  //
  // static const SCEV *getOverflowLimitForStep(const SCEV *Step,
  //                                           ICmpInst::Predicate *Pred,
  //                                           ScalarEvolution *SE);
};

template <>
struct ExtendOpTraits<SCEVSignExtendExpr> : public ExtendOpTraitsBase {
  static const SCEV::NoWrapFlags WrapType = SCEV::FlagNSW;

  static const GetExtendExprTy GetExtendExpr;

  static const SCEV *getOverflowLimitForStep(const SCEV *Step,
                                             ICmpInst::Predicate *Pred,
                                             ScalarEvolution *SE) {
    return getSignedOverflowLimitForStep(Step, Pred, SE);
  }
};

const ExtendOpTraitsBase::GetExtendExprTy
    ExtendOpTraits<SCEVSignExtendExpr>::GetExtendExpr =
        &ScalarEvolution::getSignExtendExpr;

template <>
struct ExtendOpTraits<SCEVZeroExtendExpr> : public ExtendOpTraitsBase {
  static const SCEV::NoWrapFlags WrapType = SCEV::FlagNUW;

  static const GetExtendExprTy GetExtendExpr;

  static const SCEV *getOverflowLimitForStep(const SCEV *Step,
                                             ICmpInst::Predicate *Pred,
                                             ScalarEvolution *SE) {
    return getUnsignedOverflowLimitForStep(Step, Pred, SE);
  }
};

const ExtendOpTraitsBase::GetExtendExprTy
    ExtendOpTraits<SCEVZeroExtendExpr>::GetExtendExpr =
        &ScalarEvolution::getZeroExtendExpr;

} // end anonymous namespace

static bool hasFlags(SCEV::NoWrapFlags Flags, SCEV::NoWrapFlags TestFlags) {
  return TestFlags == ScalarEvolution::maskFlags(Flags, TestFlags);
};

template <typename ExtendOpTy>
static const SCEV *getPreStartForExtend(const SCEVAddRecExpr *AR, Type *Ty,
                                        ScalarEvolution *SE, unsigned Depth) {
  auto WrapType = ExtendOpTraits<ExtendOpTy>::WrapType;
  auto GetExtendExpr = ExtendOpTraits<ExtendOpTy>::GetExtendExpr;

  const Loop *L = AR->getLoop();
  const SCEV *Start = AR->getStart();
  const SCEV *Step = AR->getStepRecurrence(*SE);

  // Check for a simple looking step prior to loop entry.
  const SCEVAddExpr *SA = dyn_cast<SCEVAddExpr>(Start);
  if (!SA)
    return nullptr;

  // Create an AddExpr for "PreStart" after subtracting Step. Full SCEV
  // subtraction is expensive. For this purpose, perform a quick and dirty
  // difference, by checking for Step in the operand list.
  SmallVector<const SCEV *, 4> DiffOps;
  for (const SCEV *Op : SA->operands())
    if (Op != Step)
      DiffOps.push_back(Op);

  if (DiffOps.size() == SA->getNumOperands())
    return nullptr;

  // Try to prove `WrapType` (SCEV::FlagNSW or SCEV::FlagNUW) on `PreStart` +
  // `Step`:

  // 1. NSW/NUW flags on the step increment.
  auto PreStartFlags =
      ScalarEvolution::maskFlags(SA->getNoWrapFlags(), SCEV::FlagNUW);
  const SCEV *PreStart = SE->getAddExpr(DiffOps, PreStartFlags);
  const SCEVAddRecExpr *PreAR = dyn_cast<SCEVAddRecExpr>(
      SE->getAddRecExpr(PreStart, Step, L, SCEV::FlagAnyWrap));

  // "{S,+,X} is <nsw>/<nuw>" and "the backedge is taken at least once" implies
  // "S+X does not sign/unsign-overflow".
  //

  const SCEV *BECount = SE->getBackedgeTakenCount(L);
  if (PreAR && PreAR->getNoWrapFlags(WrapType) &&
      !isa<SCEVCouldNotCompute>(BECount) && SE->isKnownPositive(BECount))
    return PreStart;

  // 2. Direct overflow check on the step operation's expression.
  unsigned BitWidth = SE->getTypeSizeInBits(AR->getType());
  Type *WideTy = IntegerType::get(SE->getContext(), BitWidth * 2);
  const SCEV *OperandExtendedStart =
      SE->getAddExpr((SE->*GetExtendExpr)(PreStart, WideTy, Depth),
                     (SE->*GetExtendExpr)(Step, WideTy, Depth));
  if ((SE->*GetExtendExpr)(Start, WideTy, Depth) == OperandExtendedStart) {
    if (PreAR && AR->getNoWrapFlags(WrapType)) {
      // If we know `AR` == {`PreStart`+`Step`,+,`Step`} is `WrapType` (FlagNSW
      // or FlagNUW) and that `PreStart` + `Step` is `WrapType` too, then
      // `PreAR` == {`PreStart`,+,`Step`} is also `WrapType`.  Cache this fact.
      SE->setNoWrapFlags(const_cast<SCEVAddRecExpr *>(PreAR), WrapType);
    }
    return PreStart;
  }

  // 3. Loop precondition.
  ICmpInst::Predicate Pred;
  const SCEV *OverflowLimit =
      ExtendOpTraits<ExtendOpTy>::getOverflowLimitForStep(Step, &Pred, SE);

  if (OverflowLimit &&
      SE->isLoopEntryGuardedByCond(L, Pred, PreStart, OverflowLimit))
    return PreStart;

  return nullptr;
}

template <typename ExtendOpTy>
static const SCEV *getExtendAddRecStart(const SCEVAddRecExpr *AR, Type *Ty,
                                        ScalarEvolution *SE, unsigned Depth) {
  auto GetExtendExpr = ExtendOpTraits<ExtendOpTy>::GetExtendExpr;

  const SCEV *PreStart = getPreStartForExtend<ExtendOpTy>(AR, Ty, SE, Depth);
  if (!PreStart)
    return (SE->*GetExtendExpr)(AR->getStart(), Ty, Depth);

  return SE->getAddExpr(
      (SE->*GetExtendExpr)(AR->getStepRecurrence(*SE), Ty, Depth),
      (SE->*GetExtendExpr)(PreStart, Ty, Depth));
}

static SCEV::NoWrapFlags StrengthenNoWrapFlags(ScalarEvolution *SE,
                                               SCEVTypes Type,
                                               const ArrayRef<const SCEV *> Ops,
                                               SCEV::NoWrapFlags Flags) {
  using namespace std::placeholders;

  using OBO = OverflowingBinaryOperator;

  bool CanAnalyze =
      Type == scAddExpr || Type == scAddRecExpr || Type == scMulExpr;
  (void)CanAnalyze;
  assert(CanAnalyze && "don't call from other places!");

  int SignOrUnsignMask = SCEV::FlagNUW | SCEV::FlagNSW;
  SCEV::NoWrapFlags SignOrUnsignWrap =
      ScalarEvolution::maskFlags(Flags, SignOrUnsignMask);

  // If FlagNSW is true and all the operands are non-negative, infer FlagNUW.
  auto IsKnownNonNegative = [&](const SCEV *S) {
    return SE->isKnownNonNegative(S);
  };

  if (SignOrUnsignWrap == SCEV::FlagNSW && all_of(Ops, IsKnownNonNegative))
    Flags =
        ScalarEvolution::setFlags(Flags, (SCEV::NoWrapFlags)SignOrUnsignMask);

  SignOrUnsignWrap = ScalarEvolution::maskFlags(Flags, SignOrUnsignMask);

  if (SignOrUnsignWrap != SignOrUnsignMask &&
      (Type == scAddExpr || Type == scMulExpr) && Ops.size() == 2 &&
      isa<SCEVConstant>(Ops[0])) {

    auto Opcode = [&] {
      switch (Type) {
      case scAddExpr:
        return Instruction::Add;
      case scMulExpr:
        return Instruction::Mul;
      default:
        llvm_unreachable("Unexpected SCEV op.");
      }
    }();

    const APInt &C = cast<SCEVConstant>(Ops[0])->getAPInt();

    // (A <opcode> C) --> (A <opcode> C)<nsw> if the op doesn't sign overflow.
    if (!(SignOrUnsignWrap & SCEV::FlagNSW)) {
      auto NSWRegion = ConstantRange::makeGuaranteedNoWrapRegion(
          Opcode, C, OBO::NoSignedWrap);
      if (NSWRegion.contains(SE->getSignedRange(Ops[1])))
        Flags = ScalarEvolution::setFlags(Flags, SCEV::FlagNSW);
    }

    // (A <opcode> C) --> (A <opcode> C)<nuw> if the op doesn't unsign overflow.
    if (!(SignOrUnsignWrap & SCEV::FlagNUW)) {
      auto NUWRegion = ConstantRange::makeGuaranteedNoWrapRegion(
          Opcode, C, OBO::NoUnsignedWrap);
      if (NUWRegion.contains(SE->getUnsignedRange(Ops[1])))
        Flags = ScalarEvolution::setFlags(Flags, SCEV::FlagNUW);
    }
  }

  // <0,+,nonnegative><nw> is also nuw
  // TODO: Add corresponding nsw case
  if (Type == scAddRecExpr && hasFlags(Flags, SCEV::FlagNW) &&
      !hasFlags(Flags, SCEV::FlagNUW) && Ops.size() == 2 && Ops[0]->isZero() &&
      IsKnownNonNegative(Ops[1]))
    Flags = ScalarEvolution::setFlags(Flags, SCEV::FlagNUW);

  // both (udiv X, Y) * Y and Y * (udiv X, Y) are always NUW
  if (Type == scMulExpr && !hasFlags(Flags, SCEV::FlagNUW) && Ops.size() == 2) {
    if (auto *UDiv = dyn_cast<SCEVUDivExpr>(Ops[0]))
      if (UDiv->getOperand(1) == Ops[1])
        Flags = ScalarEvolution::setFlags(Flags, SCEV::FlagNUW);
    if (auto *UDiv = dyn_cast<SCEVUDivExpr>(Ops[1]))
      if (UDiv->getOperand(1) == Ops[0])
        Flags = ScalarEvolution::setFlags(Flags, SCEV::FlagNUW);
  }

  return Flags;
}

ScalarEvolution::ExitLimit MustExitScalarEvolution::howManyLessThans(
    const SCEV *LHS, const SCEV *RHS, const Loop *L, bool IsSigned,
    bool ControlsExit, bool AllowPredicates) {
  SmallPtrSet<const SCEVPredicate *, 4> Predicates;

  const SCEVAddRecExpr *IV = dyn_cast<SCEVAddRecExpr>(LHS);
  bool PredicatedIV = false;

  auto canAssumeNoSelfWrap = [&](const SCEVAddRecExpr *AR) {
    // Can we prove this loop *must* be UB if overflow of IV occurs?
    // Reasoning goes as follows:
    // * Suppose the IV did self wrap.
    // * If Stride evenly divides the iteration space, then once wrap
    //   occurs, the loop must revisit the same values.
    // * We know that RHS is invariant, and that none of those values
    //   caused this exit to be taken previously.  Thus, this exit is
    //   dynamically dead.
    // * If this is the sole exit, then a dead exit implies the loop
    //   must be infinite if there are no abnormal exits.
    // * If the loop were infinite, then it must either not be mustprogress
    //   or have side effects. Otherwise, it must be UB.
    // * It can't (by assumption), be UB so we have contradicted our
    //   premise and can conclude the IV did not in fact self-wrap.
    if (!isLoopInvariant(RHS, L))
      return false;

    auto *StrideC = dyn_cast<SCEVConstant>(AR->getStepRecurrence(*this));
    if (!StrideC || !StrideC->getAPInt().isPowerOf2())
      return false;

    if (!ControlsExit || !loopHasNoAbnormalExits(L))
      return false;

    return loopIsFiniteByAssumption(L);
  };

  if (!IV) {
    if (auto *ZExt = dyn_cast<SCEVZeroExtendExpr>(LHS)) {
      const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(ZExt->getOperand());
      if (AR && AR->getLoop() == L && AR->isAffine()) {
        auto Flags = AR->getNoWrapFlags();
        if (!hasFlags(Flags, SCEV::FlagNW) && canAssumeNoSelfWrap(AR)) {
          Flags = setFlags(Flags, SCEV::FlagNW);

          SmallVector<const SCEV *> Operands{AR->operands()};
          Flags = StrengthenNoWrapFlags(this, scAddRecExpr, Operands, Flags);

          setNoWrapFlags(const_cast<SCEVAddRecExpr *>(AR), Flags);
        }
        if (AR->hasNoUnsignedWrap()) {
          // Emulate what getZeroExtendExpr would have done during construction
          // if we'd been able to infer the fact just above at that time.
          const SCEV *Step = AR->getStepRecurrence(*this);
          Type *Ty = ZExt->getType();
          auto *S = getAddRecExpr(
              getExtendAddRecStart<SCEVZeroExtendExpr>(AR, Ty, this, 0),
              getZeroExtendExpr(Step, Ty, 0), L, AR->getNoWrapFlags());
          IV = dyn_cast<SCEVAddRecExpr>(S);
        }
      }
    }
  }

  if (!IV && AllowPredicates) {
    // Try to make this an AddRec using runtime tests, in the first X
    // iterations of this loop, where X is the SCEV expression found by the
    // algorithm below.
    IV = convertSCEVToAddRecWithPredicates(LHS, L, Predicates);
    PredicatedIV = true;
  }

  // Avoid weird loops
  if (!IV || IV->getLoop() != L || !IV->isAffine())
    return getCouldNotCompute();

  // A precondition of this method is that the condition being analyzed
  // reaches an exiting branch which dominates the latch.  Given that, we can
  // assume that an increment which violates the nowrap specification and
  // produces poison must cause undefined behavior when the resulting poison
  // value is branched upon and thus we can conclude that the backedge is
  // taken no more often than would be required to produce that poison value.
  // Note that a well defined loop can exit on the iteration which violates
  // the nowrap specification if there is another exit (either explicit or
  // implicit/exceptional) which causes the loop to execute before the
  // exiting instruction we're analyzing would trigger UB.
  auto WrapType = IsSigned ? SCEV::FlagNSW : SCEV::FlagNUW;
  bool NoWrap = ControlsExit && IV->getNoWrapFlags(WrapType);
  ICmpInst::Predicate Cond = IsSigned ? ICmpInst::ICMP_SLT : ICmpInst::ICMP_ULT;

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
    // b) the loop is guaranteed to be finite (e.g. is mustprogress and has
    //    no side effects within the loop)
    // c) loop has a single static exit (with no abnormal exits)
    //
    // Precondition a) implies that if the stride is negative, this is a single
    // trip loop. The backedge taken count formula reduces to zero in this case.
    //
    // Precondition b) and c) combine to imply that if rhs is invariant in L,
    // then a zero stride means the backedge can't be taken without executing
    // undefined behavior.
    //
    // The positive stride case is the same as isKnownPositive(Stride) returning
    // true (original behavior of the function).
    //
    if (PredicatedIV || !NoWrap || !loopIsFiniteByAssumption(L) ||
        !loopHasNoAbnormalExits(L)) {
      return getCouldNotCompute();
    }

    // This bailout is protecting the logic in computeMaxBECountForLT which
    // has not yet been sufficiently auditted or tested with negative strides.
    // We used to filter out all known-non-positive cases here, we're in the
    // process of being less restrictive bit by bit.
    if (IsSigned && isKnownNonPositive(Stride))
      return getCouldNotCompute();

    if (!isKnownNonZero(Stride)) {
      // If we have a step of zero, and RHS isn't invariant in L, we don't know
      // if it might eventually be greater than start and if so, on which
      // iteration.  We can't even produce a useful upper bound.
      if (!isLoopInvariant(RHS, L))
        return getCouldNotCompute();

      // We allow a potentially zero stride, but we need to divide by stride
      // below.  Since the loop can't be infinite and this check must control
      // the sole exit, we can infer the exit must be taken on the first
      // iteration (e.g. backedge count = 0) if the stride is zero.  Given that,
      // we know the numerator in the divides below must be zero, so we can
      // pick an arbitrary non-zero value for the denominator (e.g. stride)
      // and produce the right result.
      // FIXME: Handle the case where Stride is poison?
      auto wouldZeroStrideBeUB = [&]() {
        // Proof by contradiction.  Suppose the stride were zero.  If we can
        // prove that the backedge *is* taken on the first iteration, then since
        // we know this condition controls the sole exit, we must have an
        // infinite loop.  We can't have a (well defined) infinite loop per
        // check just above.
        // Note: The (Start - Stride) term is used to get the start' term from
        // (start' + stride,+,stride). Remember that we only care about the
        // result of this expression when stride == 0 at runtime.
        auto *StartIfZero = getMinusSCEV(IV->getStart(), Stride);
        return isLoopEntryGuardedByCond(L, Cond, StartIfZero, RHS);
      };
      if (!wouldZeroStrideBeUB()) {
        Stride = getUMaxExpr(Stride, getOne(Stride->getType()));
      }
    }
  } else if (!Stride->isOne() && !NoWrap) {
    auto isUBOnWrap = [&]() {
      // From no-self-wrap, we need to then prove no-(un)signed-wrap.  This
      // follows trivially from the fact that every (un)signed-wrapped, but
      // not self-wrapped value must be LT than the last value before
      // (un)signed wrap.  Since we know that last value didn't exit, nor
      // will any smaller one.
      return canAssumeNoSelfWrap(IV);
    };

    // Avoid proven overflow cases: this will ensure that the backedge taken
    // count will not generate any unsigned overflow. Relaxed no-overflow
    // conditions exploit NoWrapFlags, allowing to optimize in presence of
    // undefined behaviors like the case of C language.
    if (canIVOverflowOnLT(RHS, Stride, IsSigned) && !isUBOnWrap())
      return getCouldNotCompute();
  }

  // On all paths just preceeding, we established the following invariant:
  //   IV can be assumed not to overflow up to and including the exiting
  //   iteration.  We proved this in one of two ways:
  //   1) We can show overflow doesn't occur before the exiting iteration
  //      1a) canIVOverflowOnLT, and b) step of one
  //   2) We can show that if overflow occurs, the loop must execute UB
  //      before any possible exit.
  // Note that we have not yet proved RHS invariant (in general).

  const SCEV *Start = IV->getStart();

  // Preserve pointer-typed Start/RHS to pass to isLoopEntryGuardedByCond.
  // If we convert to integers, isLoopEntryGuardedByCond will miss some cases.
  // Use integer-typed versions for actual computation; we can't subtract
  // pointers in general.
  const SCEV *OrigStart = Start;
  const SCEV *OrigRHS = RHS;
  if (Start->getType()->isPointerTy()) {
    Start = getLosslessPtrToIntExpr(Start);
    if (isa<SCEVCouldNotCompute>(Start))
      return Start;
  }
  if (RHS->getType()->isPointerTy()) {
    RHS = getLosslessPtrToIntExpr(RHS);
    if (isa<SCEVCouldNotCompute>(RHS))
      return RHS;
  }

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

  // We use the expression (max(End,Start)-Start)/Stride to describe the
  // backedge count, as if the backedge is taken at least once max(End,Start)
  // is End and so the result is as above, and if not max(End,Start) is Start
  // so we get a backedge count of zero.
  const SCEV *BECount = nullptr;
  auto *OrigStartMinusStride = getMinusSCEV(OrigStart, Stride);
  assert(isAvailableAtLoopEntry(OrigStartMinusStride, L) && "Must be!");
  assert(isAvailableAtLoopEntry(OrigStart, L) && "Must be!");
  assert(isAvailableAtLoopEntry(OrigRHS, L) && "Must be!");
  // Can we prove (max(RHS,Start) > Start - Stride?
  if (isLoopEntryGuardedByCond(L, Cond, OrigStartMinusStride, OrigStart) &&
      isLoopEntryGuardedByCond(L, Cond, OrigStartMinusStride, OrigRHS)) {
    // In this case, we can use a refined formula for computing backedge taken
    // count.  The general formula remains:
    //   "End-Start /uceiling Stride" where "End = max(RHS,Start)"
    // We want to use the alternate formula:
    //   "((End - 1) - (Start - Stride)) /u Stride"
    // Let's do a quick case analysis to show these are equivalent under
    // our precondition that max(RHS,Start) > Start - Stride.
    // * For RHS <= Start, the backedge-taken count must be zero.
    //   "((End - 1) - (Start - Stride)) /u Stride" reduces to
    //   "((Start - 1) - (Start - Stride)) /u Stride" which simplies to
    //   "Stride - 1 /u Stride" which is indeed zero for all non-zero values
    //     of Stride.  For 0 stride, we've use umin(1,Stride) above, reducing
    //     this to the stride of 1 case.
    // * For RHS >= Start, the backedge count must be "RHS-Start /uceil Stride".
    //   "((End - 1) - (Start - Stride)) /u Stride" reduces to
    //   "((RHS - 1) - (Start - Stride)) /u Stride" reassociates to
    //   "((RHS - (Start - Stride) - 1) /u Stride".
    //   Our preconditions trivially imply no overflow in that form.
    const SCEV *MinusOne = getMinusOne(Stride->getType());
    const SCEV *Numerator =
        getMinusSCEV(getAddExpr(RHS, MinusOne), getMinusSCEV(Start, Stride));
    BECount = getUDivExpr(Numerator, Stride);
  }

  const SCEV *BECountIfBackedgeTaken = nullptr;
  if (!BECount) {
    auto canProveRHSGreaterThanEqualStart = [&]() {
      auto CondGE = IsSigned ? ICmpInst::ICMP_SGE : ICmpInst::ICMP_UGE;
      if (isLoopEntryGuardedByCond(L, CondGE, OrigRHS, OrigStart))
        return true;

      // (RHS > Start - 1) implies RHS >= Start.
      // * "RHS >= Start" is trivially equivalent to "RHS > Start - 1" if
      //   "Start - 1" doesn't overflow.
      // * For signed comparison, if Start - 1 does overflow, it's equal
      //   to INT_MAX, and "RHS >s INT_MAX" is trivially false.
      // * For unsigned comparison, if Start - 1 does overflow, it's equal
      //   to UINT_MAX, and "RHS >u UINT_MAX" is trivially false.
      //
      // FIXME: Should isLoopEntryGuardedByCond do this for us?
      auto CondGT = IsSigned ? ICmpInst::ICMP_SGT : ICmpInst::ICMP_UGT;
      auto *StartMinusOne =
          getAddExpr(OrigStart, getMinusOne(OrigStart->getType()));
      return isLoopEntryGuardedByCond(L, CondGT, OrigRHS, StartMinusOne);
    };

    // If we know that RHS >= Start in the context of loop, then we know that
    // max(RHS, Start) = RHS at this point.
    const SCEV *End;
    if (canProveRHSGreaterThanEqualStart()) {
      End = RHS;
    } else {
      // If RHS < Start, the backedge will be taken zero times.  So in
      // general, we can write the backedge-taken count as:
      //
      //     RHS >= Start ? ceil(RHS - Start) / Stride : 0
      //
      // We convert it to the following to make it more convenient for SCEV:
      //
      //     ceil(max(RHS, Start) - Start) / Stride
      End = IsSigned ? getSMaxExpr(RHS, Start) : getUMaxExpr(RHS, Start);

      // See what would happen if we assume the backedge is taken. This is
      // used to compute MaxBECount.
      BECountIfBackedgeTaken =
          getUDivCeilSCEV(getMinusSCEV(RHS, Start), Stride);
    }

    // At this point, we know:
    //
    // 1. If IsSigned, Start <=s End; otherwise, Start <=u End
    // 2. The index variable doesn't overflow.
    //
    // Therefore, we know N exists such that
    // (Start + Stride * N) >= End, and computing "(Start + Stride * N)"
    // doesn't overflow.
    //
    // Using this information, try to prove whether the addition in
    // "(Start - End) + (Stride - 1)" has unsigned overflow.
    const SCEV *One = getOne(Stride->getType());
    bool MayAddOverflow = [&] {
      if (auto *StrideC = dyn_cast<SCEVConstant>(Stride)) {
        if (StrideC->getAPInt().isPowerOf2()) {
          // Suppose Stride is a power of two, and Start/End are unsigned
          // integers.  Let UMAX be the largest representable unsigned
          // integer.
          //
          // By the preconditions of this function, we know
          // "(Start + Stride * N) >= End", and this doesn't overflow.
          // As a formula:
          //
          //   End <= (Start + Stride * N) <= UMAX
          //
          // Subtracting Start from all the terms:
          //
          //   End - Start <= Stride * N <= UMAX - Start
          //
          // Since Start is unsigned, UMAX - Start <= UMAX.  Therefore:
          //
          //   End - Start <= Stride * N <= UMAX
          //
          // Stride * N is a multiple of Stride. Therefore,
          //
          //   End - Start <= Stride * N <= UMAX - (UMAX mod Stride)
          //
          // Since Stride is a power of two, UMAX + 1 is divisible by Stride.
          // Therefore, UMAX mod Stride == Stride - 1.  So we can write:
          //
          //   End - Start <= Stride * N <= UMAX - Stride - 1
          //
          // Dropping the middle term:
          //
          //   End - Start <= UMAX - Stride - 1
          //
          // Adding Stride - 1 to both sides:
          //
          //   (End - Start) + (Stride - 1) <= UMAX
          //
          // In other words, the addition doesn't have unsigned overflow.
          //
          // A similar proof works if we treat Start/End as signed values.
          // Just rewrite steps before "End - Start <= Stride * N <= UMAX" to
          // use signed max instead of unsigned max. Note that we're trying
          // to prove a lack of unsigned overflow in either case.
          return false;
        }
      }
      if (Start == Stride || Start == getMinusSCEV(Stride, One)) {
        // If Start is equal to Stride, (End - Start) + (Stride - 1) == End - 1.
        // If !IsSigned, 0 <u Stride == Start <=u End; so 0 <u End - 1 <u End.
        // If IsSigned, 0 <s Stride == Start <=s End; so 0 <s End - 1 <s End.
        //
        // If Start is equal to Stride - 1, (End - Start) + Stride - 1 == End.
        return false;
      }
      return true;
    }();

    const SCEV *Delta = getMinusSCEV(End, Start);
    if (!MayAddOverflow) {
      // floor((D + (S - 1)) / S)
      // We prefer this formulation if it's legal because it's fewer operations.
      BECount =
          getUDivExpr(getAddExpr(Delta, getMinusSCEV(Stride, One)), Stride);
    } else {
      BECount = getUDivCeilSCEV(Delta, Stride);
    }
  }

  const SCEV *MaxBECount;
  bool MaxOrZero = false;
  if (isa<SCEVConstant>(BECount)) {
    MaxBECount = BECount;
  } else if (BECountIfBackedgeTaken &&
             isa<SCEVConstant>(BECountIfBackedgeTaken)) {
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
#else
ScalarEvolution::ExitLimit MustExitScalarEvolution::howManyLessThans(
    const SCEV *LHS, const SCEV *RHS, const Loop *L, bool IsSigned,
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
#endif
