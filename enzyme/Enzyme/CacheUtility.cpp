//===- CacheUtility.cpp - Caching values in the forward pass for later use
//-===//
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
// This file defines a base helper class CacheUtility that manages the cache
// of values from the forward pass for later use.
//
//===----------------------------------------------------------------------===//

#include "CacheUtility.h"
#include "FunctionUtils.h"

using namespace llvm;

/// Pack 8 bools together in a single byte
llvm::cl::opt<bool>
    EfficientBoolCache("enzyme-smallbool", cl::init(false), cl::Hidden,
                       cl::desc("Place 8 bools together in a single byte"));

llvm::cl::opt<bool>
    EnzymePrintPerf("enzyme-print-perf", cl::init(false), cl::Hidden,
                    cl::desc("Enable Enzyme to print performance info"));

llvm::cl::opt<bool> EfficientMaxCache(
    "enzyme-max-cache", cl::init(false), cl::Hidden,
    cl::desc(
        "Avoid reallocs when possible by potentially overallocating cache"));

CacheUtility::~CacheUtility() {}

/// Erase this instruction both from LLVM modules and any local data-structures
void CacheUtility::erase(Instruction *I) {
  assert(I);
  for (auto &lim : LimitCache) {
    assert(I != lim.first.first);
    assert(I != lim.second);
  }
  for (auto &ctx : loopContexts) {
    assert(ctx.second.var != I);
    assert(ctx.second.incvar != I);
    assert(ctx.second.antivaralloc != I);
    assert(ctx.second.trueLimit != I);
    assert(ctx.second.maxLimit != I);
  }
  for (const auto &pair : scopeMap) {
    if (pair.second.first == I) {
      llvm::errs() << *newFunc << "\n";
      dumpScope();
      llvm::errs() << *pair.first << "\n";
      llvm::errs() << *I << "\n";
      assert(0 && "erasing something in scope map");
    }
  }
  if (auto CI = dyn_cast<CallInst>(I)) {
    for (const auto &pair : scopeFrees) {
      if (pair.second.count(CI)) {
        llvm::errs() << *newFunc << "\n";
        llvm::errs() << *pair.first << "\n";
        llvm::errs() << *I << "\n";
        assert(0 && "erasing something in scopeFrees map");
      }
    }
    for (const auto &pair : scopeAllocs) {
      if (std::find(pair.second.begin(), pair.second.end(), CI) !=
          pair.second.end()) {
        llvm::errs() << *newFunc << "\n";
        llvm::errs() << *pair.first << "\n";
        llvm::errs() << *I << "\n";
        assert(0 && "erasing something in scopeAllocs map");
      }
    }
  }
  for (const auto &pair : scopeInstructions) {
    if (std::find(pair.second.begin(), pair.second.end(), I) !=
        pair.second.end()) {
      llvm::errs() << *newFunc << "\n";
      llvm::errs() << *pair.first << "\n";
      llvm::errs() << *I << "\n";
      assert(0 && "erasing something in scopeInstructions map");
    }
  }

  if (auto found = findInMap(scopeMap, (Value *)I)) {
    scopeFrees.erase(found->first);
    scopeAllocs.erase(found->first);
    scopeInstructions.erase(found->first);
  }
  if (auto AI = dyn_cast<AllocaInst>(I)) {
    scopeFrees.erase(AI);
    scopeAllocs.erase(AI);
    scopeInstructions.erase(AI);
  }
  scopeMap.erase(I);
  SE.eraseValueFromMap(I);

  if (!I->use_empty()) {
    llvm::errs() << *newFunc->getParent() << "\n";
    llvm::errs() << *newFunc << "\n";
    llvm::errs() << *I << "\n";
  }
  assert(I->use_empty());
  I->eraseFromParent();
}

/// Replace this instruction both in LLVM modules and any local data-structures
void CacheUtility::replaceAWithB(Value *A, Value *B, bool storeInCache) {
  for (auto &ctx : loopContexts) {
    if (ctx.second.maxLimit == A) {
      ctx.second.maxLimit = B;
    }
    if (ctx.second.trueLimit == A) {
      ctx.second.trueLimit = B;
    }
  }

  auto found = scopeMap.find(A);
  if (found != scopeMap.end()) {
    insert_or_assign2(scopeMap, B, found->second);

    llvm::AllocaInst *cache = found->second.first;
    if (storeInCache) {
      assert(isa<Instruction>(B));
      if (scopeInstructions.find(cache) != scopeInstructions.end()) {
        for (auto st : scopeInstructions[cache])
          cast<StoreInst>(st)->eraseFromParent();
        scopeInstructions.erase(cache);
        storeInstructionInCache(found->second.second, cast<Instruction>(B),
                                cache);
      }
    }

    scopeMap.erase(A);
  }
  A->replaceAllUsesWith(B);
}

// Create a new canonical induction variable of Type Ty for Loop L
// Return the variable and the increment instruction
std::pair<PHINode *, Instruction *> InsertNewCanonicalIV(Loop *L, Type *Ty,
                                                         std::string name) {
  assert(L);
  assert(Ty);

  BasicBlock *Header = L->getHeader();
  assert(Header);
  IRBuilder<> B(&Header->front());
  PHINode *CanonicalIV = B.CreatePHI(Ty, 1, name);

  B.SetInsertPoint(Header->getFirstNonPHIOrDbg());
  Instruction *Inc = cast<Instruction>(
      B.CreateAdd(CanonicalIV, ConstantInt::get(Ty, 1), name + ".next",
                  /*NUW*/ true, /*NSW*/ true));

  for (BasicBlock *Pred : predecessors(Header)) {
    assert(Pred);
    if (L->contains(Pred)) {
      CanonicalIV->addIncoming(Inc, Pred);
    } else {
      CanonicalIV->addIncoming(ConstantInt::get(Ty, 0), Pred);
    }
  }
  return std::pair<PHINode *, Instruction *>(CanonicalIV, Inc);
}

// Attempt to rewrite all phinode's in the loop in terms of the
// induction variable
void RemoveRedundantIVs(BasicBlock *Header, PHINode *CanonicalIV,
                        MustExitScalarEvolution &SE,
                        std::function<void(Instruction *)> eraser) {
  assert(Header);
  assert(CanonicalIV);
  SmallVector<Instruction *, 8> IVsToRemove;

  // This scope is necessary to ensure scevexpander cleans up before we erase
  // things
  {
#if LLVM_VERSION_MAJOR >= 12
    SCEVExpander Exp(SE, Header->getParent()->getParent()->getDataLayout(),
                     "enzyme");
#else
    fake::SCEVExpander Exp(
        SE, Header->getParent()->getParent()->getDataLayout(), "enzyme");
#endif

    for (BasicBlock::iterator II = Header->begin(); isa<PHINode>(II); ++II) {
      PHINode *PN = cast<PHINode>(II);
      if (PN == CanonicalIV)
        continue;
      if (!SE.isSCEVable(PN->getType()))
        continue;
      const SCEV *S = SE.getSCEV(PN);
      if (SE.getCouldNotCompute() == S)
        continue;
      // we may expand code for phi where not legal (computing with
      // subloop expressions). Check that this isn't the case
      if (!SE.dominates(S, Header))
        continue;
      // We place that at first non phi as it may produce a non-phi instruction
      // and must thus be expanded after all phi's
      Value *NewIV =
          Exp.expandCodeFor(S, PN->getType(), Header->getFirstNonPHI());
      assert(NewIV->getType() == PN->getType());
      if (NewIV == PN) {
        continue;
      }
      if (auto BO = dyn_cast<BinaryOperator>(NewIV)) {
        if (BO->getOpcode() == BinaryOperator::Add ||
            BO->getOpcode() == BinaryOperator::Mul) {
          BO->setHasNoSignedWrap(true);
          BO->setHasNoUnsignedWrap(true);
        }
        for (int i = 0; i < 2; ++i) {
          if (auto BO2 = dyn_cast<BinaryOperator>(BO->getOperand(i))) {
            if (BO2->getOpcode() == BinaryOperator::Add ||
                BO2->getOpcode() == BinaryOperator::Mul) {
              BO2->setHasNoSignedWrap(true);
              BO2->setHasNoUnsignedWrap(true);
            }
          }
        }
      }

      PN->replaceAllUsesWith(NewIV);
      IVsToRemove.push_back(PN);
    }
  }

  for (Instruction *PN : IVsToRemove) {
    eraser(PN);
  }
}

void CanonicalizeLatches(const Loop *L, BasicBlock *Header,
                         BasicBlock *Preheader, PHINode *CanonicalIV,
                         MustExitScalarEvolution &SE, CacheUtility &gutils,
                         Instruction *Increment,
                         const SmallVectorImpl<BasicBlock *> &&latches) {
  // Attempt to explicitly rewrite the latch
  if (latches.size() == 1 && isa<BranchInst>(latches[0]->getTerminator()) &&
      cast<BranchInst>(latches[0]->getTerminator())->isConditional())
    for (auto use : CanonicalIV->users()) {
      if (auto cmp = dyn_cast<ICmpInst>(use)) {
        if (cast<BranchInst>(latches[0]->getTerminator())->getCondition() !=
            cmp)
          continue;
        // Force i to be on LHS
        if (cmp->getOperand(0) != CanonicalIV) {
          // Below also swaps predicate correctly
          cmp->swapOperands();
        }
        assert(cmp->getOperand(0) == CanonicalIV);

        auto scv = SE.getSCEVAtScope(cmp->getOperand(1), L);
        if (cmp->isUnsigned() ||
            (scv != SE.getCouldNotCompute() && SE.isKnownNonNegative(scv))) {

          // valid replacements (since unsigned comparison and i starts at 0
          // counting up)

          // * i < n => i != n, valid since first time i >= n occurs at i == n
          if (cmp->getPredicate() == ICmpInst::ICMP_ULT ||
              cmp->getPredicate() == ICmpInst::ICMP_SLT) {
            cmp->setPredicate(ICmpInst::ICMP_NE);
            goto cend;
          }

          // * i <= n => i != n+1, valid since first time i > n occurs at i ==
          // n+1 [ which we assert is in bitrange as not infinite loop ]
          if (cmp->getPredicate() == ICmpInst::ICMP_ULE ||
              cmp->getPredicate() == ICmpInst::ICMP_SLE) {
            IRBuilder<> builder(Preheader->getTerminator());
            if (auto inst = dyn_cast<Instruction>(cmp->getOperand(1))) {
              builder.SetInsertPoint(inst->getNextNode());
            }
            cmp->setOperand(
                1,
                builder.CreateNUWAdd(
                    cmp->getOperand(1),
                    ConstantInt::get(cmp->getOperand(1)->getType(), 1, false)));
            cmp->setPredicate(ICmpInst::ICMP_NE);
            goto cend;
          }

          // * i >= n => i == n, valid since first time i >= n occurs at i == n
          if (cmp->getPredicate() == ICmpInst::ICMP_UGE ||
              cmp->getPredicate() == ICmpInst::ICMP_SGE) {
            cmp->setPredicate(ICmpInst::ICMP_EQ);
            goto cend;
          }

          // * i > n => i == n+1, valid since first time i > n occurs at i ==
          // n+1 [ which we assert is in bitrange as not infinite loop ]
          if (cmp->getPredicate() == ICmpInst::ICMP_UGT ||
              cmp->getPredicate() == ICmpInst::ICMP_SGT) {
            IRBuilder<> builder(Preheader->getTerminator());
            if (auto inst = dyn_cast<Instruction>(cmp->getOperand(1))) {
              builder.SetInsertPoint(inst->getNextNode());
            }
            cmp->setOperand(
                1,
                builder.CreateNUWAdd(
                    cmp->getOperand(1),
                    ConstantInt::get(cmp->getOperand(1)->getType(), 1, false)));
            cmp->setPredicate(ICmpInst::ICMP_EQ);
            goto cend;
          }
        }
      cend:;
        if (cmp->getPredicate() == ICmpInst::ICMP_NE) {
        }
      }
    }

  // Replace previous increment usage with new increment value
  if (Increment) {
    Increment->moveAfter(CanonicalIV->getParent()->getFirstNonPHI());
    std::vector<Instruction *> toerase;
    // Replace existing increments with canonical Increment
    for (auto use : CanonicalIV->users()) {
      auto BO = dyn_cast<BinaryOperator>(use);
      if (BO == nullptr)
        continue;
      if (BO->getOpcode() != BinaryOperator::Add)
        continue;
      if (use == Increment)
        continue;

      Value *toadd = nullptr;
      if (BO->getOperand(0) == CanonicalIV) {
        toadd = BO->getOperand(1);
      } else {
        assert(BO->getOperand(1) == CanonicalIV);
        toadd = BO->getOperand(0);
      }
      if (auto CI = dyn_cast<ConstantInt>(toadd)) {
        if (!CI->isOne())
          continue;
        BO->replaceAllUsesWith(Increment);
        toerase.push_back(BO);
      } else {
        continue;
      }
    }
    for (auto inst : toerase) {
      gutils.erase(inst);
    }
    toerase.clear();

    if (latches.size() == 1 && isa<BranchInst>(latches[0]->getTerminator()) &&
        cast<BranchInst>(latches[0]->getTerminator())->isConditional())
      for (auto use : Increment->users()) {
        if (auto cmp = dyn_cast<ICmpInst>(use)) {
          if (cast<BranchInst>(latches[0]->getTerminator())->getCondition() !=
              cmp)
            continue;

          // Force i+1 to be on LHS
          if (cmp->getOperand(0) != Increment) {
            // Below also swaps predicate correctly
            cmp->swapOperands();
          }
          assert(cmp->getOperand(0) == Increment);

          auto scv = SE.getSCEVAtScope(cmp->getOperand(1), L);
          if (cmp->isUnsigned() ||
              (scv != SE.getCouldNotCompute() && SE.isKnownNonNegative(scv))) {

            // valid replacements (since unsigned comparison and i starts at 0
            // counting up)

            // * i+1 < n => i+1 != n, valid since first time i+1 >= n occurs at
            // i+1 == n
            if (cmp->getPredicate() == ICmpInst::ICMP_ULT ||
                cmp->getPredicate() == ICmpInst::ICMP_SLT) {
              cmp->setPredicate(ICmpInst::ICMP_NE);
              continue;
            }

            // * i+1 <= n => i != n, valid since first time i+1 > n occurs at
            // i+1 == n+1 => i == n
            if (cmp->getPredicate() == ICmpInst::ICMP_ULE ||
                cmp->getPredicate() == ICmpInst::ICMP_SLE) {
              cmp->setOperand(0, CanonicalIV);
              cmp->setPredicate(ICmpInst::ICMP_NE);
              continue;
            }

            // * i+1 >= n => i+1 == n, valid since first time i+1 >= n occurs at
            // i+1 == n
            if (cmp->getPredicate() == ICmpInst::ICMP_UGE ||
                cmp->getPredicate() == ICmpInst::ICMP_SGE) {
              cmp->setPredicate(ICmpInst::ICMP_EQ);
              continue;
            }

            // * i+1 > n => i == n, valid since first time i+1 > n occurs at i+1
            // == n+1 => i == n
            if (cmp->getPredicate() == ICmpInst::ICMP_UGT ||
                cmp->getPredicate() == ICmpInst::ICMP_SGT) {
              cmp->setOperand(0, CanonicalIV);
              cmp->setPredicate(ICmpInst::ICMP_EQ);
              continue;
            }
          }
        }
      }
  }
}

bool CacheUtility::getContext(BasicBlock *BB, LoopContext &loopContext) {
  Loop *L = LI.getLoopFor(BB);

  // Not inside a loop
  if (L == nullptr)
    return false;

  // Previously handled this loop
  if (auto found = findInMap(loopContexts, L)) {
    loopContext = *found;
    return true;
  }

  // Need to canonicalize
  loopContexts[L].parent = L->getParentLoop();

  loopContexts[L].header = L->getHeader();
  assert(loopContexts[L].header && "loop must have header");

  loopContexts[L].preheader = L->getLoopPreheader();
  if (!L->getLoopPreheader()) {
    llvm::errs() << "fn: " << *L->getHeader()->getParent() << "\n";
    llvm::errs() << "L: " << *L << "\n";
  }
  assert(loopContexts[L].preheader && "loop must have preheader");
  getExitBlocks(L, loopContexts[L].exitBlocks);

  auto pair = InsertNewCanonicalIV(L, Type::getInt64Ty(BB->getContext()));
  PHINode *CanonicalIV = pair.first;
  assert(CanonicalIV);
  loopContexts[L].var = CanonicalIV;
  loopContexts[L].incvar = pair.second;
  RemoveRedundantIVs(loopContexts[L].header, CanonicalIV, SE,
                     [&](Instruction *I) { erase(I); });
  CanonicalizeLatches(L, loopContexts[L].header, loopContexts[L].preheader,
                      CanonicalIV, SE, *this, pair.second,
                      getLatches(L, loopContexts[L].exitBlocks));
  loopContexts[L].antivaralloc =
      IRBuilder<>(inversionAllocs)
          .CreateAlloca(CanonicalIV->getType(), nullptr,
                        CanonicalIV->getName() + "'ac");
#if LLVM_VERSION_MAJOR >= 10
  loopContexts[L].antivaralloc->setAlignment(
      Align(cast<IntegerType>(CanonicalIV->getType())->getBitWidth() / 8));
#else
  loopContexts[L].antivaralloc->setAlignment(
      cast<IntegerType>(CanonicalIV->getType())->getBitWidth() / 8);
#endif

  SCEVUnionPredicate BackedgePred;

  const SCEV *Limit = nullptr;
  const SCEV *MaxIterations = nullptr;
  {
    const SCEV *MayExitMaxBECount = nullptr;

    SmallVector<BasicBlock *, 8> ExitingBlocks;
    L->getExitingBlocks(ExitingBlocks);

    // Remove all exiting blocks that are guaranteed
    // to result in unreachable
    for (auto &ExitingBlock : ExitingBlocks) {
      BasicBlock *Exit = nullptr;
      for (auto *SBB : successors(ExitingBlock)) {
        if (!L->contains(SBB)) {
          if (SE.GuaranteedUnreachable.count(SBB))
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

    // Compute the exit in the scenarios where an unreachable
    // is not hit
    for (BasicBlock *ExitingBlock : ExitingBlocks) {
      assert(L->contains(ExitingBlock));

      ScalarEvolution::ExitLimit EL =
          SE.computeExitLimit(L, ExitingBlock, /*AllowPredicates*/ true);

      bool seenHeaders = false;
      SmallPtrSet<BasicBlock *, 4> Seen;
      std::deque<BasicBlock *> Todo = {ExitingBlock};
      while (Todo.size()) {
        auto cur = Todo.front();
        Todo.pop_front();
        if (Seen.count(cur))
          continue;
        if (!L->contains(cur))
          continue;
        if (cur == loopContexts[L].header) {
          seenHeaders = true;
          break;
        }
        for (auto S : successors(cur)) {
          Todo.push_back(S);
        }
      }
      if (seenHeaders) {
        if (MaxIterations == nullptr ||
            MaxIterations == SE.getCouldNotCompute()) {
          MaxIterations = EL.ExactNotTaken;
        }
        if (MaxIterations != SE.getCouldNotCompute()) {
          if (EL.ExactNotTaken != SE.getCouldNotCompute()) {
            MaxIterations =
                SE.getUMaxFromMismatchedTypes(MaxIterations, EL.ExactNotTaken);
          }
        }

        if (MayExitMaxBECount == nullptr ||
            EL.ExactNotTaken == SE.getCouldNotCompute())
          MayExitMaxBECount = EL.ExactNotTaken;

        if (EL.ExactNotTaken != MayExitMaxBECount) {
          MayExitMaxBECount = SE.getCouldNotCompute();
        }
      }
    }
    if (MayExitMaxBECount == nullptr) {
      MayExitMaxBECount = SE.getCouldNotCompute();
    }
    if (MaxIterations == nullptr) {
      MaxIterations = SE.getCouldNotCompute();
    }
    Limit = MayExitMaxBECount;
  }
  assert(Limit);
  Value *LimitVar = nullptr;

  if (SE.getCouldNotCompute() != Limit) {

    if (CanonicalIV == nullptr) {
      report_fatal_error("Couldn't get canonical IV.");
    }

    if (Limit->getType() != CanonicalIV->getType())
      Limit = SE.getZeroExtendExpr(Limit, CanonicalIV->getType());

#if LLVM_VERSION_MAJOR >= 12
    SCEVExpander Exp(SE, BB->getParent()->getParent()->getDataLayout(),
                     "enzyme");
#else
    fake::SCEVExpander Exp(SE, BB->getParent()->getParent()->getDataLayout(),
                           "enzyme");
#endif
    LimitVar = Exp.expandCodeFor(Limit, CanonicalIV->getType(),
                                 loopContexts[L].preheader->getTerminator());
    loopContexts[L].dynamic = false;
    loopContexts[L].maxLimit = LimitVar;
  } else {
    DebugLoc loc = L->getHeader()->begin()->getDebugLoc();
    for (auto &I : *L->getHeader()) {
      if (loc)
        break;
      loc = I.getDebugLoc();
    }
    EmitWarning("NoLimit", loc, newFunc, L->getHeader(),
                "SE could not compute loop limit of ",
                L->getHeader()->getName(), " of ",
                L->getHeader()->getParent()->getName(), "lim: ", *Limit,
                " maxlim: ", *MaxIterations);

    LimitVar = createCacheForScope(LimitContext(loopContexts[L].preheader),
                                   CanonicalIV->getType(), "loopLimit",
                                   /*shouldfree*/ false);

    for (auto ExitBlock : loopContexts[L].exitBlocks) {
      IRBuilder<> B(&ExitBlock->front());
      auto Limit = B.CreatePHI(CanonicalIV->getType(), 1);

      for (BasicBlock *Pred : predecessors(ExitBlock)) {
        if (LI.getLoopFor(Pred) == L) {
          Limit->addIncoming(CanonicalIV, Pred);
        } else {
          Limit->addIncoming(UndefValue::get(CanonicalIV->getType()), Pred);
        }
      }

      storeInstructionInCache(loopContexts[L].preheader, Limit,
                              cast<AllocaInst>(LimitVar));
    }
    loopContexts[L].dynamic = true;
    loopContexts[L].maxLimit = nullptr;
  }
  loopContexts[L].trueLimit = LimitVar;
  if (EfficientMaxCache && loopContexts[L].dynamic &&
      SE.getCouldNotCompute() != MaxIterations) {
    if (MaxIterations->getType() != CanonicalIV->getType())
      MaxIterations =
          SE.getZeroExtendExpr(MaxIterations, CanonicalIV->getType());

#if LLVM_VERSION_MAJOR >= 12
    SCEVExpander Exp(SE, BB->getParent()->getParent()->getDataLayout(),
                     "enzyme");
#else
    fake::SCEVExpander Exp(SE, BB->getParent()->getParent()->getDataLayout(),
                           "enzyme");
#endif

    loopContexts[L].maxLimit =
        Exp.expandCodeFor(MaxIterations, CanonicalIV->getType(),
                          loopContexts[L].preheader->getTerminator());
  }

  loopContext = loopContexts.find(L)->second;
  return true;
}

/// Caching mechanism: creates a cache of type T in a scope given by ctx
/// (where if ctx is in a loop there will be a corresponding number of slots)
AllocaInst *CacheUtility::createCacheForScope(LimitContext ctx, Type *T,
                                              StringRef name, bool shouldFree,
                                              bool allocateInternal,
                                              Value *extraSize) {
  assert(ctx.Block);
  assert(T);

  auto sublimits = getSubLimits(/*inForwardPass*/ true, nullptr, ctx);

  // List of types stored in the cache for each Loop-Chunk
  // This is stored from innner-most chunk to outermost
  // Thus it begins with the underlying type, and adds pointers
  // to the previous type.
  std::vector<Type *> types = {T};
  bool isi1 = T->isIntegerTy() && cast<IntegerType>(T)->getBitWidth() == 1;
  if (EfficientBoolCache && isi1 && sublimits.size() != 0)
    types[0] = Type::getInt8Ty(T->getContext());
  for (size_t i = 0; i < sublimits.size(); ++i) {
    types.push_back(PointerType::getUnqual(types.back()));
  }

  // Allocate the outermost type on the stack
  IRBuilder<> entryBuilder(inversionAllocs);
  entryBuilder.setFastMathFlags(getFast());
  AllocaInst *alloc =
      entryBuilder.CreateAlloca(types.back(), nullptr, name + "_cache");
  {
    ConstantInt *byteSizeOfType = ConstantInt::get(
        Type::getInt64Ty(T->getContext()),
        newFunc->getParent()->getDataLayout().getTypeAllocSizeInBits(
            types.back()) /
            8);
    unsigned bsize = (unsigned)byteSizeOfType->getZExtValue();
    if ((bsize & (bsize - 1)) == 0) {
#if LLVM_VERSION_MAJOR >= 10
      alloc->setAlignment(Align(bsize));
#else
      alloc->setAlignment(bsize);
#endif
    }
  }

  Type *BPTy = Type::getInt8PtrTy(T->getContext());
  auto realloc = newFunc->getParent()->getOrInsertFunction(
      "realloc", BPTy, BPTy, Type::getInt64Ty(T->getContext()));

  Value *storeInto = alloc;

  // Iterating from outermost chunk to innermost chunk
  // Allocate and store the requisite memory if needed
  // and lookup the next level pointer of the cache
  for (int i = sublimits.size() - 1; i >= 0; i--) {
    const auto &containedloops = sublimits[i].second;

    Type *myType = types[i];

    ConstantInt *byteSizeOfType = ConstantInt::get(
        Type::getInt64Ty(T->getContext()),
        newFunc->getParent()->getDataLayout().getTypeAllocSizeInBits(myType) /
            8);

    // if (i != sublimits.size() -1 || !ompOffset)
    // Allocate and store the required memory
    if (allocateInternal) {

      IRBuilder<> allocationBuilder(
          &containedloops.back().first.preheader->back());

      Value *size = sublimits[i].first;
      if (EfficientBoolCache && isi1 && i == 0) {
        size = allocationBuilder.CreateLShr(
            allocationBuilder.CreateAdd(
                size, ConstantInt::get(Type::getInt64Ty(T->getContext()), 7),
                "", true),
            ConstantInt::get(Type::getInt64Ty(T->getContext()), 3));
      }
      if (extraSize && i == 0) {
        Value *es = unwrapM(extraSize, allocationBuilder,
                            /*available*/ ValueToValueMapTy(),
                            UnwrapMode::AttemptFullUnwrapWithLookup);
        assert(es);
        size = allocationBuilder.CreateMul(size, es, "", /*NUW*/ true,
                                           /*NSW*/ true);
      }

      StoreInst *storealloc = nullptr;
      // Statically allocate memory for all iterations if possible
      if (sublimits[i].second.back().first.maxLimit) {
        auto firstallocation = CallInst::CreateMalloc(
            &allocationBuilder.GetInsertBlock()->back(), size->getType(),
            myType, byteSizeOfType, size, nullptr, name + "_malloccache");
        CallInst *malloccall = dyn_cast<CallInst>(firstallocation);
        if (malloccall == nullptr) {
          malloccall =
              cast<CallInst>(cast<Instruction>(firstallocation)->getOperand(0));
        }

        // Assert computation of size of array doesn't wrap
        if (auto BI = dyn_cast<BinaryOperator>(malloccall->getArgOperand(0))) {
          if ((BI->getOperand(0) == byteSizeOfType &&
               BI->getOperand(1) == size) ||
              (BI->getOperand(1) == byteSizeOfType &&
               BI->getOperand(0) == size))
            BI->setHasNoSignedWrap(true);
          BI->setHasNoUnsignedWrap(true);
        }

        if (auto ci = dyn_cast<ConstantInt>(size)) {
          malloccall->addDereferenceableAttr(
              llvm::AttributeList::ReturnIndex,
              ci->getLimitedValue() * byteSizeOfType->getLimitedValue());
          malloccall->addDereferenceableOrNullAttr(
              llvm::AttributeList::ReturnIndex,
              ci->getLimitedValue() * byteSizeOfType->getLimitedValue());
          // malloccall->removeAttribute(llvm::AttributeList::ReturnIndex,
          // Attribute::DereferenceableOrNull);
        }
        malloccall->addAttribute(AttributeList::ReturnIndex,
                                 Attribute::NoAlias);
        malloccall->addAttribute(AttributeList::ReturnIndex,
                                 Attribute::NonNull);

        storealloc = allocationBuilder.CreateStore(firstallocation, storeInto);

        scopeAllocs[alloc].push_back(malloccall);

        // Mark the store as invariant since the allocation is static and
        // will not be changed
        if (CachePointerInvariantGroups.find(std::make_pair(
                (Value *)alloc, i)) == CachePointerInvariantGroups.end()) {
          MDNode *invgroup = MDNode::getDistinct(alloc->getContext(), {});
          CachePointerInvariantGroups[std::make_pair((Value *)alloc, i)] =
              invgroup;
        }
        storealloc->setMetadata(
            LLVMContext::MD_invariant_group,
            CachePointerInvariantGroups[std::make_pair((Value *)alloc, i)]);
      } else {
        // Reallocate memory dynamically as a fallback
        // TODO change this to a power-of-two allocation strategy
        auto zerostore = allocationBuilder.CreateStore(
            ConstantPointerNull::get(PointerType::getUnqual(myType)),
            storeInto);
        scopeInstructions[alloc].push_back(zerostore);

        IRBuilder<> build(containedloops.back().first.incvar->getNextNode());
        Value *allocation = build.CreateLoad(storeInto);
        Value *realloc_size = nullptr;
        if (isa<ConstantInt>(sublimits[i].first) &&
            cast<ConstantInt>(sublimits[i].first)->isOne()) {
          realloc_size = containedloops.back().first.incvar;
        } else {
          realloc_size = build.CreateMul(containedloops.back().first.incvar,
                                         sublimits[i].first, "", /*NUW*/ true,
                                         /*NSW*/ true);
        }

        Value *idxs[2] = {
            build.CreatePointerCast(allocation, BPTy),
            build.CreateMul(
                ConstantInt::get(size->getType(),
                                 newFunc->getParent()
                                         ->getDataLayout()
                                         .getTypeAllocSizeInBits(myType) /
                                     8),
                realloc_size, "", /*NUW*/ true, /*NSW*/ true)};

        Value *realloccall = nullptr;
        allocation = build.CreatePointerCast(
            realloccall =
                build.CreateCall(realloc, idxs, name + "_realloccache"),
            allocation->getType(), name + "_realloccast");
        scopeAllocs[alloc].push_back(cast<CallInst>(realloccall));
        storealloc = build.CreateStore(allocation, storeInto);
        // Unlike the static case we can not mark the memory as invariant
        // since we are reloading/storing based off the number of loop
        // iterations
      }

      // Regardless of how allocated (dynamic vs static), mark it
      // as having the requisite alignment
      unsigned bsize = (unsigned)byteSizeOfType->getZExtValue();
      if ((bsize & (bsize - 1)) == 0) {
#if LLVM_VERSION_MAJOR >= 10
        storealloc->setAlignment(Align(bsize));
#else
        storealloc->setAlignment(bsize);
#endif
      }
      scopeInstructions[alloc].push_back(storealloc);
    }

    // Free the memory, if requested
    if (i != sublimits.size() - 1 || !ompOffset)
      if (shouldFree) {
        if (CachePointerInvariantGroups.find(std::make_pair(
                (Value *)alloc, i)) == CachePointerInvariantGroups.end()) {
          MDNode *invgroup = MDNode::getDistinct(alloc->getContext(), {});
          CachePointerInvariantGroups[std::make_pair((Value *)alloc, i)] =
              invgroup;
        }
        freeCache(
            containedloops.back().first.preheader, sublimits, i, alloc,
            byteSizeOfType, storeInto,
            CachePointerInvariantGroups[std::make_pair((Value *)alloc, i)]);
      }

    // If we are not the final iteration, lookup the next pointer by indexing
    // into the relevant location of the current chunk allocation
    if (i != 0) {
      IRBuilder<> v(&sublimits[i - 1].second.back().first.preheader->back());

      Value *idx = computeIndexOfChunk(
          /*inForwardPass*/ true, v, containedloops,
          (i == sublimits.size() - 1) ? ompOffset : nullptr);

      storeInto = v.CreateGEP(v.CreateLoad(storeInto), idx);
      cast<GetElementPtrInst>(storeInto)->setIsInBounds(true);
    }
  }
  return alloc;
}

Value *CacheUtility::computeIndexOfChunk(
    bool inForwardPass, IRBuilder<> &v,
    const std::vector<std::pair<LoopContext, llvm::Value *>> &containedloops,
    Value *outerOffset) {
  // List of loop indices in chunk from innermost to outermost
  SmallVector<Value *, 3> indices;
  // List of cumulative indices in chunk from innermost to outermost
  // where limit[i] = prod(loop limit[0..i])
  SmallVector<Value *, 3> limits;

  // list of contained loop induction variables available for limit
  // computation
  ValueToValueMapTy available;

  // Iterate from innermost loop to outermost loop within a chunk
  for (size_t i = 0; i < containedloops.size(); ++i) {
    const auto &pair = containedloops[i];

    const auto &idx = pair.first;
    Value *var = idx.var;

    // In the SingleIteration, var may be null (since there's no legal phinode)
    // In that case the current iteration is simply the constnat Zero
    if (var == nullptr)
      var = ConstantInt::get(Type::getInt64Ty(newFunc->getContext()), 0);
    else if (!inForwardPass) {
      var = v.CreateLoad(idx.antivaralloc);
      available[idx.var] = var;
    } else {
      var = idx.var;
      available[idx.var] = var;
    }
    if (i == containedloops.size() - 1 && outerOffset) {
      var = v.CreateAdd(var, lookupM(outerOffset, v), "", /*NUW*/ true,
                        /*NSW*/ true);
    }

    indices.push_back(var);
    Value *lim = pair.second;
    assert(lim);
    if (limits.size() == 0) {
      limits.push_back(lim);
    } else {
      limits.push_back(v.CreateMul(limits.back(), lim, "",
                                   /*NUW*/ true, /*NSW*/ true));
    }
  }

  assert(indices.size() > 0);

  // Compute the index into the pointer
  Value *idx = indices[0];
  for (unsigned ind = 1; ind < indices.size(); ++ind) {
    idx = v.CreateAdd(idx,
                      v.CreateMul(indices[ind], limits[ind - 1], "",
                                  /*NUW*/ true, /*NSW*/ true),
                      "", /*NUW*/ true, /*NSW*/ true);
  }
  return idx;
}

/// Given a LimitContext ctx, representing a location inside a loop nest,
/// break each of the loops up into chunks of loops where each chunk's number
/// of iterations can be computed at the chunk preheader. Every dynamic loop
/// defines the start of a chunk. SubLimitType is a vector of chunk objects.
/// More specifically it is a vector of { # iters in a Chunk (sublimit), Chunk }
/// Each chunk object is a vector of loops contained within the chunk.
/// For every loop, this returns pair of the LoopContext and the limit of that
/// loop Both the vector of Chunks and vector of Loops within a Chunk go from
/// innermost loop to outermost loop.
CacheUtility::SubLimitType CacheUtility::getSubLimits(bool inForwardPass,
                                                      IRBuilder<> *RB,
                                                      LimitContext ctx) {
  // Given a ``SingleIteration'' Limit Context, return a chunking of
  // one loop with size 1, and header/preheader of the BasicBlock
  // This is done to create a context for a block outside a loop
  // and is part of an experimental mechanism for merging stores
  // into a unified memcpy
  if (ctx.ForceSingleIteration) {
    LoopContext idx;
    auto subctx = ctx.Block;
    auto zero = ConstantInt::get(Type::getInt64Ty(newFunc->getContext()), 0);
    auto one = ConstantInt::get(Type::getInt64Ty(newFunc->getContext()), 1);
    // The iteration count is always zero so we can set it as such
    idx.var = nullptr; // = zero;
    idx.incvar = nullptr;
    idx.antivaralloc = nullptr;
    idx.trueLimit = zero;
    idx.maxLimit = zero;
    idx.header = subctx;
    idx.preheader = subctx;
    idx.dynamic = false;
    idx.parent = nullptr;
    idx.exitBlocks = {};
    SubLimitType sublimits;
    sublimits.push_back({one, {{idx, one}}});
    return sublimits;
  }

  // Store the LoopContext's in InnerMost => Outermost order
  std::vector<LoopContext> contexts;
  for (BasicBlock *blk = ctx.Block; blk != nullptr;) {
    LoopContext idx;
    if (!getContext(blk, idx)) {
      break;
    }
    contexts.emplace_back(idx);
    blk = idx.preheader;
  }
  if (ompTrueLimit && contexts.size()) {
    contexts.back().trueLimit = ompTrueLimit;
    contexts.back().maxLimit = ompTrueLimit;
  }

  // Legal preheaders for loop i (indexed from inner => outer)
  std::vector<BasicBlock *> allocationPreheaders(contexts.size(), nullptr);
  // Limit of loop i (indexed from inner => outer)
  std::vector<Value *> limits(contexts.size(), nullptr);

  // Iterate from outermost loop to innermost loop
  for (int i = contexts.size() - 1; i >= 0; --i) {
    // The outermost loop's preheader is the preheader directly
    // outside the loop nest
    if ((unsigned)i == contexts.size() - 1) {
      allocationPreheaders[i] = contexts[i].preheader;
    } else if (!contexts[i].maxLimit) {
      // For dynamic loops, the preheader is now forced to be the preheader
      // of that loop
      allocationPreheaders[i] = contexts[i].preheader;
    } else {
      // Otherwise try to use the preheader of the loop just outside this
      // one to allocate all iterations across both loops together
      allocationPreheaders[i] = allocationPreheaders[i + 1];
    }

    // Dynamic loops are considered to have a limit of one for allocation
    // purposes This is because we want to allocate 1 x (# of iterations inside
    // chunk) inside every dynamic iteration
    if (!contexts[i].maxLimit) {
      limits[i] =
          ConstantInt::get(Type::getInt64Ty(ctx.Block->getContext()), 1);
    } else {
      // Map of previous induction variables we are allowed to use as part
      // of the computation of the number of iterations in this chunk
      ValueToValueMapTy prevMap;

      // Iterate from outermost loop down
      for (int j = contexts.size() - 1;; --j) {
        // If the preheader allocating memory for loop i
        // is distinct from this preheader, we are therefore allocating
        // memory in a different chunk. We can use induction variables
        // from chunks outside us to compute loop bounds so add it to the
        // map
        if (allocationPreheaders[i] != contexts[j].preheader) {
          prevMap[contexts[j].var] = contexts[j].var;
        } else {
          break;
        }
      }

      IRBuilder<> allocationBuilder(&allocationPreheaders[i]->back());
      Value *limitMinus1 = nullptr;

      // Attempt to compute the limit of this loop at the corresponding
      // allocation preheader. This is null if it was not legal to compute
      limitMinus1 = unwrapM(contexts[i].maxLimit, allocationBuilder, prevMap,
                            UnwrapMode::AttemptFullUnwrap);

      // We have a loop with static bounds, but whose limit is not available
      // to be computed at the current loop preheader (such as the innermost
      // loop of triangular iteration domain) Handle this case like a dynamic
      // loop and create a new chunk.
      if (limitMinus1 == nullptr) {
        EmitWarning("NoOuterLimit",
                    cast<Instruction>(contexts[i].maxLimit)->getDebugLoc(),
                    newFunc,
                    cast<Instruction>(contexts[i].maxLimit)->getParent(),
                    "Could not compute outermost loop limit by moving value ",
                    *contexts[i].maxLimit, " computed at block",
                    contexts[i].header->getName(), " function ",
                    contexts[i].header->getParent()->getName());
        allocationPreheaders[i] = contexts[i].preheader;
        allocationBuilder.SetInsertPoint(&allocationPreheaders[i]->back());
        limitMinus1 = unwrapM(contexts[i].maxLimit, allocationBuilder, prevMap,
                              UnwrapMode::AttemptFullUnwrap);
      }
      assert(limitMinus1 != nullptr);

      ValueToValueMapTy reverseMap;
      // Iterate from outermost loop down
      for (int j = contexts.size() - 1;; --j) {
        // If the preheader allocating memory for loop i
        // is distinct from this preheader, we are therefore allocating
        // memory in a different chunk. We can use induction variables
        // from chunks outside us to compute loop bounds so add it to the
        // map
        if (allocationPreheaders[i] != contexts[j].preheader) {
          if (!inForwardPass) {
            reverseMap[contexts[j].var] =
                RB->CreateLoad(contexts[j].antivaralloc);
          }
        } else {
          break;
        }
      }

      // We now need to compute the actual limit as opposed to the limit
      // minus one. For efficiency, avoid doing this multiple times for
      // the same <limitMinus1, Block requested at> pair by caching inside
      // of LimitCache.
      auto cidx = std::make_pair(limitMinus1, allocationPreheaders[i]);
      if (LimitCache.find(cidx) == LimitCache.end()) {
        LimitCache[cidx] = allocationBuilder.CreateNUWAdd(
            limitMinus1, ConstantInt::get(limitMinus1->getType(), 1));
      }
      if (inForwardPass)
        limits[i] = LimitCache[cidx];
      else {
        Value *lim = unwrapM(contexts[i].maxLimit, *RB, reverseMap,
                             UnwrapMode::AttemptFullUnwrapWithLookup);
        if (!lim) {
          llvm::errs() << *newFunc << "\n";
          llvm::errs() << *contexts[i].maxLimit << "\n";
        }
        assert(lim);
        limits[i] = RB->CreateNUWAdd(lim, ConstantInt::get(lim->getType(), 1));
      }
    }
  }

  SubLimitType sublimits;

  // Total number of iterations of current chunk of loops
  Value *size = nullptr;
  // Loops inside current chunk (stored innermost to outermost)
  std::vector<std::pair<LoopContext, Value *>> lims;

  // Iterate from innermost to outermost loops
  for (unsigned i = 0; i < contexts.size(); ++i) {
    IRBuilder<> allocationBuilder(&allocationPreheaders[i]->back());
    lims.push_back(std::make_pair(contexts[i], limits[i]));
    // Compute the cumulative size
    if (size == nullptr) {
      // If starting with no cumulative size, this is the cumulative size
      size = limits[i];
    } else if (!inForwardPass) {
      size = RB->CreateMul(size, limits[i], "",
                           /*NUW*/ true, /*NSW*/ true);
    } else {
      // Otherwise new size = old size * limits[i];
      auto cidx = std::make_tuple(size, limits[i], allocationPreheaders[i]);
      if (SizeCache.find(cidx) == SizeCache.end()) {
        SizeCache[cidx] =
            allocationBuilder.CreateMul(size, limits[i], "",
                                        /*NUW*/ true, /*NSW*/ true);
      }
      size = SizeCache[cidx];
    }

    // If we are starting a new chunk in the next iteration
    // push this chunk to sublimits and clear the cumulative calculations
    if ((i + 1 < contexts.size()) &&
        (allocationPreheaders[i] != allocationPreheaders[i + 1])) {
      sublimits.push_back(std::make_pair(size, lims));
      size = nullptr;
      lims.clear();
    }
  }

  // For any remaining loop chunks, add them to the list
  if (size != nullptr) {
    sublimits.push_back(std::make_pair(size, lims));
    lims.clear();
  }

  return sublimits;
}

/// Given an allocation defined at a particular ctx, store the value val
/// in the cache at the location defined in the given builder
void CacheUtility::storeInstructionInCache(LimitContext ctx,
                                           IRBuilder<> &BuilderM, Value *val,
                                           AllocaInst *cache) {
  assert(BuilderM.GetInsertBlock()->getParent() == newFunc);
  if (auto inst = dyn_cast<Instruction>(val))
    assert(inst->getParent()->getParent() == newFunc);
  IRBuilder<> v(BuilderM.GetInsertBlock());
  v.SetInsertPoint(BuilderM.GetInsertBlock(), BuilderM.GetInsertPoint());
  v.setFastMathFlags(getFast());

  // Note for dynamic loops where the allocation is stored somewhere inside
  // the loop, we must ensure that we load the allocation after actually
  // storing the allocation itself.
  // To simplify things and ensure we always store after a
  // potential realloc occurs in this loop, we put our store after
  // any existing stores in the loop.
  // This is okay as there should be no load to the cache in the same block
  // where this instruction is defined as we will just use this instruction
  // TODO check that the store is actually aliasing/related
  if (BuilderM.GetInsertPoint() != BuilderM.GetInsertBlock()->end()) {
    for (auto I = BuilderM.GetInsertBlock()->rbegin(),
              E = BuilderM.GetInsertBlock()->rend();
         I != E; ++I) {
      if (&*I == &*BuilderM.GetInsertPoint())
        break;
      if (auto si = dyn_cast<StoreInst>(&*I)) {
        auto ni = getNextNonDebugInstructionOrNull(si);
        if (ni != nullptr) {
          v.SetInsertPoint(ni);
        } else {
          v.SetInsertPoint(si->getParent());
        }
      }
    }
  }

  bool isi1 = val->getType()->isIntegerTy() &&
              cast<IntegerType>(val->getType())->getBitWidth() == 1;
  Value *loc = getCachePointer(/*inForwardPass*/ true, v, ctx, cache, isi1,
                               /*storeInInstructionsMap*/ true);

  Value *tostore = val;

  // If we are doing the efficient bool cache, the actual value
  // we want to store needs to have the existing surrounding bits
  // set appropriately
  if (EfficientBoolCache && isi1) {
    if (auto gep = dyn_cast<GetElementPtrInst>(loc)) {
      auto bo = cast<BinaryOperator>(*gep->idx_begin());
      assert(bo->getOpcode() == BinaryOperator::LShr);
      auto subidx = v.CreateAnd(
          v.CreateTrunc(bo->getOperand(0),
                        Type::getInt8Ty(cache->getContext())),
          ConstantInt::get(Type::getInt8Ty(cache->getContext()), 7));
      auto mask = v.CreateNot(v.CreateShl(
          ConstantInt::get(Type::getInt8Ty(cache->getContext()), 1), subidx));

      auto cleared = v.CreateAnd(v.CreateLoad(loc), mask);

      auto toset = v.CreateShl(
          v.CreateZExt(val, Type::getInt8Ty(cache->getContext())), subidx);
      tostore = v.CreateOr(cleared, toset);
      assert(tostore->getType() ==
             cast<PointerType>(loc->getType())->getElementType());
    }
  }

  if (tostore->getType() !=
      cast<PointerType>(loc->getType())->getElementType()) {
    llvm::errs() << "val: " << *val << "\n";
    llvm::errs() << "tostore: " << *tostore << "\n";
    llvm::errs() << "loc: " << *loc << "\n";
  }
  assert(tostore->getType() ==
         cast<PointerType>(loc->getType())->getElementType());
  StoreInst *storeinst = v.CreateStore(tostore, loc);

  // If the value stored doesnt change (per efficient bool cache),
  // mark it as invariant
  if (tostore == val) {
    if (ValueInvariantGroups.find(cache) == ValueInvariantGroups.end()) {
      MDNode *invgroup = MDNode::getDistinct(cache->getContext(), {});
      ValueInvariantGroups[cache] = invgroup;
    }
    storeinst->setMetadata(LLVMContext::MD_invariant_group,
                           ValueInvariantGroups[cache]);
  }

  // Set alignment
  ConstantInt *byteSizeOfType =
      ConstantInt::get(Type::getInt64Ty(cache->getContext()),
                       ctx.Block->getParent()
                               ->getParent()
                               ->getDataLayout()
                               .getTypeAllocSizeInBits(val->getType()) /
                           8);
  unsigned bsize = (unsigned)byteSizeOfType->getZExtValue();
  if ((bsize & (bsize - 1)) == 0) {
#if LLVM_VERSION_MAJOR >= 10
    storeinst->setAlignment(Align(bsize));
#else
    storeinst->setAlignment(bsize);
#endif
  }
  scopeInstructions[cache].push_back(storeinst);
}

/// Given an allocation defined at a particular ctx, store the instruction
/// in the cache right after the instruction is executed
void CacheUtility::storeInstructionInCache(LimitContext ctx,
                                           llvm::Instruction *inst,
                                           llvm::AllocaInst *cache) {
  assert(ctx.Block);
  assert(inst);
  assert(cache);

  // Find the correct place to issue the store
  IRBuilder<> v(inst->getParent());
  // If this is a PHINode, we need to store after all phinodes,
  // otherwise just after inst sufficies
  if (&*inst->getParent()->rbegin() != inst) {
    auto pn = dyn_cast<PHINode>(inst);
    Instruction *putafter = (pn && pn->getNumIncomingValues() > 0)
                                ? (inst->getParent()->getFirstNonPHI())
                                : getNextNonDebugInstruction(inst);
    assert(putafter);
    v.SetInsertPoint(putafter);
  }
  v.setFastMathFlags(getFast());
  storeInstructionInCache(ctx, v, inst, cache);
}

/// Given an allocation specified by the LimitContext ctx and cache, compute a
/// pointer that can hold the underlying type being cached. This value should be
/// computed at BuilderM. Optionally, instructions needed to generate this
/// pointer can be stored in scopeInstructions
Value *CacheUtility::getCachePointer(bool inForwardPass, IRBuilder<> &BuilderM,
                                     LimitContext ctx, Value *cache, bool isi1,
                                     bool storeInInstructionsMap,
                                     Value *extraSize) {
  assert(ctx.Block);
  assert(cache);

  auto sublimits = getSubLimits(inForwardPass, &BuilderM, ctx);

  ValueToValueMapTy available;

  Value *next = cache;
  assert(next->getType()->isPointerTy());

  // Iterate from outermost loop to innermost loop
  for (int i = sublimits.size() - 1; i >= 0; i--) {
    // Lookup the next allocation pointer
    next = BuilderM.CreateLoad(next);
    if (storeInInstructionsMap && isa<AllocaInst>(cache))
      scopeInstructions[cast<AllocaInst>(cache)].push_back(
          cast<Instruction>(next));

    if (!next->getType()->isPointerTy()) {
      llvm::errs() << *newFunc << "\n";
      llvm::errs() << "cache: " << *cache << "\n";
      llvm::errs() << "next: " << *next << "\n";
      assert(next->getType()->isPointerTy());
    }

    // Set appropriate invairant lookup flags
    if (CachePointerInvariantGroups.find(std::make_pair(cache, i)) ==
        CachePointerInvariantGroups.end()) {
      MDNode *invgroup = MDNode::getDistinct(cache->getContext(), {});
      CachePointerInvariantGroups[std::make_pair(cache, i)] = invgroup;
    }
    cast<LoadInst>(next)->setMetadata(
        LLVMContext::MD_invariant_group,
        CachePointerInvariantGroups[std::make_pair(cache, i)]);

    // Set dereferenceable and alignment flags
    ConstantInt *byteSizeOfType = ConstantInt::get(
        Type::getInt64Ty(cache->getContext()),
        newFunc->getParent()->getDataLayout().getTypeAllocSizeInBits(
            next->getType()) /
            8);
    cast<LoadInst>(next)->setMetadata(
        LLVMContext::MD_dereferenceable,
        MDNode::get(cache->getContext(),
                    {ConstantAsMetadata::get(byteSizeOfType)}));
    unsigned bsize = (unsigned)byteSizeOfType->getZExtValue();
    if ((bsize & (bsize - 1)) == 0) {
#if LLVM_VERSION_MAJOR >= 10
      cast<LoadInst>(next)->setAlignment(Align(bsize));
#else
      cast<LoadInst>(next)->setAlignment(bsize);
#endif
    }

    const auto &containedloops = sublimits[i].second;

    if (containedloops.size() > 0) {
      Value *idx = computeIndexOfChunk(inForwardPass, BuilderM, containedloops,
                                       (i == sublimits.size() - 1) ? ompOffset
                                                                   : nullptr);
      if (EfficientBoolCache && isi1 && i == 0)
        idx = BuilderM.CreateLShr(
            idx, ConstantInt::get(Type::getInt64Ty(newFunc->getContext()), 3));
      if (i == 0 && extraSize) {
        Value *es = lookupM(extraSize, BuilderM);
        assert(es);
        idx = BuilderM.CreateMul(idx, es, "", /*NUW*/ true, /*NSW*/ true);
      }
      next = BuilderM.CreateGEP(next, {idx});
      cast<GetElementPtrInst>(next)->setIsInBounds(true);
      if (storeInInstructionsMap && isa<AllocaInst>(cache))
        scopeInstructions[cast<AllocaInst>(cache)].push_back(
            cast<Instruction>(next));
    }
    assert(next->getType()->isPointerTy());
  }
  return next;
}

/// Given an allocation specified by the LimitContext ctx and cache, lookup the
/// underlying cached value.
Value *CacheUtility::lookupValueFromCache(bool inForwardPass,
                                          IRBuilder<> &BuilderM,
                                          LimitContext ctx, Value *cache,
                                          bool isi1, Value *extraSize,
                                          Value *extraOffset) {
  // Get the underlying cache pointer
  auto cptr = getCachePointer(inForwardPass, BuilderM, ctx, cache, isi1,
                              /*storeInInstructionsMap*/ false, extraSize);

  // Optionally apply the additional offset
  if (extraOffset) {
    cptr = BuilderM.CreateGEP(cptr, {extraOffset});
    cast<GetElementPtrInst>(cptr)->setIsInBounds(true);
  }

  // Retrieve the actual result
  auto result = BuilderM.CreateLoad(cptr);

  // Apply requisite invariant, alignment, etc
  if (ValueInvariantGroups.find(cache) == ValueInvariantGroups.end()) {
    MDNode *invgroup = MDNode::getDistinct(cache->getContext(), {});
    ValueInvariantGroups[cache] = invgroup;
  }
  CacheLookups.insert(result);
  result->setMetadata(LLVMContext::MD_invariant_group,
                      ValueInvariantGroups[cache]);
  ConstantInt *byteSizeOfType = ConstantInt::get(
      Type::getInt64Ty(cache->getContext()),
      newFunc->getParent()->getDataLayout().getTypeAllocSizeInBits(
          result->getType()) /
          8);
  unsigned bsize = (unsigned)byteSizeOfType->getZExtValue();
  if ((bsize & (bsize - 1)) == 0) {
#if LLVM_VERSION_MAJOR >= 10
    result->setAlignment(Align(bsize));
#else
    result->setAlignment(bsize);
#endif
  }

  // If using the efficient bool cache, do the corresponding
  // mask and shift to retrieve the actual value
  if (EfficientBoolCache && isi1) {
    if (auto gep = dyn_cast<GetElementPtrInst>(cptr)) {
      auto bo = cast<BinaryOperator>(*gep->idx_begin());
      assert(bo->getOpcode() == BinaryOperator::LShr);
      Value *res = BuilderM.CreateLShr(
          result,
          BuilderM.CreateAnd(
              BuilderM.CreateTrunc(bo->getOperand(0),
                                   Type::getInt8Ty(cache->getContext())),
              ConstantInt::get(Type::getInt8Ty(cache->getContext()), 7)));
      return BuilderM.CreateTrunc(res, Type::getInt1Ty(result->getContext()));
    }
  }
  return result;
}