//===- CacheUtility.cpp - Caching values in the forward pass for later use  -===//
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
// This file defines a base helper class CacheUtility that manages the cache
// of values from the forward pass for later use.
//
//===----------------------------------------------------------------------===//

#include "CacheUtility.h"
#include "FunctionUtils.h"

using namespace llvm;

/// Pack 8 bools together in a single byte
llvm::cl::opt<bool>
    EfficientBoolCache("enzyme_smallbool", cl::init(false), cl::Hidden,
                       cl::desc("Place 8 bools together in a single byte"));

CacheUtility::~CacheUtility(){}

// Create a new canonical induction variable of Type Ty for Loop L
// Return the variable and the increment instruction
static std::pair<PHINode *, Instruction *> insertNewCanonicalIV(Loop *L, Type *Ty) {
  assert(L);
  assert(Ty);

  BasicBlock *Header = L->getHeader();
  assert(Header);
  IRBuilder<> B(&Header->front());
  PHINode *CanonicalIV = B.CreatePHI(Ty, 1, "iv");

  B.SetInsertPoint(Header->getFirstNonPHIOrDbg());
  Instruction *inc = cast<Instruction>(
      B.CreateAdd(CanonicalIV, ConstantInt::get(CanonicalIV->getType(), 1),
                  "iv.next", /*NUW*/ true, /*NSW*/ true));

  for (BasicBlock *Pred : predecessors(Header)) {
    assert(Pred);
    if (L->contains(Pred)) {
      CanonicalIV->addIncoming(inc, Pred);
    } else {
      CanonicalIV->addIncoming(ConstantInt::get(CanonicalIV->getType(), 0),
                               Pred);
    }
  }
  return std::pair<PHINode *, Instruction *>(CanonicalIV, inc);
}

// Attempt to rewrite all phinode's in the loop in terms of the 
// induction variable
void removeRedundantIVs(const Loop *L, BasicBlock *Header,
                        BasicBlock *Preheader, PHINode *CanonicalIV,
                        MustExitScalarEvolution &SE, CacheUtility &gutils,
                        Instruction *increment,
                        const SmallVectorImpl<BasicBlock *> &&latches) {
  assert(Header);
  assert(CanonicalIV);

  SmallVector<Instruction *, 8> IVsToRemove;

  // This scope is necessary to ensure scevexpander cleans up before we erase
  // things
  {
    fake::SCEVExpander Exp(
        SE, Header->getParent()->getParent()->getDataLayout(), "enzyme");

    for (BasicBlock::iterator II = Header->begin(); isa<PHINode>(II); ++II) {
      PHINode *PN = cast<PHINode>(II);
      if (PN == CanonicalIV)
        continue;
      if (PN->getType()->isPointerTy())
        continue;
      if (!SE.isSCEVable(PN->getType()))
        continue;
      const SCEV *S = SE.getSCEV(PN);
      if (SE.getCouldNotCompute() == S)
        continue;
      Value *NewIV = Exp.expandCodeFor(S, S->getType(), CanonicalIV);
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
    gutils.erase(PN);
  }

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
  if (increment) {
    increment->moveAfter(CanonicalIV->getParent()->getFirstNonPHI());
    std::vector<Instruction *> toerase;
    for (auto use : CanonicalIV->users()) {
      auto bo = dyn_cast<BinaryOperator>(use);

      if (bo == nullptr)
        continue;
      if (bo->getOpcode() != BinaryOperator::Add)
        continue;
      if (use == increment)
        continue;

      Value *toadd = nullptr;
      if (bo->getOperand(0) == CanonicalIV) {
        toadd = bo->getOperand(1);
      } else {
        assert(bo->getOperand(1) == CanonicalIV);
        toadd = bo->getOperand(0);
      }
      if (auto ci = dyn_cast<ConstantInt>(toadd)) {
        if (!ci->isOne())
          continue;
        bo->replaceAllUsesWith(increment);
        toerase.push_back(bo);
      } else {
        continue;
      }
    }
    for (auto inst : toerase) {
      gutils.erase(inst);
    }

    if (latches.size() == 1 && isa<BranchInst>(latches[0]->getTerminator()) &&
        cast<BranchInst>(latches[0]->getTerminator())->isConditional())
      for (auto use : increment->users()) {
        if (auto cmp = dyn_cast<ICmpInst>(use)) {
          if (cast<BranchInst>(latches[0]->getTerminator())->getCondition() !=
              cmp)
            continue;

          // Force i+1 to be on LHS
          if (cmp->getOperand(0) != increment) {
            // Below also swaps predicate correctly
            cmp->swapOperands();
          }
          assert(cmp->getOperand(0) == increment);

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

  // Already canonicalized
  if (loopContexts.find(L) == loopContexts.end()) {

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

    auto pair = insertNewCanonicalIV(L, Type::getInt64Ty(BB->getContext()));
    PHINode *CanonicalIV = pair.first;
    assert(CanonicalIV);
    loopContexts[L].var = CanonicalIV;
    loopContexts[L].incvar = pair.second;
    removeRedundantIVs(L, loopContexts[L].header, loopContexts[L].preheader,
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
    {

      const SCEV *MayExitMaxBECount = nullptr;

      SmallVector<BasicBlock *, 8> ExitingBlocks;
      L->getExitingBlocks(ExitingBlocks);

      for (BasicBlock *ExitBB : ExitingBlocks) {
        assert(L->contains(ExitBB));
        auto EL = SE.computeExitLimit(L, ExitBB, /*AllowPredicates*/ true);

        if (MayExitMaxBECount != SE.getCouldNotCompute()) {
          if (!MayExitMaxBECount || EL.ExactNotTaken == SE.getCouldNotCompute())
            MayExitMaxBECount = EL.ExactNotTaken;
          else {
            if (MayExitMaxBECount != EL.ExactNotTaken) {
              llvm::errs() << "Missed cache optimization opportunity! could allocate max!\n";
              MayExitMaxBECount = SE.getCouldNotCompute();
              break;
            }

            MayExitMaxBECount = SE.getUMaxFromMismatchedTypes(MayExitMaxBECount,
                                                              EL.ExactNotTaken);
          }
        } else {
          MayExitMaxBECount = SE.getCouldNotCompute();
        }
      }
      if (ExitingBlocks.size() == 0) {
        MayExitMaxBECount = SE.getCouldNotCompute();
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

      fake::SCEVExpander Exp(SE, BB->getParent()->getParent()->getDataLayout(),
                             "enzyme");
      LimitVar = Exp.expandCodeFor(Limit, CanonicalIV->getType(),
                                   loopContexts[L].preheader->getTerminator());
      loopContexts[L].dynamic = false;
    } else {
      llvm::errs() << "SE could not compute loop limit of "
                   << L->getHeader()->getName() << " "
                   << L->getHeader()->getParent()->getName() << "\n";

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
            Limit->addIncoming(UndefValue::get(CanonicalIV->getType()),
                                 Pred);
          }
        }

        storeInstructionInCache(loopContexts[L].preheader, Limit,
                                       cast<AllocaInst>(LimitVar));
      }
      loopContexts[L].dynamic = true;
    }
    loopContexts[L].limit = LimitVar;
  }

  loopContext = loopContexts.find(L)->second;
  return true;
}

  /// Caching mechanism: creates a cache of type T in a scope given by ctx
  /// (where if ctx is in a loop there will be a corresponding number of slots)
  AllocaInst *CacheUtility::createCacheForScope(LimitContext ctx, Type *T, StringRef name,
                                  bool shouldFree, bool allocateInternal,
                                  Value *extraSize) {
    assert(ctx.Block);
    assert(T);

    auto sublimits = getSubLimits(ctx);

    /* goes from inner loop to outer loop*/
    std::vector<Type *> types = {T};
    bool isi1 = T->isIntegerTy() && cast<IntegerType>(T)->getBitWidth() == 1;
    if (EfficientBoolCache && isi1 && sublimits.size() != 0)
      types[0] = Type::getInt8Ty(T->getContext());
    for (size_t i=0; i<sublimits.size(); ++i) {
      types.push_back(PointerType::getUnqual(types.back()));
    }

    assert(inversionAllocs && "must be able to allocate inverted caches");
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

    for (int i = sublimits.size() - 1; i >= 0; i--) {
      const auto &containedloops = sublimits[i].second;

      Type *myType = types[i];

      ConstantInt *byteSizeOfType = ConstantInt::get(
          Type::getInt64Ty(T->getContext()),
          newFunc->getParent()->getDataLayout().getTypeAllocSizeInBits(myType) /
              8);

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
                              /*available*/ValueToValueMapTy(),
                              UnwrapMode::AttemptFullUnwrapWithLookup);
          assert(es);
          size = allocationBuilder.CreateMul(size, es, "", /*NUW*/ true,
                                             /*NSW*/ true);
        }

        StoreInst *storealloc = nullptr;
        if (!sublimits[i].second.back().first.dynamic) {
          auto firstallocation = CallInst::CreateMalloc(
              &allocationBuilder.GetInsertBlock()->back(), size->getType(),
              myType, byteSizeOfType, size, nullptr, name + "_malloccache");
          CallInst *malloccall = dyn_cast<CallInst>(firstallocation);
          if (malloccall == nullptr) {
            malloccall = cast<CallInst>(
                cast<Instruction>(firstallocation)->getOperand(0));
          }
          if (auto bi =
                  dyn_cast<BinaryOperator>(malloccall->getArgOperand(0))) {
            if ((bi->getOperand(0) == byteSizeOfType &&
                 bi->getOperand(1) == size) ||
                (bi->getOperand(1) == byteSizeOfType &&
                 bi->getOperand(0) == size))
              bi->setHasNoSignedWrap(true);
            bi->setHasNoUnsignedWrap(true);
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

          storealloc =
              allocationBuilder.CreateStore(firstallocation, storeInto);

          scopeAllocs[alloc].push_back(malloccall);
        } else {
          auto zerostore = allocationBuilder.CreateStore(
              ConstantPointerNull::get(PointerType::getUnqual(myType)),
              storeInto);
          scopeStores[alloc].push_back(zerostore);

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
        }

        if (invariantGroups.find(std::make_pair((Value *)alloc, i)) ==
            invariantGroups.end()) {
          MDNode *invgroup = MDNode::getDistinct(alloc->getContext(), {});
          invariantGroups[std::make_pair((Value *)alloc, i)] = invgroup;
        }
        storealloc->setMetadata(
            LLVMContext::MD_invariant_group,
            invariantGroups[std::make_pair((Value *)alloc, i)]);
        unsigned bsize = (unsigned)byteSizeOfType->getZExtValue();
        if ((bsize & (bsize - 1)) == 0) {
#if LLVM_VERSION_MAJOR >= 10
          storealloc->setAlignment(Align(bsize));
#else
          storealloc->setAlignment(bsize);
#endif
        }
        scopeStores[alloc].push_back(storealloc);
      }

      if (shouldFree) {
          freeCache(containedloops.back().first.preheader, sublimits, i, alloc, byteSizeOfType, storeInto);
      }

      if (i != 0) {
        IRBuilder<> v(&sublimits[i - 1].second.back().first.preheader->back());

        SmallVector<Value *, 3> indices;
        SmallVector<Value *, 3> limits;
        ValueToValueMapTy available;
        for (auto riter = containedloops.rbegin(), rend = containedloops.rend();
             riter != rend; ++riter) {
          // Only include dynamic index on last iteration (== skip dynamic index
          // on non-last iterations)
          // if (i != 0 && riter+1 == rend) break;

          const auto &idx = riter->first;
          Value *var = idx.var;
          if (var == nullptr)
            var = ConstantInt::get(idx.limit->getType(), 0);
          indices.push_back(var);
          if (idx.var)
            available[var] = var;

          Value *lim = unwrapM(riter->second, v, available,
                               UnwrapMode::AttemptFullUnwrapWithLookup);
          assert(lim);
          if (limits.size() == 0) {
            limits.push_back(lim);
          } else {
            limits.push_back(v.CreateMul(lim, limits.back(), "", /*NUW*/ true,
                                         /*NSW*/ true));
          }
        }

        assert(indices.size() > 0);

        Value *idx = indices[0];
        for (unsigned ind = 1; ind < indices.size(); ++ind) {
          idx = v.CreateAdd(idx,
                            v.CreateMul(indices[ind], limits[ind - 1], "",
                                        /*NUW*/ true, /*NSW*/ true),
                            "", /*NUW*/ true, /*NSW*/ true);
        }

        storeInto = v.CreateGEP(v.CreateLoad(storeInto), idx);
        cast<GetElementPtrInst>(storeInto)->setIsInBounds(true);
      }
    }
    return alloc;
  }


//! returns true indices
CacheUtility::SubLimitType CacheUtility::getSubLimits(LimitContext ctx) {

// The following ficticious context is part of a disabled
// experimental mechanism for merging stores into a unified memcpy
{
    LoopContext idx;
    if (ctx.Experimental) {
    auto subctx = ctx.Block;
    auto zero =
        ConstantInt::get(Type::getInt64Ty(newFunc->getContext()), 0);
    auto one = ConstantInt::get(Type::getInt64Ty(newFunc->getContext()), 1);
    idx.var = nullptr;
    idx.incvar = nullptr;
    idx.antivaralloc = nullptr;
    idx.limit = zero;
    idx.header = subctx;
    idx.preheader = subctx;
    idx.dynamic = false;
    idx.parent = nullptr;
    idx.exitBlocks = {};
    SubLimitType sublimits;
    sublimits.push_back({one, {{idx, one}}});
    return sublimits;
    }
}

std::vector<LoopContext> contexts;
for (BasicBlock *blk = ctx.Block; blk != nullptr;) {
    LoopContext idx;
    if (!getContext(blk, idx)) {
    break;
    }
    contexts.emplace_back(idx);
    blk = idx.preheader;
}

std::vector<BasicBlock *> allocationPreheaders(contexts.size(), nullptr);
std::vector<Value *> limits(contexts.size(), nullptr);
for (int i = contexts.size() - 1; i >= 0; --i) {
    if ((unsigned)i == contexts.size() - 1) {
    allocationPreheaders[i] = contexts[i].preheader;
    } else if (contexts[i].dynamic) {
    allocationPreheaders[i] = contexts[i].preheader;
    } else {
    allocationPreheaders[i] = allocationPreheaders[i + 1];
    }

    if (contexts[i].dynamic) {
    limits[i] = ConstantInt::get(Type::getInt64Ty(ctx.Block->getContext()), 1);
    } else {
    ValueToValueMapTy prevMap;

    for (int j = contexts.size() - 1;; --j) {
        if (allocationPreheaders[i] == contexts[j].preheader)
        break;
        prevMap[contexts[j].var] = contexts[j].var;
    }

    IRBuilder<> allocationBuilder(&allocationPreheaders[i]->back());
    Value *limitMinus1 = nullptr;

    // TODO ensure unwrapM considers the legality of illegal caching / etc
    //   legalRecompute does not fulfill this need as its whether its legal
    //   at a certain location, where as legalRecompute specifies it being
    //   recomputable anywhere
    // if (legalRecompute(contexts[i].limit, prevMap)) {
    limitMinus1 = unwrapM(contexts[i].limit, allocationBuilder, prevMap,
                            UnwrapMode::AttemptFullUnwrap);

    // We have a loop with static bounds, but whose limit is not available
    // to be computed at the current loop preheader (such as the innermost
    // loop of triangular iteration domain) Handle this case like a dynamic
    // loop
    if (limitMinus1 == nullptr) {
        allocationPreheaders[i] = contexts[i].preheader;
        allocationBuilder.SetInsertPoint(&allocationPreheaders[i]->back());
        limitMinus1 = unwrapM(contexts[i].limit, allocationBuilder, prevMap,
                            UnwrapMode::AttemptFullUnwrap);
    }
    assert(limitMinus1 != nullptr);
    static std::map<std::pair<Value *, BasicBlock *>, Value *> limitCache;
    auto cidx = std::make_pair(limitMinus1, allocationPreheaders[i]);
    if (limitCache.find(cidx) == limitCache.end()) {
        limitCache[cidx] = allocationBuilder.CreateNUWAdd(
            limitMinus1, ConstantInt::get(limitMinus1->getType(), 1));
    }
    limits[i] = limitCache[cidx];
    }
}

SubLimitType sublimits;

Value *size = nullptr;
std::vector<std::pair<LoopContext, Value *>> lims;
for (unsigned i = 0; i < contexts.size(); ++i) {
    IRBuilder<> allocationBuilder(&allocationPreheaders[i]->back());
    lims.push_back(std::make_pair(contexts[i], limits[i]));
    if (size == nullptr) {
    size = limits[i];
    } else {
    static std::map<std::pair<Value *, BasicBlock *>, Value *> sizeCache;
    auto cidx = std::make_pair(size, allocationPreheaders[i]);
    if (sizeCache.find(cidx) == sizeCache.end()) {
        sizeCache[cidx] = allocationBuilder.CreateNUWMul(size, limits[i]);
    }
    size = sizeCache[cidx];
    }

    // We are now starting a new allocation context
    if ((i + 1 < contexts.size()) &&
        (allocationPreheaders[i] != allocationPreheaders[i + 1])) {
    sublimits.push_back(std::make_pair(size, lims));
    size = nullptr;
    lims.clear();
    }
}

if (size != nullptr) {
    sublimits.push_back(std::make_pair(size, lims));
    lims.clear();
}
return sublimits;
}

void CacheUtility::storeInstructionInCache(LimitContext ctx, IRBuilder<> &BuilderM,
                               Value *val, AllocaInst *cache) {
    assert(BuilderM.GetInsertBlock()->getParent() == newFunc);
    if (auto inst = dyn_cast<Instruction>(val))
      assert(inst->getParent()->getParent() == newFunc);
    IRBuilder<> v(BuilderM.GetInsertBlock());
    v.SetInsertPoint(BuilderM.GetInsertBlock(), BuilderM.GetInsertPoint());
    v.setFastMathFlags(getFast());

    // Note for dynamic loops where the allocation is stored somewhere inside
    // the loop,
    // we must ensure that we load the allocation after the store ensuring
    // memory exists to simplify things and ensure we always store after a
    // potential realloc occurs in this loop This is okay as there should be no
    // load to the cache in the same block where this instruction is defined
    // (since we will just use this instruction)
    // TODO check that the store is actually aliasing/related
    if (BuilderM.GetInsertPoint() != BuilderM.GetInsertBlock()->end())
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
    bool isi1 = val->getType()->isIntegerTy() &&
                cast<IntegerType>(val->getType())->getBitWidth() == 1;
    Value *loc =
        getCachePointer(/*inForwardPass*/true, v, ctx, cache, isi1, /*storeinstorecache*/ true);

    Value *tostore = val;
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

    if (tostore == val &&
        valueInvariantGroups.find(cache) == valueInvariantGroups.end()) {
      MDNode *invgroup = MDNode::getDistinct(cache->getContext(), {});
      valueInvariantGroups[cache] = invgroup;
    }
    storeinst->setMetadata(LLVMContext::MD_invariant_group,
                           valueInvariantGroups[cache]);
    ConstantInt *byteSizeOfType = ConstantInt::get(
        Type::getInt64Ty(cache->getContext()),
        ctx.Block->getParent()->getParent()->getDataLayout().getTypeAllocSizeInBits(
            val->getType()) /
            8);
    unsigned bsize = (unsigned)byteSizeOfType->getZExtValue();
    if ((bsize & (bsize - 1)) == 0) {
#if LLVM_VERSION_MAJOR >= 10
      storeinst->setAlignment(Align(bsize));
#else
      storeinst->setAlignment(bsize);
#endif
    }
    scopeStores[cache].push_back(storeinst);
}

void CacheUtility::storeInstructionInCache(LimitContext ctx, llvm::Instruction *inst,
                            llvm::AllocaInst *cache) {
    assert(ctx.Block);
    assert(inst);
    assert(cache);

    IRBuilder<> v(inst->getParent());

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

Value *CacheUtility::getCachePointer(bool inForwardPass, IRBuilder<> &BuilderM, LimitContext ctx, Value *cache,
                         bool isi1, bool storeInStoresMap,
                         Value *extraSize) {
    assert(ctx.Block);
    assert(cache);

    auto sublimits = getSubLimits(ctx);

    ValueToValueMapTy available;

    Value *next = cache;
    assert(next->getType()->isPointerTy());
    for (int i = sublimits.size() - 1; i >= 0; i--) {
      next = BuilderM.CreateLoad(next);
      if (storeInStoresMap && isa<AllocaInst>(cache))
        scopeStores[cast<AllocaInst>(cache)].push_back(next);

      if (!next->getType()->isPointerTy()) {
        llvm::errs() << *newFunc << "\n";
        llvm::errs() << "cache: " << *cache << "\n";
        llvm::errs() << "next: " << *next << "\n";
      }
      assert(next->getType()->isPointerTy());
      if (invariantGroups.find(std::make_pair(cache, i)) ==
          invariantGroups.end()) {
        MDNode *invgroup = MDNode::getDistinct(cache->getContext(), {});
        invariantGroups[std::make_pair(cache, i)] = invgroup;
      }
      cast<LoadInst>(next)->setMetadata(
          LLVMContext::MD_invariant_group,
          invariantGroups[std::make_pair(cache, i)]);
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

      SmallVector<Value *, 3> indices;
      SmallVector<Value *, 3> limits;
      for (auto riter = containedloops.begin(), rend = containedloops.end();
           riter != rend; ++riter) {
        // Only include dynamic index on last iteration (== skip dynamic index
        // on non-last iterations)
        // if (i != 0 && riter+1 == rend) break;
        const auto &idx = riter->first;
        if (riter + 1 != rend) {
          assert(!idx.dynamic);
        }
        if (!inForwardPass) {
          Value *av;
          if (idx.var)
            av = BuilderM.CreateLoad(idx.antivaralloc);
          else
            av = ConstantInt::get(idx.limit->getType(), 0);

          indices.push_back(av);
          if (idx.var)
            available[idx.var] = av;
        } else {
          assert(idx.limit);
          indices.push_back(
              idx.var ? (Value *)idx.var
                      : (Value *)ConstantInt::get(idx.limit->getType(), 0));
          if (idx.var)
            available[idx.var] = idx.var;
        }

        Value *lim = unwrapM(riter->second, BuilderM, available,
                             UnwrapMode::AttemptFullUnwrapWithLookup);
        assert(lim);
        if (limits.size() == 0) {
          limits.push_back(lim);
        } else {
          limits.push_back(BuilderM.CreateMul(lim, limits.back(), "",
                                              /*NUW*/ true, /*NSW*/ true));
        }
      }

      if (indices.size() > 0) {
        Value *idx = indices[0];
        for (unsigned ind = 1; ind < indices.size(); ++ind) {
          idx = BuilderM.CreateAdd(
              idx,
              BuilderM.CreateMul(indices[ind], limits[ind - 1], "",
                                 /*NUW*/ true, /*NSW*/ true),
              "", /*NUW*/ true, /*NSW*/ true);
        }
        if (EfficientBoolCache && isi1 && i == 0)
          idx = BuilderM.CreateLShr(
              idx,
              ConstantInt::get(Type::getInt64Ty(newFunc->getContext()), 3));
        if (i == 0 && extraSize) {
          Value *es = lookupM(extraSize, BuilderM);
          assert(es);
          idx = BuilderM.CreateMul(idx, es, "", /*NUW*/ true, /*NSW*/ true);
        }
        next = BuilderM.CreateGEP(next, {idx});
        cast<GetElementPtrInst>(next)->setIsInBounds(true);
        if (storeInStoresMap && isa<AllocaInst>(cache))
          scopeStores[cast<AllocaInst>(cache)].push_back(next);
      }
      assert(next->getType()->isPointerTy());
    }
    return next;
  }


Value *CacheUtility::lookupValueFromCache(bool inForwardPass, IRBuilder<> &BuilderM, LimitContext ctx,
                              Value *cache, bool isi1,
                              Value *extraSize,
                              Value *extraOffset) {
    auto cptr = getCachePointer(inForwardPass, BuilderM, ctx, cache, isi1,
                                /*storeInStoresMap*/ false, extraSize);
    if (extraOffset) {
      cptr = BuilderM.CreateGEP(cptr, {extraOffset});
      cast<GetElementPtrInst>(cptr)->setIsInBounds(true);
    }
    auto result = BuilderM.CreateLoad(cptr);

    if (valueInvariantGroups.find(cache) == valueInvariantGroups.end()) {
      MDNode *invgroup = MDNode::getDistinct(cache->getContext(), {});
      valueInvariantGroups[cache] = invgroup;
    }
    result->setMetadata("enzyme_fromcache",
                        MDNode::get(result->getContext(), {}));
    result->setMetadata(LLVMContext::MD_invariant_group,
                        valueInvariantGroups[cache]);
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