//===- GradientUtils.h - Helper class and utilities for AD       ---------===//
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
// This file declares two helper classes GradientUtils and subclass
// DiffeGradientUtils. These classes contain utilities for managing the cache,
// recomputing statements, and in the case of DiffeGradientUtils, managing
// adjoint values and shadow pointers.
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_GUTILS_H_
#define ENZYME_GUTILS_H_

#include <algorithm>
#include <deque>
#include <map>

#include <llvm/Config/llvm-config.h>

#include "ActivityAnalysis.h"
#include "SCEV/ScalarEvolutionExpander.h"
#include "Utils.h"

#include "llvm/ADT/SmallVector.h"

#include "llvm/IR/Dominators.h"

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"

#include "llvm/Analysis/PostDominators.h"
#include "llvm/Support/Casting.h"

#include "llvm/Transforms/Utils/ValueMapper.h"

#include "llvm/Support/ErrorHandling.h"

#include "ActivityAnalysis.h"
#include "EnzymeLogic.h"

using namespace llvm;

enum class DerivativeMode { Forward, Reverse, Both };

static inline std::string to_string(DerivativeMode mode) {
  switch (mode) {
  case DerivativeMode::Forward:
    return "Forward";
  case DerivativeMode::Reverse:
    return "Reverse";
  case DerivativeMode::Both:
    return "Both";
  }
  llvm_unreachable("illegal derivative mode");
}

extern llvm::cl::opt<bool> efficientBoolCache;

enum class AugmentedStruct;
typedef struct {
  PHINode *var;
  Instruction *incvar;
  AllocaInst *antivaralloc;
  BasicBlock *latchMerge;
  BasicBlock *header;
  BasicBlock *preheader;
  bool dynamic;
  // limit is last value, iters is number of iters (thus iters = limit + 1)
  Value *limit;
  SmallPtrSet<BasicBlock *, 8> exitBlocks;
  Loop *parent;
} LoopContext;

enum class UnwrapMode {
  LegalFullUnwrap,
  AttemptFullUnwrapWithLookup,
  AttemptFullUnwrap,
  AttemptSingleUnwrap,
};

static inline bool operator==(const LoopContext &lhs, const LoopContext &rhs) {
  return lhs.parent == rhs.parent;
}

class MyScalarEvolution : public ScalarEvolution {
public:
  using ScalarEvolution::ScalarEvolution;

  ScalarEvolution::ExitLimit computeExitLimit(const Loop *L,
                                              BasicBlock *ExitingBlock,
                                              bool AllowPredicates);
  ScalarEvolution::ExitLimit
  computeExitLimitFromCond(const Loop *L, Value *ExitCond, bool ExitIfTrue,
                           bool ControlsExit, bool AllowPredicates);

  ScalarEvolution::ExitLimit
  computeExitLimitFromCondCached(ExitLimitCacheTy &Cache, const Loop *L,
                                 Value *ExitCond, bool ExitIfTrue,
                                 bool ControlsExit, bool AllowPredicates);

  ScalarEvolution::ExitLimit
  computeExitLimitFromCondImpl(ExitLimitCacheTy &Cache, const Loop *L,
                               Value *ExitCond, bool ExitIfTrue,
                               bool ControlsExit, bool AllowPredicates);

  ScalarEvolution::ExitLimit
  computeExitLimitFromICmp(const Loop *L, ICmpInst *ExitCond, bool ExitIfTrue,
                           bool ControlsExit, bool AllowPredicates = false);

  ScalarEvolution::ExitLimit howManyLessThans(const SCEV *LHS, const SCEV *RHS,
                                              const Loop *L, bool IsSigned,
                                              bool ControlsExit,
                                              bool AllowPredicates);
};

class GradientUtils {
public:
  DerivativeMode mode;
  llvm::Function *newFunc;
  llvm::Function *oldFunc;
  ValueToValueMapTy invertedPointers;
  DominatorTree DT;
  DominatorTree OrigDT;
  PostDominatorTree OrigPDT;
  ActivityAnalyzer ATA;
  LoopInfo OrigLI;
  LoopInfo LI;
  AssumptionCache AC;
  MyScalarEvolution SE;
  std::map<Loop *, LoopContext> loopContexts;
  SmallVector<BasicBlock *, 12> originalBlocks;
  ValueMap<BasicBlock *, BasicBlock *> reverseBlocks;
  BasicBlock *inversionAllocs;
  std::map<Value *, std::pair<AllocaInst *, /*ctx*/ BasicBlock *>> scopeMap;
  std::map<AllocaInst *, std::set<CallInst *>> scopeFrees;
  std::map<AllocaInst *, std::vector<CallInst *>> scopeAllocs;
  std::map<AllocaInst *, std::vector<Value *>> scopeStores;
  SmallVector<PHINode *, 4> fictiousPHIs;
  ValueToValueMapTy originalToNewFn;

  const std::map<Instruction *, bool> *can_modref_map;

  Value *getNewIfOriginal(Value *originst) const {
    assert(originst);
    auto f = originalToNewFn.find(originst);
    if (f == originalToNewFn.end()) {
      return originst;
    }
    assert(f != originalToNewFn.end());
    if (f->second == nullptr) {
      llvm::errs() << *oldFunc << "\n";
      llvm::errs() << *newFunc << "\n";
      llvm::errs() << *originst << "\n";
    }
    assert(f->second);
    return f->second;
  }

  Value *getNewFromOriginal(const Value *originst) const {
    assert(originst);
    auto f = originalToNewFn.find(originst);
    if (f == originalToNewFn.end()) {
      llvm::errs() << *oldFunc << "\n";
      llvm::errs() << *newFunc << "\n";
      dumpMap(originalToNewFn, [&](const Value *const &v) -> bool {
        if (isa<Instruction>(originst))
          return isa<Instruction>(v);
        if (isa<BasicBlock>(originst))
          return isa<BasicBlock>(v);
        if (isa<Function>(originst))
          return isa<Function>(v);
        if (isa<Argument>(originst))
          return isa<Argument>(v);
        if (isa<Constant>(originst))
          return isa<Constant>(v);
        return true;
      });
      llvm::errs() << *originst << "\n";
    }
    assert(f != originalToNewFn.end());
    if (f->second == nullptr) {
      llvm::errs() << *oldFunc << "\n";
      llvm::errs() << *newFunc << "\n";
      llvm::errs() << *originst << "\n";
    }
    assert(f->second);
    return f->second;
  }
  Instruction *getNewFromOriginal(const Instruction *newinst) const {
    return cast<Instruction>(getNewFromOriginal((Value *)newinst));
  }

  Value *hasUninverted(const Value *inverted) const {
    for (auto v : invertedPointers) {
      if (v.second == inverted)
        return const_cast<Value *>(v.first);
    }
    return nullptr;
  }

  Value *isOriginal(const Value *newinst) const {
    if (isa<Constant>(newinst) || isa<UndefValue>(newinst))
      return const_cast<Value *>(newinst);
    if (auto arg = dyn_cast<Argument>(newinst)) {
      assert(arg->getParent() == newFunc);
    }
    if (auto inst = dyn_cast<Instruction>(newinst)) {
      assert(inst->getParent()->getParent() == newFunc);
    }
    for (auto v : originalToNewFn) {
      if (v.second == newinst)
        return const_cast<Value *>(v.first);
    }
    return nullptr;
  }

  Instruction *isOriginal(const Instruction *newinst) const {
    return cast_or_null<Instruction>(isOriginal((const Value *)newinst));
  }

  template <typename T> T *isOriginalT(T *newinst) const {
    return cast_or_null<T>(isOriginal((const Value *)newinst));
  }

private:
  SmallVector<Value *, 4> addedTapeVals;
  unsigned tapeidx;
  Value *tape;

  std::map<std::pair<Value *, int>, MDNode *> invariantGroups;
  std::map<Value *, MDNode *> valueInvariantGroups;
  std::map<std::pair<Value *, BasicBlock *>, Value *> unwrap_cache;
  std::map<std::pair<Value *, BasicBlock *>, Value *> lookup_cache;

public:
  bool legalRecompute(const Value *val,
                      const ValueToValueMapTy &available) const;
  bool shouldRecompute(const Value *val,
                       const ValueToValueMapTy &available) const;

  void replaceAWithB(Value *A, Value *B) {
    for (unsigned i = 0; i < addedTapeVals.size(); ++i) {
      if (addedTapeVals[i] == A) {
        addedTapeVals[i] = B;
      }
    }

    if (scopeMap.find(A) != scopeMap.end()) {
      scopeMap[B] = scopeMap[A];
      scopeMap.erase(A);
    }
    if (invertedPointers.find(A) != invertedPointers.end()) {
      invertedPointers[B] = invertedPointers[A];
      invertedPointers.erase(A);
    }
    if (auto orig = isOriginal(A)) {
      originalToNewFn[orig] = B;
    }

    A->replaceAllUsesWith(B);
  }

  void erase(Instruction *I) {
    assert(I);
    invertedPointers.erase(I);
    //constants.erase(I);
    //constant_values.erase(I);
    //nonconstant.erase(I);
    //nonconstant_values.erase(I);
    if (scopeMap.find(I) != scopeMap.end()) {
      scopeFrees.erase(scopeMap[I].first);
      scopeAllocs.erase(scopeMap[I].first);
      scopeStores.erase(scopeMap[I].first);
    }
    if (auto ai = dyn_cast<AllocaInst>(I)) {
      scopeFrees.erase(ai);
      scopeAllocs.erase(ai);
      scopeStores.erase(ai);
    }
    scopeMap.erase(I);
    SE.eraseValueFromMap(I);
    originalToNewFn.erase(I);
  eraser:
    for (auto v : originalToNewFn) {
      if (v.second == I) {
        originalToNewFn.erase(v.first);
        goto eraser;
      }
    }
    for (auto v : scopeMap) {
      if (v.second.first == I) {
        llvm::errs() << *oldFunc << "\n";
        llvm::errs() << *newFunc << "\n";
        dumpScope();
        llvm::errs() << *v.first << "\n";
        llvm::errs() << *I << "\n";
        assert(0 && "erasing something in scope map");
      }
    }
    if (auto ci = dyn_cast<CallInst>(I))
      for (auto v : scopeFrees) {
        if (v.second.count(ci)) {
          llvm::errs() << *oldFunc << "\n";
          llvm::errs() << *newFunc << "\n";
          llvm::errs() << *v.first << "\n";
          llvm::errs() << *I << "\n";
          assert(0 && "erasing something in scopeFrees map");
        }
      }
    if (auto ci = dyn_cast<CallInst>(I))
      for (auto v : scopeAllocs) {
        if (std::find(v.second.begin(), v.second.end(), ci) != v.second.end()) {
          llvm::errs() << *oldFunc << "\n";
          llvm::errs() << *newFunc << "\n";
          llvm::errs() << *v.first << "\n";
          llvm::errs() << *I << "\n";
          assert(0 && "erasing something in scopeAllocs map");
        }
      }
    for (auto v : scopeStores) {
      if (std::find(v.second.begin(), v.second.end(), I) != v.second.end()) {
        llvm::errs() << *oldFunc << "\n";
        llvm::errs() << *newFunc << "\n";
        llvm::errs() << *v.first << "\n";
        llvm::errs() << *I << "\n";
        assert(0 && "erasing something in scopeStores map");
      }
    }
    for (auto v : invertedPointers) {
      if (v.second == I) {
        llvm::errs() << *oldFunc << "\n";
        llvm::errs() << *newFunc << "\n";
        dumpPointers();
        llvm::errs() << *v.first << "\n";
        llvm::errs() << *I << "\n";
        assert(0 && "erasing something in invertedPointers map");
      }
    }

    {
      std::vector<std::pair<Value *, BasicBlock *>> unwrap_cache_pairs;
      for (auto &a : unwrap_cache) {
        if (a.second == I) {
          unwrap_cache_pairs.push_back(a.first);
        }
        if (a.first.first == I) {
          unwrap_cache_pairs.push_back(a.first);
        }
      }
      for (auto a : unwrap_cache_pairs) {
        unwrap_cache.erase(a);
      }
    }

    {
      std::vector<std::pair<Value *, BasicBlock *>> lookup_cache_pairs;
      for (auto &a : lookup_cache) {
        if (a.second == I) {
          lookup_cache_pairs.push_back(a.first);
        }
        if (a.first.first == I) {
          lookup_cache_pairs.push_back(a.first);
        }
      }
      for (auto a : lookup_cache_pairs) {
        lookup_cache.erase(a);
      }
    }

    if (!I->use_empty()) {
      llvm::errs() << *oldFunc << "\n";
      llvm::errs() << *newFunc << "\n";
      llvm::errs() << *I << "\n";
    }
    assert(I->use_empty());
    I->eraseFromParent();
  }
  // TODO consider invariant group and/or valueInvariant group

  void setTape(Value *newtape) {
    assert(tape == nullptr);
    assert(newtape != nullptr);
    assert(tapeidx == 0);
    assert(addedTapeVals.size() == 0);
    tape = newtape;
  }

  void dumpPointers() {
    llvm::errs() << "invertedPointers:\n";
    for (auto a : invertedPointers) {
      llvm::errs() << "   invertedPointers[" << *a.first << "] = " << *a.second
                   << "\n";
    }
    llvm::errs() << "end invertedPointers\n";
  }

  void dumpScope() {
    llvm::errs() << "scope:\n";
    for (auto a : scopeMap) {
      llvm::errs() << "   scopeMap[" << *a.first << "] = " << *a.second.first
                   << " ctx:" << a.second.second->getName() << "\n";
    }
    llvm::errs() << "end scope\n";
  }

  Value *createAntiMalloc(CallInst *orig, unsigned idx) {
    assert(orig->getParent()->getParent() == oldFunc);
    PHINode *placeholder = cast<PHINode>(invertedPointers[orig]);

    assert(placeholder->getParent()->getParent() == newFunc);
    placeholder->setName("");
    IRBuilder<> bb(placeholder);

    SmallVector<Value *, 8> args;
    for (unsigned i = 0; i < orig->getNumArgOperands(); ++i) {
      args.push_back(getNewFromOriginal(orig->getArgOperand(i)));
    }
    Value *anti =
        bb.CreateCall(orig->getCalledFunction(), args, orig->getName() + "'mi");
    cast<CallInst>(anti)->setAttributes(orig->getAttributes());
    cast<CallInst>(anti)->setCallingConv(orig->getCallingConv());
    cast<CallInst>(anti)->setTailCallKind(orig->getTailCallKind());
    cast<CallInst>(anti)->setDebugLoc(orig->getDebugLoc());
    cast<CallInst>(anti)->addAttribute(AttributeList::ReturnIndex,
                                       Attribute::NoAlias);
    cast<CallInst>(anti)->addAttribute(AttributeList::ReturnIndex,
                                       Attribute::NonNull);

    unsigned derefBytes = 0;
    if (orig->getCalledFunction()->getName() == "malloc" ||
        orig->getCalledFunction()->getName() == "_Znwm") {
      if (auto ci = dyn_cast<ConstantInt>(args[0])) {
        derefBytes = ci->getLimitedValue();
        cast<CallInst>(anti)->addDereferenceableAttr(
            llvm::AttributeList::ReturnIndex, ci->getLimitedValue());
        cast<CallInst>(anti)->addDereferenceableOrNullAttr(
            llvm::AttributeList::ReturnIndex, ci->getLimitedValue());
        CallInst *cal = cast<CallInst>(getNewFromOriginal(orig));
        cal->addDereferenceableAttr(llvm::AttributeList::ReturnIndex,
                                    ci->getLimitedValue());
        cal->addDereferenceableOrNullAttr(llvm::AttributeList::ReturnIndex,
                                          ci->getLimitedValue());
        cal->addAttribute(AttributeList::ReturnIndex, Attribute::NoAlias);
        cal->addAttribute(AttributeList::ReturnIndex, Attribute::NonNull);
      }
    }

    invertedPointers[orig] = anti;
    // assert(placeholder != anti);
    bb.SetInsertPoint(placeholder->getNextNode());
    replaceAWithB(placeholder, anti);
    erase(placeholder);

    anti = cacheForReverse(bb, anti, idx);
    invertedPointers[orig] = anti;

    if (tape == nullptr) {
      auto dst_arg =
          bb.CreateBitCast(anti, Type::getInt8PtrTy(orig->getContext()));
      auto val_arg = ConstantInt::get(Type::getInt8Ty(orig->getContext()), 0);
      auto len_arg =
          bb.CreateZExtOrTrunc(args[0], Type::getInt64Ty(orig->getContext()));
      auto volatile_arg = ConstantInt::getFalse(orig->getContext());

#if LLVM_VERSION_MAJOR == 6
      auto align_arg =
          ConstantInt::get(Type::getInt32Ty(orig->getContext()), 0);
      Value *nargs[] = {dst_arg, val_arg, len_arg, align_arg, volatile_arg};
#else
      Value *nargs[] = {dst_arg, val_arg, len_arg, volatile_arg};
#endif

      Type *tys[] = {dst_arg->getType(), len_arg->getType()};

      auto memset = cast<CallInst>(
          bb.CreateCall(Intrinsic::getDeclaration(newFunc->getParent(),
                                                  Intrinsic::memset, tys),
                        nargs));
      // memset->addParamAttr(0, Attribute::getWithAlignment(Context,
      // inst->getAlignment()));
      memset->addParamAttr(0, Attribute::NonNull);
      if (derefBytes) {
        memset->addDereferenceableAttr(llvm::AttributeList::FirstArgIndex,
                                       derefBytes);
        memset->addDereferenceableOrNullAttr(llvm::AttributeList::FirstArgIndex,
                                             derefBytes);
      }
    }

    return anti;
  }

  int getIndex(std::pair<Instruction *, CacheType> idx,
               std::map<std::pair<Instruction *, CacheType>, int> &mapping) {
    if (tape) {
      if (mapping.find(idx) == mapping.end()) {
        llvm::errs() << "oldFunc: " << *oldFunc << "\n";
        llvm::errs() << "newFunc: " << *newFunc << "\n";
        llvm::errs() << " <mapping>\n";
        for (auto &p : mapping) {
          llvm::errs() << "   idx: " << *p.first.first << ", " << p.first.second
                       << " pos=" << p.second << "\n";
        }
        llvm::errs() << " </mapping>\n";

        if (mapping.find(idx) == mapping.end()) {
          llvm::errs() << "idx: " << *idx.first << ", " << idx.second << "\n";
          assert(0 && "could not find index in mapping");
        }
      }
      return mapping[idx];
    } else {
      if (mapping.find(idx) != mapping.end()) {
        return mapping[idx];
      }
      mapping[idx] = tapeidx;
      ++tapeidx;
      return mapping[idx];
    }
  }

  Value *cacheForReverse(IRBuilder<> &BuilderQ, Value *malloc, int idx) {
    assert(BuilderQ.GetInsertBlock()->getParent() == newFunc);

    if (tape) {
      if (idx >= 0 && !tape->getType()->isStructTy()) {
        llvm::errs() << "cacheForReverse incorrect tape type: " << *tape
                     << " idx: " << idx << "\n";
      }
      assert(idx < 0 || tape->getType()->isStructTy());
      if (idx >= 0 && (unsigned)idx >=
                          cast<StructType>(tape->getType())->getNumElements()) {
        llvm::errs() << "oldFunc: " << *oldFunc << "\n";
        llvm::errs() << "newFunc: " << *newFunc << "\n";
        if (malloc)
          llvm::errs() << "malloc: " << *malloc << "\n";
        llvm::errs() << "tape: " << *tape << "\n";
        llvm::errs() << "idx: " << idx << "\n";
      }
      assert(idx < 0 ||
             (unsigned)idx <
                 cast<StructType>(tape->getType())->getNumElements());
      Value *ret = (idx < 0) ? tape
                             : cast<Instruction>(BuilderQ.CreateExtractValue(
                                   tape, {(unsigned)idx}));

      if (ret->getType()->isEmptyTy()) {
        if (auto inst = dyn_cast_or_null<Instruction>(malloc)) {
          if (inst->getType() != ret->getType()) {
            llvm::errs() << "oldFunc: " << *oldFunc << "\n";
            llvm::errs() << "newFunc: " << *newFunc << "\n";
            llvm::errs() << "inst==malloc: " << *inst << "\n";
            llvm::errs() << "ret: " << *ret << "\n";
          }
          assert(inst->getType() == ret->getType());
          inst->replaceAllUsesWith(UndefValue::get(ret->getType()));
          erase(inst);
        }
        Type *retType = ret->getType();
        if (auto ri = dyn_cast<Instruction>(ret))
          erase(ri);
        return UndefValue::get(retType);
      }

      BasicBlock *ctx = BuilderQ.GetInsertBlock();
      if (auto inst = dyn_cast<Instruction>(malloc))
        ctx = inst->getParent();
      auto found = scopeMap.find(malloc);
      if (found != scopeMap.end()) {
        ctx = found->second.second;
      }

      bool inLoop;
      if ((size_t)ctx & 1) {
        inLoop = true;
        ctx = (BasicBlock *)((size_t)ctx ^ 1);
      } else {
        LoopContext lc;
        inLoop = getContext(ctx, lc);
      }

      if (!inLoop) {
        if (malloc)
          ret->setName(malloc->getName() + "_fromtape");
      } else {
        if (auto ri = dyn_cast<Instruction>(ret))
          erase(ri);
        IRBuilder<> entryBuilder(inversionAllocs);
        entryBuilder.setFastMathFlags(getFast());
        ret = (idx < 0) ? tape
                        : cast<Instruction>(entryBuilder.CreateExtractValue(
                              tape, {(unsigned)idx}));

        Type *innerType = ret->getType();
        for (size_t i=0, limit=getSubLimits(BuilderQ.GetInsertBlock()).size(); i<limit; ++i) {
          if (!isa<PointerType>(innerType)) {
            llvm::errs() << "fn: " << *BuilderQ.GetInsertBlock()->getParent()
                         << "\n";
            llvm::errs() << "bq insertblock: " << *BuilderQ.GetInsertBlock()
                         << "\n";
            llvm::errs() << "ret: " << *ret << " type: " << *ret->getType()
                         << "\n";
            llvm::errs() << "innerType: " << *innerType << "\n";
            if (malloc)
              llvm::errs() << " malloc: " << *malloc << "\n";
          }
          assert(isa<PointerType>(innerType));
          innerType = cast<PointerType>(innerType)->getElementType();
        }

        assert(malloc);
        if (efficientBoolCache && malloc->getType()->isIntegerTy() &&
            cast<IntegerType>(malloc->getType())->getBitWidth() == 1 &&
            innerType != ret->getType()) {
          assert(innerType == Type::getInt8Ty(malloc->getContext()));
        } else {
          if (innerType != malloc->getType()) {
            llvm::errs() << *cast<Instruction>(malloc)->getParent()->getParent()
                         << "\n";
            llvm::errs() << "innerType: " << *innerType << "\n";
            llvm::errs() << "malloc->getType(): " << *malloc->getType() << "\n";
            llvm::errs() << "ret: " << *ret << "\n";
            llvm::errs() << "malloc: " << *malloc << "\n";
          }
        }

        AllocaInst *cache =
            createCacheForScope(BuilderQ.GetInsertBlock(), innerType,
                                "mdyncache_fromtape", true, false);
        assert(malloc);
        bool isi1 = malloc->getType()->isIntegerTy() &&
                    cast<IntegerType>(malloc->getType())->getBitWidth() == 1;
        entryBuilder.CreateStore(ret, cache);

        auto v = lookupValueFromCache(BuilderQ, BuilderQ.GetInsertBlock(),
                                      cache, isi1);
        if (malloc) {
          assert(v->getType() == malloc->getType());
        }
        scopeMap[v] = std::make_pair(cache, ctx);
        ret = cast<Instruction>(v);
      }

      if (malloc && !isa<UndefValue>(malloc)) {
        if (malloc->getType() != ret->getType()) {
          llvm::errs() << *oldFunc << "\n";
          llvm::errs() << *newFunc << "\n";
          llvm::errs() << *malloc << "\n";
          llvm::errs() << *ret << "\n";
        }
        assert(malloc->getType() == ret->getType());

        if (auto orig = isOriginal(malloc))
          originalToNewFn[orig] = ret;

        if (scopeMap.find(malloc) != scopeMap.end()) {
          // There already exists an alloaction for this, we should fully remove
          // it
          if (!inLoop) {

            // Remove stores into
            auto stores = scopeStores[scopeMap[malloc].first];
            scopeStores.erase(scopeMap[malloc].first);
            for (int i = stores.size() - 1; i >= 0; i--) {
              if (auto inst = dyn_cast<Instruction>(stores[i])) {
                erase(inst);
              }
            }

            std::vector<User *> users;
            for (auto u : scopeMap[malloc].first->users()) {
              users.push_back(u);
            }
            for (auto u : users) {
              if (auto li = dyn_cast<LoadInst>(u)) {
                IRBuilder<> lb(li);
                ValueToValueMapTy empty;
                li->replaceAllUsesWith(
                    unwrapM(ret, lb, empty, UnwrapMode::LegalFullUnwrap));
                erase(li);
              } else {
                llvm::errs() << "newFunc: " << *newFunc << "\n";
                llvm::errs() << "malloc: " << *malloc << "\n";
                llvm::errs()
                    << "scopeMap[malloc]: " << *scopeMap[malloc].first << "\n";
                llvm::errs() << "u: " << *u << "\n";
                assert(0 && "illegal use for out of loop scopeMap");
              }
            }

            {
              AllocaInst *preerase = scopeMap[malloc].first;
              scopeMap.erase(malloc);
              erase(preerase);
            }
          } else {
            // Remove stores into
            auto stores = scopeStores[scopeMap[malloc].first];
            scopeStores.erase(scopeMap[malloc].first);
            for (int i = stores.size() - 1; i >= 0; i--) {
              if (auto inst = dyn_cast<Instruction>(stores[i])) {
                erase(inst);
              }
            }

            // Remove allocations for scopealloc since it is already allocated
            // by the augmented forward pass
            auto allocs = scopeAllocs[scopeMap[malloc].first];
            scopeAllocs.erase(scopeMap[malloc].first);
            for (auto allocinst : allocs) {
              CastInst *cast = nullptr;
              StoreInst *store = nullptr;
              for (auto use : allocinst->users()) {
                if (auto ci = dyn_cast<CastInst>(use)) {
                  assert(cast == nullptr);
                  cast = ci;
                }
                if (auto si = dyn_cast<StoreInst>(use)) {
                  if (si->getValueOperand() == allocinst) {
                    assert(store == nullptr);
                    store = si;
                  }
                }
              }
              if (cast) {
                assert(store == nullptr);
                for (auto use : cast->users()) {
                  if (auto si = dyn_cast<StoreInst>(use)) {
                    if (si->getValueOperand() == cast) {
                      assert(store == nullptr);
                      store = si;
                    }
                  }
                }
              }
              /*
              if (!store) {
                  allocinst->getParent()->getParent()->dump();
                  allocinst->dump();
              }
              assert(store);
              erase(store);
              */

              Instruction *storedinto =
                  cast ? (Instruction *)cast : (Instruction *)allocinst;
              for (auto use : storedinto->users()) {
                // llvm::errs() << " found use of " << *storedinto << " of " <<
                // use << "\n";
                if (auto si = dyn_cast<StoreInst>(use))
                  erase(si);
              }

              if (cast)
                erase(cast);
              // llvm::errs() << "considering inner loop for malloc: " <<
              // *malloc << " allocinst " << *allocinst << "\n";
              erase(allocinst);
            }

            // Remove frees
            auto tofree = scopeFrees[scopeMap[malloc].first];
            scopeFrees.erase(scopeMap[malloc].first);
            for (auto freeinst : tofree) {
              std::deque<Value *> ops = {freeinst->getArgOperand(0)};
              erase(freeinst);

              while (ops.size()) {
                auto z = dyn_cast<Instruction>(ops[0]);
                ops.pop_front();
                if (z && z->getNumUses() == 0) {
                  for (unsigned i = 0; i < z->getNumOperands(); ++i) {
                    ops.push_back(z->getOperand(i));
                  }
                  erase(z);
                }
              }
            }

            // uses of the alloc
            std::vector<User *> users;
            for (auto u : scopeMap[malloc].first->users()) {
              users.push_back(u);
            }
            for (auto u : users) {
              if (auto li = dyn_cast<LoadInst>(u)) {
                IRBuilder<> lb(li);
                // llvm::errs() << "fixing li: " << *li << "\n";
                auto replacewith =
                    (idx < 0) ? tape
                              : lb.CreateExtractValue(tape, {(unsigned)idx});
                // llvm::errs() << "fixing with rw: " << *replacewith << "\n";
                li->replaceAllUsesWith(replacewith);
                erase(li);
              } else {
                llvm::errs() << "newFunc: " << *newFunc << "\n";
                llvm::errs() << "malloc: " << *malloc << "\n";
                llvm::errs()
                    << "scopeMap[malloc]: " << *scopeMap[malloc].first << "\n";
                llvm::errs() << "u: " << *u << "\n";
                assert(0 && "illegal use for out of loop scopeMap");
              }
            }

            // cast<Instruction>(scopeMap[malloc])->getParent()->getParent()->dump();

            // llvm::errs() << "did erase for malloc: " << *malloc << " " <<
            // *scopeMap[malloc] << "\n";

            AllocaInst *preerase = scopeMap[malloc].first;
            scopeMap.erase(malloc);
            erase(preerase);
          }
        }
        // llvm::errs() << "replacing " << *malloc << " with " << *ret << "\n";
        cast<Instruction>(malloc)->replaceAllUsesWith(ret);
        std::string n = malloc->getName().str();
        erase(cast<Instruction>(malloc));
        ret->setName(n);
      }
      return ret;
    } else {
      assert(malloc);
      // assert(!isa<PHINode>(malloc));

      assert(idx >= 0 && (unsigned)idx == addedTapeVals.size());

      if (isa<UndefValue>(malloc)) {
        addedTapeVals.push_back(malloc);
        return malloc;
      }

      BasicBlock *ctx = BuilderQ.GetInsertBlock();
      if (auto inst = dyn_cast<Instruction>(malloc))
        ctx = inst->getParent();
      auto found = scopeMap.find(malloc);
      if (found != scopeMap.end()) {
        ctx = found->second.second;
      }

      bool inLoop;

      if ((size_t)ctx & 1) {
        inLoop = true;
        ctx = (BasicBlock *)((size_t)ctx ^ 1);
      } else {
        LoopContext lc;
        inLoop = getContext(ctx, lc);
      }

      if (!inLoop) {
        addedTapeVals.push_back(malloc);
        return malloc;
      }

      ensureLookupCached(cast<Instruction>(malloc),
                         /*shouldFree=*/reverseBlocks.size() > 0);
      assert(scopeMap[malloc].first);

      Instruction *toadd = scopeAllocs[scopeMap[malloc].first][0];
      for (auto u : toadd->users()) {
        if (auto ci = dyn_cast<CastInst>(u)) {
          toadd = ci;
        }
      }

      // llvm::errs() << " malloc: " << *malloc << "\n";
      // llvm::errs() << " toadd: " << *toadd << "\n";
      Type *innerType = toadd->getType();
      for (size_t i=0, limit=getSubLimits(BuilderQ.GetInsertBlock()).size(); i<limit; ++i) {
        innerType = cast<PointerType>(innerType)->getElementType();
      }

      if (efficientBoolCache && malloc->getType()->isIntegerTy() &&
          toadd->getType() != innerType &&
          cast<IntegerType>(malloc->getType())->getBitWidth() == 1) {
        assert(innerType == Type::getInt8Ty(toadd->getContext()));
      } else {
        if (innerType != malloc->getType()) {
          llvm::errs() << "oldFunc:" << *oldFunc << "\n";
          llvm::errs() << "newFunc: " << *newFunc << "\n";
          llvm::errs() << " toadd: " << *toadd << "\n";
          llvm::errs() << "innerType: " << *innerType << "\n";
          llvm::errs() << "malloc: " << *malloc << "\n";
        }
        assert(innerType == malloc->getType());
      }
      addedTapeVals.push_back(toadd);
      return malloc;
    }
    llvm::errs() << "Fell through on cacheForReverse. This should never happen.\n";
    assert(false);
  }

  const SmallVectorImpl<Value *> &getTapeValues() const { return addedTapeVals; }

public:
  TargetLibraryInfo &TLI;
  AAResults &AA;
  TypeAnalysis &TA;
  GradientUtils(Function *newFunc_, Function *oldFunc_, TargetLibraryInfo &TLI_,
                TypeAnalysis &TA_, AAResults &AA_,
                ValueToValueMapTy &invertedPointers_,
                const SmallPtrSetImpl<Value *> &constants_,
                const SmallPtrSetImpl<Value *> &nonconstant_,
                const SmallPtrSetImpl<Value *> &constantvalues_,
                const SmallPtrSetImpl<Value *> &returnvals_,
                ValueToValueMapTy &originalToNewFn_, DerivativeMode mode)
      : mode(mode), newFunc(newFunc_), oldFunc(oldFunc_), invertedPointers(),
        DT(*newFunc_), OrigDT(*oldFunc_), OrigPDT(*oldFunc_),
        ATA(AA_, 3),
        OrigLI(OrigDT), LI(DT), AC(*newFunc_), SE(*newFunc_, TLI_, AC, DT, LI),
        inversionAllocs(nullptr), TLI(TLI_), AA(AA_), TA(TA_) {

        ATA.constants.insert(constants_.begin(), constants_.end());
        ATA.nonconstant.insert(nonconstant_.begin(), nonconstant_.end());
        ATA.constantvals.insert(constantvalues_.begin(), constantvalues_.end());
        //nonconstant vals
        ATA.retvals.insert(returnvals_.begin(), returnvals_.end());

    invertedPointers.insert(invertedPointers_.begin(), invertedPointers_.end());
    originalToNewFn.insert(originalToNewFn_.begin(), originalToNewFn_.end());
    for (BasicBlock &BB : *newFunc) {
      originalBlocks.emplace_back(&BB);
    }
    tape = nullptr;
    tapeidx = 0;
    assert(originalBlocks.size() > 0);
    inversionAllocs = BasicBlock::Create(newFunc_->getContext(),
                                         "allocsForInversion", newFunc);
  }

public:
  static GradientUtils *
  CreateFromClone(Function *todiff, TargetLibraryInfo &TLI, TypeAnalysis &TA,
                  AAResults &AA, DIFFE_TYPE retType,
                  const std::vector<DIFFE_TYPE> &constant_args, bool returnUsed,
                  std::map<AugmentedStruct, int> &returnMapping);

  StoreInst *setPtrDiffe(Value *ptr, Value *newval, IRBuilder<> &BuilderM) {
    if (auto inst = dyn_cast<Instruction>(ptr)) {
      assert(inst->getParent()->getParent() == oldFunc);
    }
    if (auto arg = dyn_cast<Argument>(ptr)) {
      assert(arg->getParent() == oldFunc);
    }
    ptr = invertPointerM(ptr, BuilderM);
    return BuilderM.CreateStore(newval, ptr);
  }

  void prepareForReverse() {
    assert(reverseBlocks.size() == 0);
    for (BasicBlock *BB : originalBlocks) {
      reverseBlocks[BB] = BasicBlock::Create(BB->getContext(),
                                             "invert" + BB->getName(), newFunc);
    }
    assert(reverseBlocks.size() != 0);
  }

  BasicBlock *originalForReverseBlock(BasicBlock &BB2) const {
    assert(reverseBlocks.size() != 0);
    for (auto BB : originalBlocks) {
      auto it = reverseBlocks.find(BB);
      assert(it != reverseBlocks.end());
      if (it->second == &BB2) {
        return BB;
      }
    }
    llvm::errs() << *newFunc << "\n";
    llvm::errs() << BB2 << "\n";
    assert(0 && "could not find original block for given reverse block");
    report_fatal_error("could not find original block for given reverse block");
  }

  //! This cache stores blocks we may insert as part of getReverseOrLatchMerge
  //! to handle inverse iv iteration
  //  As we don't want to create redundant blocks, we use this convenient cache
  std::map<std::tuple<BasicBlock *, BasicBlock *>, BasicBlock *>
      newBlocksForLoop_cache;
  BasicBlock *getReverseOrLatchMerge(BasicBlock *BB,
                                     BasicBlock *branchingBlock);

  void forceContexts();

  bool getContext(BasicBlock *BB, LoopContext &loopContext);

  bool isOriginalBlock(const BasicBlock &BB) const {
    for (auto A : originalBlocks) {
      if (A == &BB)
        return true;
    }
    return false;
  }

  bool isConstantValueInternal(Value *val, AAResults &AA, TypeResults &TR) {
    cast<Value>(val);
    if (auto inst = dyn_cast<Instruction>(val))
      assert(inst->getParent()->getParent() == oldFunc);
    return ATA.isconstantValueM(TR, val);
  };

  bool isConstantInstructionInternal(Instruction *val, AAResults &AA,
                                     TypeResults &TR) {
    cast<Instruction>(val);
    assert(val->getParent()->getParent() == oldFunc);
    return ATA.isconstantM(TR, val);
  }

  void eraseFictiousPHIs() {
    for (auto pp : fictiousPHIs) {
      if (pp->getNumUses() != 0) {
        llvm::errs() << "oldFunc:" << *oldFunc << "\n";
        llvm::errs() << "newFunc:" << *newFunc << "\n";
        llvm::errs() << " pp: " << *pp << "\n";
      }
      assert(pp->getNumUses() == 0);
      pp->replaceAllUsesWith(UndefValue::get(pp->getType()));
      erase(pp);
    }
    fictiousPHIs.clear();
  }

  std::map<llvm::Value *, bool> internal_isConstantValue;
  std::map<const llvm::Instruction *, bool> internal_isConstantInstruction;

  void forceActiveDetection(AAResults &AA, TypeResults &TR) {
    for (auto a = oldFunc->arg_begin(); a != oldFunc->arg_end(); ++a) {
      if (ATA.constants.find(a) == ATA.constants.end() &&
          ATA.nonconstant.find(a) == ATA.nonconstant.end())
        continue;

      bool const_value = isConstantValueInternal(a, AA, TR);

      internal_isConstantValue[a] = const_value;

      // a->addAttr(llvm::Attribute::get(a->getContext(),
      // "enzyme_activity_value", const_value ? "const" : "active"));
      // cast<Argument>(getNewFromOriginal(a))->addAttr(llvm::Attribute::get(a->getContext(),
      // "enzyme_activity_value", const_value ? "const" : "active"));
    }

    for (BasicBlock &BB : *oldFunc) {
      for (Instruction &I : BB) {
        bool const_inst = isConstantInstructionInternal(&I, AA, TR);

        // I.setMetadata("enzyme_activity_inst", MDNode::get(I.getContext(),
        // MDString::get(I.getContext(), const_inst ? "const" : "active")));
        // I.setMetadata(const_inst ? "enzyme_constinst" : "enzyme_activeinst",
        // MDNode::get(I.getContext(), {}));

        // I.addAttr(llvm::Attribute::get(I.getContext(),
        // "enzyme_activity_inst", const_inst ? "const" : "active"));
        bool const_value = isConstantValueInternal(&I, AA, TR);
        // I.setMetadata(const_value ? "enzyme_constvalue" :
        // "enzyme_activevalue", MDNode::get(I.getContext(), {}));
        // I.setMetadata("enzyme_activity_value", MDNode::get(I.getContext(),
        // MDString::get(I.getContext(), const_value ? "const" : "active")));
        // I.addAttr(llvm::Attribute::get(I.getContext(),
        // "enzyme_activity_value", const_value ? "const" : "active"));

        internal_isConstantValue[&I] = const_value;
        internal_isConstantInstruction[&I] = const_inst;
      }
    }
  }

  llvm::StringRef getAttribute(Argument *arg, std::string attr) const {
    return arg->getParent()
        ->getAttributes()
        .getParamAttr(arg->getArgNo(), attr)
        .getValueAsString();
  }

  bool isConstantValue(Value *val) const {
    if (auto inst = dyn_cast<Instruction>(val)) {
      assert(inst->getParent()->getParent() == oldFunc);
      assert(internal_isConstantValue.find(inst) !=
             internal_isConstantValue.end());
      return internal_isConstantValue.find(inst)->second;
    }

    if (auto arg = dyn_cast<Argument>(val)) {
      assert(arg->getParent() == oldFunc);
      assert(internal_isConstantValue.find(arg) !=
             internal_isConstantValue.end());
      return internal_isConstantValue.find(arg)->second;
    }

    //! Functions must be false so we can replace function with augmentation,
    //! fallback to analysis
    if (isa<Function>(val) || isa<InlineAsm>(val) || isa<Constant>(val) ||
        isa<UndefValue>(val) || isa<MetadataAsValue>(val)) {
      // Note that not actually passing in type results here as (hopefully) it
      // shouldn't be needed
      TypeResults TR(TA, FnTypeInfo(oldFunc));
      return const_cast<GradientUtils *>(this)->isConstantValueInternal(val, AA,
                                                                        TR);
    }

    if (auto gv = dyn_cast<GlobalVariable>(val)) {
      if (hasMetadata(gv, "enzyme_shadow"))
        return false;
      if (auto md = gv->getMetadata("enzyme_activity_value")) {
        auto res = cast<MDString>(md->getOperand(0))->getString();
        if (res == "const")
          return true;
        if (res == "active")
          return false;
      }
      if (nonmarkedglobals_inactive)
        return true;
      goto err;
    }
    if (isa<GlobalValue>(val)) {
      if (nonmarkedglobals_inactive)
        return true;
      goto err;
    }

  err:;
    llvm::errs() << *oldFunc << "\n";
    llvm::errs() << *newFunc << "\n";
    llvm::errs() << *val << "\n";
    llvm::errs() << "  unknown did status attribute\n";
    assert(0 && "bad");
    exit(1);
  }

  bool isConstantInstruction(const Instruction *inst) const {
    assert(inst->getParent()->getParent() == oldFunc);
    if (internal_isConstantInstruction.find(inst) ==
        internal_isConstantInstruction.end()) {
      llvm::errs() << *oldFunc << "\n";
      for (auto &pair : internal_isConstantInstruction) {
        llvm::errs() << " constantinst[" << *pair.first << "] = " << pair.second
                     << "\n";
      }
      llvm::errs() << "inst: " << *inst << "\n";
    }
    assert(internal_isConstantInstruction.find(inst) !=
           internal_isConstantInstruction.end());
    return internal_isConstantInstruction.find(inst)->second;
  }

  void forceAugmentedReturns(
      TypeResults &TR,
      const SmallPtrSetImpl<BasicBlock *> &guaranteedUnreachable) {
    assert(TR.info.Function == oldFunc);

    for (BasicBlock &oBB : *oldFunc) {
      // Don't create derivatives for code that results in termination
      if (guaranteedUnreachable.find(&oBB) != guaranteedUnreachable.end())
        continue;

      LoopContext loopContext;
      getContext(cast<BasicBlock>(getNewFromOriginal(&oBB)), loopContext);

      for (Instruction &I : oBB) {
        Instruction *inst = &I;

        if (inst->getType()->isEmptyTy())
          continue;

        if (inst->getType()->isFPOrFPVectorTy())
          continue; //! op->getType()->isPointerTy() &&
                    //! !op->getType()->isIntegerTy()) {

        if (!TR.query(inst).Data0()[{}].isPossiblePointer())
          continue;

        Instruction *newi = getNewFromOriginal(inst);

        if (isa<LoadInst>(inst)) {
          IRBuilder<> BuilderZ(getNextNonDebugInstruction(newi));
          BuilderZ.setFastMathFlags(getFast());
          PHINode *anti = BuilderZ.CreatePHI(inst->getType(), 1,
                                             inst->getName() + "'il_phi");
          invertedPointers[inst] = anti;
          continue;
        }

        if (!isa<CallInst>(inst)) {
          continue;
        }

        if (isa<IntrinsicInst>(inst)) {
          continue;
        }

        if (isConstantValue(inst)) {
          continue;
        }

        CallInst *op = cast<CallInst>(inst);
        Function *called = op->getCalledFunction();

        if (called && isCertainPrintOrFree(called)) {
          continue;
        }

        IRBuilder<> BuilderZ(getNextNonDebugInstruction(newi));
        BuilderZ.setFastMathFlags(getFast());
        PHINode *anti =
            BuilderZ.CreatePHI(op->getType(), 1, op->getName() + "'ip_phi");
        invertedPointers[inst] = anti;

        if (called &&
            (called->getName() == "malloc" || called->getName() == "_Znwm")) {
          invertedPointers[inst]->setName(op->getName() + "'mi");
        }
      }
    }
  }

  //! if full unwrap, don't just unwrap this instruction, but also its operands,
  //! etc

  Value *
  unwrapM(Value *const val, IRBuilder<> &BuilderM,
          const ValueToValueMapTy &available,
          UnwrapMode mode) { // bool lookupIfAble, bool fullUnwrap=true) {
    assert(val);
    assert(val->getName() != "<badref>");
    assert(val->getType());

    // llvm::errs() << " attempting unwrap of: " << *val << "\n";

    for (auto pair : available) {
      assert(pair.first);
      assert(pair.second);
      assert(pair.first->getType());
      assert(pair.second->getType());
      assert(pair.first->getType() == pair.second->getType());
    }

    if (isa<LoadInst>(val) &&
        cast<LoadInst>(val)->getMetadata("enzyme_mustcache")) {
      return val;
    }

    // assert(!val->getName().startswith("$tapeload"));

    auto cidx = std::make_pair(val, BuilderM.GetInsertBlock());
    if (unwrap_cache.find(cidx) != unwrap_cache.end()) {
      if (unwrap_cache[cidx]->getType() != val->getType()) {
        llvm::errs() << "val: " << *val << "\n";
        llvm::errs() << "unwrap_cache[cidx]: " << *unwrap_cache[cidx] << "\n";
      }
      assert(unwrap_cache[cidx]->getType() == val->getType());
      return unwrap_cache[cidx];
    }

    if (available.count(val)) {
      auto avail = available.lookup(val);
      assert(avail->getType());
      if (avail->getType() != val->getType()) {
        llvm::errs() << "val: " << *val << "\n";
        llvm::errs() << "available[val]: " << *available.lookup(val) << "\n";
      }
      assert(available.lookup(val)->getType() == val->getType());
      return available.lookup(val);
    }

    if (auto inst = dyn_cast<Instruction>(val)) {
      // if (inst->getParent() == &newFunc->getEntryBlock()) {
      //  return inst;
      //}
      if (isOriginalBlock(*BuilderM.GetInsertBlock())) {
        if (BuilderM.GetInsertBlock()->size() &&
            BuilderM.GetInsertPoint() != BuilderM.GetInsertBlock()->end()) {
          if (DT.dominates(inst, &*BuilderM.GetInsertPoint())) {
            // llvm::errs() << "allowed " << *inst << "from domination\n";
            assert(inst->getType() == val->getType());
            return inst;
          }
        } else {
          if (DT.dominates(inst, BuilderM.GetInsertBlock())) {
            // llvm::errs() << "allowed " << *inst << "from block domination\n";
            assert(inst->getType() == val->getType());
            return inst;
          }
        }
      }
    }

#define SAFE(a, b)                                                             \
  ({                                                                           \
    Value *res = a->b;                                                         \
    res;                                                                       \
  })

    // llvm::errs() << "uwval: " << *val << "\n";
    auto getOp = [&](Value *v) -> Value * {
      if (mode == UnwrapMode::LegalFullUnwrap ||
          mode == UnwrapMode::AttemptFullUnwrap ||
          mode == UnwrapMode::AttemptFullUnwrapWithLookup) {
        return unwrapM(v, BuilderM, available, mode);
      } else {
        assert(mode == UnwrapMode::AttemptSingleUnwrap);
        return lookupM(v, BuilderM, available);
      }
    };

    if (isa<Argument>(val) || isa<Constant>(val)) {
      unwrap_cache[std::make_pair(val, BuilderM.GetInsertBlock())] = val;
      return val;
    } else if (isa<AllocaInst>(val)) {
      unwrap_cache[std::make_pair(val, BuilderM.GetInsertBlock())] = val;
      return val;
    } else if (auto op = dyn_cast<CastInst>(val)) {
      auto op0 = getOp(SAFE(op, getOperand(0)));
      if (op0 == nullptr)
        goto endCheck;
      auto toreturn = BuilderM.CreateCast(op->getOpcode(), op0, op->getDestTy(),
                                          op->getName() + "_unwrap");
      if (auto newi = dyn_cast<Instruction>(toreturn))
        newi->copyIRFlags(op);
      unwrap_cache[cidx] = toreturn;
      assert(val->getType() == toreturn->getType());
      return toreturn;
    } else if (auto op = dyn_cast<ExtractValueInst>(val)) {
      auto op0 = getOp(SAFE(op, getAggregateOperand()));
      if (op0 == nullptr)
        goto endCheck;
      auto toreturn = BuilderM.CreateExtractValue(op0, op->getIndices(),
                                                  op->getName() + "_unwrap");
      unwrap_cache[cidx] = toreturn;
      if (auto newi = dyn_cast<Instruction>(toreturn))
        newi->copyIRFlags(op);
      assert(val->getType() == toreturn->getType());
      return toreturn;
    } else if (auto op = dyn_cast<BinaryOperator>(val)) {
      auto op0 = getOp(SAFE(op, getOperand(0)));
      if (op0 == nullptr)
        goto endCheck;
      auto op1 = getOp(SAFE(op, getOperand(1)));
      if (op1 == nullptr)
        goto endCheck;
      auto toreturn = BuilderM.CreateBinOp(op->getOpcode(), op0, op1,
                                           op->getName() + "_unwrap");
      if (auto newi = dyn_cast<Instruction>(toreturn))
        newi->copyIRFlags(op);
      unwrap_cache[cidx] = toreturn;
      assert(val->getType() == toreturn->getType());
      return toreturn;
    } else if (auto op = dyn_cast<ICmpInst>(val)) {
      auto op0 = getOp(SAFE(op, getOperand(0)));
      if (op0 == nullptr)
        goto endCheck;
      auto op1 = getOp(SAFE(op, getOperand(1)));
      if (op1 == nullptr)
        goto endCheck;
      auto toreturn = BuilderM.CreateICmp(op->getPredicate(), op0, op1,
                                          op->getName() + "_unwrap");
      if (auto newi = dyn_cast<Instruction>(toreturn))
        newi->copyIRFlags(op);
      unwrap_cache[cidx] = toreturn;
      assert(val->getType() == toreturn->getType());
      return toreturn;
    } else if (auto op = dyn_cast<FCmpInst>(val)) {
      auto op0 = getOp(SAFE(op, getOperand(0)));
      if (op0 == nullptr)
        goto endCheck;
      auto op1 = getOp(SAFE(op, getOperand(1)));
      if (op1 == nullptr)
        goto endCheck;
      auto toreturn = BuilderM.CreateFCmp(op->getPredicate(), op0, op1,
                                          op->getName() + "_unwrap");
      if (auto newi = dyn_cast<Instruction>(toreturn))
        newi->copyIRFlags(op);
      unwrap_cache[cidx] = toreturn;
      assert(val->getType() == toreturn->getType());
      return toreturn;
    } else if (auto op = dyn_cast<SelectInst>(val)) {
      auto op0 = getOp(SAFE(op, getOperand(0)));
      if (op0 == nullptr)
        goto endCheck;
      auto op1 = getOp(SAFE(op, getOperand(1)));
      if (op1 == nullptr)
        goto endCheck;
      auto op2 = getOp(SAFE(op, getOperand(2)));
      if (op2 == nullptr)
        goto endCheck;
      auto toreturn =
          BuilderM.CreateSelect(op0, op1, op2, op->getName() + "_unwrap");
      if (auto newi = dyn_cast<Instruction>(toreturn))
        newi->copyIRFlags(op);
      unwrap_cache[cidx] = toreturn;
      assert(val->getType() == toreturn->getType());
      return toreturn;
    } else if (auto inst = dyn_cast<GetElementPtrInst>(val)) {
      auto ptr = getOp(SAFE(inst, getPointerOperand()));
      if (ptr == nullptr)
        goto endCheck;
      SmallVector<Value *, 4> ind;
      // llvm::errs() << "inst: " << *inst << "\n";
      for (unsigned i = 0; i < inst->getNumIndices(); ++i) {
        Value *a = SAFE(inst, getOperand(1 + i));
        assert(a->getName() != "<badref>");
        auto op = getOp(a);
        if (op == nullptr)
          goto endCheck;
        ind.push_back(op);
      }
      auto toreturn = BuilderM.CreateGEP(ptr, ind, inst->getName() + "_unwrap");
      if (isa<GetElementPtrInst>(toreturn))
        cast<GetElementPtrInst>(toreturn)->setIsInBounds(inst->isInBounds());
      else {
        // llvm::errs() << "gep tr: " << *toreturn << " inst: " << *inst << "
        // ptr: " << *ptr << "\n"; llvm::errs() << "safe: " << *SAFE(inst,
        // getPointerOperand()) << "\n"; assert(0 && "illegal");
      }
      if (auto newi = dyn_cast<Instruction>(toreturn))
        newi->copyIRFlags(inst);
      unwrap_cache[cidx] = toreturn;
      assert(val->getType() == toreturn->getType());
      return toreturn;
    } else if (auto load = dyn_cast<LoadInst>(val)) {
      if (load->getMetadata("enzyme_noneedunwrap"))
        return load;

      bool legalMove = mode == UnwrapMode::LegalFullUnwrap;
      if (mode != UnwrapMode::LegalFullUnwrap) {
        // TODO actually consider whether this is legal to move to the new
        // location, rather than recomputable anywhere
        legalMove = legalRecompute(load, available);
      }
      if (!legalMove)
        return nullptr;

      Value *idx = getOp(SAFE(load, getOperand(0)));
      if (idx == nullptr)
        goto endCheck;

      if (idx->getType() != load->getOperand(0)->getType()) {
        llvm::errs() << "load: " << *load << "\n";
        llvm::errs() << "load->getOperand(0): " << *load->getOperand(0) << "\n";
        llvm::errs() << "idx: " << *idx << "\n";
      }
      assert(idx->getType() == load->getOperand(0)->getType());
      auto toreturn = BuilderM.CreateLoad(idx, load->getName() + "_unwrap");
      if (auto newi = dyn_cast<Instruction>(toreturn))
        newi->copyIRFlags(load);
#if LLVM_VERSION_MAJOR >= 10
      toreturn->setAlignment(load->getAlign());
#else
      toreturn->setAlignment(load->getAlignment());
#endif
      toreturn->setVolatile(load->isVolatile());
      toreturn->setOrdering(load->getOrdering());
      toreturn->setSyncScopeID(load->getSyncScopeID());
      toreturn->setMetadata(LLVMContext::MD_tbaa,
                            load->getMetadata(LLVMContext::MD_tbaa));
      toreturn->setMetadata("enzyme_unwrapped",
                            MDNode::get(toreturn->getContext(), {}));
      // toreturn->setMetadata(LLVMContext::MD_invariant,
      // load->getMetadata(LLVMContext::MD_invariant));
      toreturn->setMetadata(LLVMContext::MD_invariant_group,
                            load->getMetadata(LLVMContext::MD_invariant_group));
      // TODO adding to cache only legal if no alias of any future writes
      unwrap_cache[cidx] = toreturn;
      assert(val->getType() == toreturn->getType());
      return toreturn;
    } else if (auto op = dyn_cast<CallInst>(val)) {

      bool legalMove = mode == UnwrapMode::LegalFullUnwrap;
      if (mode != UnwrapMode::LegalFullUnwrap) {
        // TODO actually consider whether this is legal to move to the new
        // location, rather than recomputable anywhere
        legalMove = legalRecompute(op, available);
      }
      if (!legalMove)
        return nullptr;

      std::vector<Value *> args;
      for (unsigned i = 0; i < op->getNumArgOperands(); ++i) {
        args.emplace_back(getOp(SAFE(op, getArgOperand(i))));
        if (args[i] == nullptr)
          return nullptr;
      }

      #if LLVM_VERSION_MAJOR >= 11
      Value *fn = getOp(SAFE(op, getCalledOperand()));
      #else
      Value *fn = getOp(SAFE(op, getCalledValue()));
      #endif
      if (fn == nullptr)
        return nullptr;

      auto toreturn = cast<CallInst>(BuilderM.CreateCall(op->getFunctionType(), fn, args));
      toreturn->copyIRFlags(op);
      toreturn->setAttributes(op->getAttributes());
      toreturn->setCallingConv(op->getCallingConv());
      toreturn->setTailCallKind(op->getTailCallKind());
      toreturn->setDebugLoc(op->getDebugLoc());
      return toreturn;
    } else if (auto phi = dyn_cast<PHINode>(val)) {
      if (phi->getNumIncomingValues() == 1) {
        assert(SAFE(phi, getIncomingValue(0)) != phi);
        auto toreturn = getOp(SAFE(phi, getIncomingValue(0)));
        if (toreturn == nullptr)
          goto endCheck;
        assert(val->getType() == toreturn->getType());
        if (auto newi = dyn_cast<Instruction>(toreturn))
          newi->copyIRFlags(op);
        return toreturn;
      }
    }

  endCheck:
    assert(val);
    if (mode == UnwrapMode::LegalFullUnwrap ||
        mode == UnwrapMode::AttemptFullUnwrapWithLookup) {
      assert(val->getName() != "<badref>");
      auto toreturn = lookupM(val, BuilderM);
      assert(val->getType() == toreturn->getType());
      return toreturn;
    }

    // llvm::errs() << "cannot unwrap following " << *val << "\n";

    if (auto inst = dyn_cast<Instruction>(val)) {
      // LoopContext lc;
      // if (BuilderM.GetInsertBlock() != inversionAllocs && !(
      // (reverseBlocks.find(BuilderM.GetInsertBlock()) != reverseBlocks.end())
      // && /*inLoop*/getContext(inst->getParent(), lc)) ) {
      if (isOriginalBlock(*BuilderM.GetInsertBlock())) {
        if (BuilderM.GetInsertBlock()->size() &&
            BuilderM.GetInsertPoint() != BuilderM.GetInsertBlock()->end()) {
          if (DT.dominates(inst, &*BuilderM.GetInsertPoint())) {
            // llvm::errs() << "allowed " << *inst << "from domination\n";
            assert(inst->getType() == val->getType());
            return inst;
          }
        } else {
          if (DT.dominates(inst, BuilderM.GetInsertBlock())) {
            // llvm::errs() << "allowed " << *inst << "from block domination\n";
            assert(inst->getType() == val->getType());
            return inst;
          }
        }
      }
    }
    return nullptr;
  }

  BasicBlock *const fakeContext = (BasicBlock *)(0xDEADBEEF);
  //! returns true indices
  std::vector<std::pair</*sublimit*/ Value *, /*loop limits*/ std::vector<
                            std::pair<LoopContext, Value *>>>>
  getSubLimits(BasicBlock *ctx) {
    {
      LoopContext idx;
      if ((size_t)ctx & 1) {
        auto subctx = (BasicBlock *)((size_t)ctx ^ 1);
        auto zero =
            ConstantInt::get(Type::getInt64Ty(oldFunc->getContext()), 0);
        auto one = ConstantInt::get(Type::getInt64Ty(oldFunc->getContext()), 1);
        idx.var = nullptr;
        idx.incvar = nullptr;
        idx.antivaralloc = nullptr;
        idx.limit = zero;
        idx.latchMerge = nullptr;
        idx.header = subctx;
        idx.preheader = subctx;
        idx.dynamic = false;
        idx.parent = nullptr;
        idx.exitBlocks = {};
        std::vector<
            std::pair<Value *, std::vector<std::pair<LoopContext, Value *>>>>
            sublimits;
        sublimits.push_back({one, {{idx, one}}});
        return sublimits;
      }
    }

    std::vector<LoopContext> contexts;
    for (BasicBlock *blk = ctx; blk != nullptr;) {
      LoopContext idx;
      if (!getContext(blk, idx)) {
        break;
      }
      contexts.emplace_back(idx);
      blk = idx.preheader;
    }

    std::vector<BasicBlock *> allocationPreheaders(contexts.size(), nullptr);
    std::vector<Value *> limits(contexts.size(), nullptr);
    for (int i = contexts.size() - 1; i >= 0; i--) {
      if ((unsigned)i == contexts.size() - 1) {
        allocationPreheaders[i] = contexts[i].preheader;
      } else if (contexts[i].dynamic) {
        allocationPreheaders[i] = contexts[i].preheader;
      } else {
        allocationPreheaders[i] = allocationPreheaders[i + 1];
      }

      if (contexts[i].dynamic) {
        limits[i] = ConstantInt::get(Type::getInt64Ty(ctx->getContext()), 1);
      } else {
        ValueToValueMapTy prevMap;

        for (int j = contexts.size() - 1;; j--) {
          if (allocationPreheaders[i] == contexts[j].preheader)
            break;
          prevMap[contexts[j].var] = contexts[j].var;
        }

        IRBuilder<> allocationBuilder(&allocationPreheaders[i]->back());
        Value *limitMinus1 = nullptr;

        // llvm::errs() << " considering limit: " << *contexts[i].limit << "\n";

        // for(auto pm : prevMap) {
        //  llvm::errs() << "    + " << pm.first << "\n";
        //}

        // TODO ensure unwrapM considers the legality of illegal caching / etc
        //   legalRecompute does not fulfill this need as its whether its legal
        //   at a certain location, where as legalRecompute specifies it being
        //   recomputable anywhere
        // if (legalRecompute(contexts[i].limit, prevMap)) {
        limitMinus1 = unwrapM(contexts[i].limit, allocationBuilder, prevMap,
                              UnwrapMode::AttemptFullUnwrap);
        //}

        // if (limitMinus1)
        //  llvm::errs() << " + considering limit: " << *contexts[i].limit << "
        //  - " << *limitMinus1 << "\n";
        // else
        //  llvm::errs() << " + considering limit: " << *contexts[i].limit << "
        //  - " << limitMinus1 << "\n";

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

    std::vector<
        std::pair<Value *, std::vector<std::pair<LoopContext, Value *>>>>
        sublimits;

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

  //! Caching mechanism: creates a cache of type T in a scope given by ctx
  //! (where if ctx is in a loop there will be a corresponding number of slots)
  AllocaInst *createCacheForScope(BasicBlock *ctx, Type *T, StringRef name,
                                  bool shouldFree, bool allocateInternal = true,
                                  Value *extraSize = nullptr) {
    assert(ctx);
    assert(T);

    auto sublimits = getSubLimits(ctx);

    /* goes from inner loop to outer loop*/
    std::vector<Type *> types = {T};
    bool isi1 = T->isIntegerTy() && cast<IntegerType>(T)->getBitWidth() == 1;
    if (efficientBoolCache && isi1 && sublimits.size() != 0)
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

    // llvm::errs() << "considering alloca builder: " << name << "\n";

    for (int i = sublimits.size() - 1; i >= 0; i--) {
      const auto &containedloops = sublimits[i].second;

      // llvm::errs() << " + size: " << *size << " ph: " <<
      // containedloops.back().first.preheader->getName() << " header: " <<
      // containedloops.back().first.header->getName() << "\n";
      Type *myType = types[i];

      ConstantInt *byteSizeOfType = ConstantInt::get(
          Type::getInt64Ty(T->getContext()),
          newFunc->getParent()->getDataLayout().getTypeAllocSizeInBits(myType) /
              8);

      if (allocateInternal) {

        IRBuilder<> allocationBuilder(
            &containedloops.back().first.preheader->back());

        Value *size = sublimits[i].first;
        if (efficientBoolCache && isi1 && i == 0) {
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
          // storealloc->setMetadata("enzyme_cache_static_store",
          // MDNode::get(storealloc->getContext(), {}));

          scopeAllocs[alloc].push_back(malloccall);

          // allocationBuilder.GetInsertBlock()->getInstList().push_back(cast<Instruction>(allocation));
          // cast<Instruction>(firstallocation)->moveBefore(allocationBuilder.GetInsertBlock()->getTerminator());
          // mallocs.push_back(firstallocation);
        } else {
          auto zerostore = allocationBuilder.CreateStore(
              ConstantPointerNull::get(PointerType::getUnqual(myType)),
              storeInto);
          scopeStores[alloc].push_back(zerostore);

          // auto mdpair = MDNode::getDistinct(zerostore->getContext(), {});
          // zerostore->setMetadata("enzyme_cache_dynamiczero_store", mdpair);

          /*
          if (containedloops.back().first.incvar !=
          containedloops.back().first.header->getFirstNonPHI()) { llvm::errs()
          << "blk:" << *containedloops.back().first.header << "\n"; llvm::errs()
          << "nonphi:" << *containedloops.back().first.header->getFirstNonPHI()
          << "\n"; llvm::errs() << "incvar:" <<
          *containedloops.back().first.incvar << "\n";
          }
          assert(containedloops.back().first.incvar ==
          containedloops.back().first.header->getFirstNonPHI());
          */
          IRBuilder<> build(containedloops.back().first.incvar->getNextNode());
          Value *allocation = build.CreateLoad(storeInto);
          // Value* foo = build.CreateNUWAdd(containedloops.back().first.var,
          // ConstantInt::get(Type::getInt64Ty(T->getContext()), 1));
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
          // storealloc->setMetadata("enzyme_cache_dynamic_store", mdpair);
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
        assert(reverseBlocks.size());

        IRBuilder<> tbuild(
            reverseBlocks[containedloops.back().first.preheader]);
        tbuild.setFastMathFlags(getFast());

        // ensure we are before the terminator if it exists
        if (tbuild.GetInsertBlock()->size()) {
          tbuild.SetInsertPoint(tbuild.GetInsertBlock()->getTerminator());
        }

        ValueToValueMapTy antimap;
        for (int j = sublimits.size() - 1; j >= i; j--) {
          auto &innercontainedloops = sublimits[j].second;
          for (auto riter = innercontainedloops.rbegin(),
                    rend = innercontainedloops.rend();
               riter != rend; ++riter) {
            const auto &idx = riter->first;
            if (idx.var)
              antimap[idx.var] = tbuild.CreateLoad(idx.antivaralloc);
          }
        }

        auto forfree = cast<LoadInst>(tbuild.CreateLoad(
            unwrapM(storeInto, tbuild, antimap, UnwrapMode::LegalFullUnwrap)));
        forfree->setMetadata(
            LLVMContext::MD_invariant_group,
            invariantGroups[std::make_pair((Value *)alloc, i)]);
        forfree->setMetadata(
            LLVMContext::MD_dereferenceable,
            MDNode::get(forfree->getContext(),
                        {ConstantAsMetadata::get(byteSizeOfType)}));
        unsigned bsize = (unsigned)byteSizeOfType->getZExtValue();
        if ((bsize & (bsize - 1)) == 0) {
#if LLVM_VERSION_MAJOR >= 10
          forfree->setAlignment(Align(bsize));
#else
          forfree->setAlignment(bsize);
#endif
        }
        // forfree->setMetadata(LLVMContext::MD_invariant_load,
        // MDNode::get(forfree->getContext(), {}));
        auto ci = cast<CallInst>(CallInst::CreateFree(
            tbuild.CreatePointerCast(forfree,
                                     Type::getInt8PtrTy(oldFunc->getContext())),
            tbuild.GetInsertBlock()));
        ci->addAttribute(AttributeList::FirstArgIndex, Attribute::NonNull);
        if (ci->getParent() == nullptr) {
          tbuild.Insert(ci);
        }
        scopeFrees[alloc].insert(ci);
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
        // sublimits[i].second.back().first.var

        storeInto = v.CreateGEP(v.CreateLoad(storeInto), idx);
        cast<GetElementPtrInst>(storeInto)->setIsInBounds(true);
      }
    }
    return alloc;
  }

  Value *getCachePointer(IRBuilder<> &BuilderM, BasicBlock *ctx, Value *cache,
                         bool isi1, bool storeInStoresMap = false,
                         Value *extraSize = nullptr) {
    assert(ctx);
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
        llvm::errs() << *oldFunc << "\n";
        llvm::errs() << *newFunc << "\n";
        llvm::errs() << "cache: " << *cache << "\n";
        llvm::errs() << "next: " << *next << "\n";
      }
      assert(next->getType()->isPointerTy());
      // cast<LoadInst>(next)->setMetadata(LLVMContext::MD_invariant_load,
      // MDNode::get(next->getContext(), {}));
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
          oldFunc->getParent()->getDataLayout().getTypeAllocSizeInBits(
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
        if (!isOriginalBlock(*BuilderM.GetInsertBlock())) {
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
        if (efficientBoolCache && isi1 && i == 0)
          idx = BuilderM.CreateLShr(
              idx,
              ConstantInt::get(Type::getInt64Ty(oldFunc->getContext()), 3));
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

  Value *lookupValueFromCache(IRBuilder<> &BuilderM, BasicBlock *ctx,
                              Value *cache, bool isi1,
                              Value *extraSize = nullptr,
                              Value *extraOffset = nullptr) {
    auto cptr = getCachePointer(BuilderM, ctx, cache, isi1,
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
        oldFunc->getParent()->getDataLayout().getTypeAllocSizeInBits(
            result->getType()) /
            8);
    // result->setMetadata(LLVMContext::MD_dereferenceable,
    // MDNode::get(cache->getContext(),
    // {ConstantAsMetadata::get(byteSizeOfType)}));
    unsigned bsize = (unsigned)byteSizeOfType->getZExtValue();
    if ((bsize & (bsize - 1)) == 0) {
#if LLVM_VERSION_MAJOR >= 10
      result->setAlignment(Align(bsize));
#else
      result->setAlignment(bsize);
#endif
    }
    if (efficientBoolCache && isi1) {
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

  void storeInstructionInCache(BasicBlock *ctx, IRBuilder<> &BuilderM,
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
        getCachePointer(v, ctx, cache, isi1, /*storeinstorecache*/ true);
    // if (!isi1) assert(cast<PointerType>(loc->getType())->getElementType() ==
    // val->getType()); else
    // assert(cast<PointerType>(loc->getType())->getElementType() ==
    // Type::getInt8Ty(val->getContext()));

    Value *tostore = val;
    if (efficientBoolCache && isi1) {
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
        ctx->getParent()->getParent()->getDataLayout().getTypeAllocSizeInBits(
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

  void storeInstructionInCache(BasicBlock *ctx, Instruction *inst,
                               AllocaInst *cache) {
    assert(ctx);
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

  void ensureLookupCached(Instruction *inst, bool shouldFree = true) {
    assert(inst);
    if (scopeMap.find(inst) != scopeMap.end())
      return;
    AllocaInst *cache = createCacheForScope(inst->getParent(), inst->getType(),
                                            inst->getName(), shouldFree);
    assert(cache);
    scopeMap[inst] = std::make_pair(cache, inst->getParent());
    storeInstructionInCache(inst->getParent(), inst, cache);
  }

  std::map<Instruction *, std::map<BasicBlock *, Instruction *>> lcssaFixes;
  Instruction *fixLCSSA(Instruction *inst, const IRBuilder<> &BuilderM) {
    assert(inst->getName() != "<badref>");
    LoopContext lc;
    bool inLoop = getContext(inst->getParent(), lc);
    if (inLoop) {
      bool isChildLoop = false;

      BasicBlock *forwardBlock = BuilderM.GetInsertBlock();

      if (!isOriginalBlock(*forwardBlock)) {
        forwardBlock = originalForReverseBlock(*forwardBlock);
      }

      auto builderLoop = LI.getLoopFor(forwardBlock);
      while (builderLoop) {
        if (builderLoop->getHeader() == lc.header) {
          isChildLoop = true;
          break;
        }
        builderLoop = builderLoop->getParentLoop();
      }

      if (!isChildLoop) {
        // llvm::errs() << "manually performing lcssa for instruction" << *inst
        // << " in block " << BuilderM.GetInsertBlock()->getName() << "\n";
        if (!DT.dominates(inst, forwardBlock)) {
          llvm::errs() << *this->newFunc << "\n";
          llvm::errs() << *forwardBlock << "\n";
          llvm::errs() << *BuilderM.GetInsertBlock() << "\n";
          llvm::errs() << *inst << "\n";
        }
        assert(DT.dominates(inst, forwardBlock));

        for (auto pair : lcssaFixes[inst]) {
          if (DT.dominates(pair.first, forwardBlock)) {
            return pair.second;
          }
        }

        // TODO replace toplace with the first block dominated by inst, that
        // dominates (or is) forwardBlock
        //  for ensuring maximum reuse
        BasicBlock *toplace = forwardBlock;

        IRBuilder<> lcssa(&toplace->front());
        auto lcssaPHI = lcssa.CreatePHI(inst->getType(), 1,
                                        inst->getName() + "!manual_lcssa");
        for (auto pred : predecessors(toplace))
          lcssaPHI->addIncoming(inst, pred);

        lcssaFixes[inst][toplace] = lcssaPHI;
        return lcssaPHI;
      }
    }
    return inst;
  }

  Value *
  lookupM(Value *val, IRBuilder<> &BuilderM,
          const ValueToValueMapTy &incoming_availalble = ValueToValueMapTy());

  Value *invertPointerM(Value *val, IRBuilder<> &BuilderM);

  void branchToCorrespondingTarget(
      BasicBlock *ctx, IRBuilder<> &BuilderM,
      const std::map<BasicBlock *,
                     std::vector<std::pair</*pred*/ BasicBlock *,
                                           /*successor*/ BasicBlock *>>>
          &targetToPreds,
      const std::map<BasicBlock *, PHINode *> *replacePHIs = nullptr);
};

class DiffeGradientUtils : public GradientUtils {
  DiffeGradientUtils(Function *newFunc_, Function *oldFunc_,
                     TargetLibraryInfo &TLI, TypeAnalysis &TA, AAResults &AA,
                     ValueToValueMapTy &invertedPointers_,
                     const SmallPtrSetImpl<Value *> &constants_,
                     const SmallPtrSetImpl<Value *> &nonconstant_,
                     const SmallPtrSetImpl<Value *> &constantvalues_,
                     const SmallPtrSetImpl<Value *> &returnvals_,
                     ValueToValueMapTy &origToNew_, DerivativeMode mode)
      : GradientUtils(newFunc_, oldFunc_, TLI, TA, AA, invertedPointers_,
                      constants_, nonconstant_, constantvalues_, returnvals_,
                      origToNew_, mode) {
    prepareForReverse();
  }

public:
  ValueToValueMapTy differentials;
  static DiffeGradientUtils *
  CreateFromClone(bool topLevel, Function *todiff, TargetLibraryInfo &TLI,
                  TypeAnalysis &TA, AAResults &AA, DIFFE_TYPE retType,
                  const std::vector<DIFFE_TYPE> &constant_args,
                  ReturnType returnValue, Type *additionalArg);

private:
  Value *getDifferential(Value *val) {
    assert(val);
    if (auto arg = dyn_cast<Argument>(val))
      assert(arg->getParent() == oldFunc);
    if (auto inst = dyn_cast<Instruction>(val))
      assert(inst->getParent()->getParent() == oldFunc);
    assert(inversionAllocs);
    if (differentials.find(val) == differentials.end()) {
      IRBuilder<> entryBuilder(inversionAllocs);
      entryBuilder.setFastMathFlags(getFast());
      differentials[val] = entryBuilder.CreateAlloca(val->getType(), nullptr,
                                                     val->getName() + "'de");
      entryBuilder.CreateStore(Constant::getNullValue(val->getType()),
                               differentials[val]);
    }
    assert(cast<PointerType>(differentials[val]->getType())->getElementType() ==
           val->getType());
    return differentials[val];
  }

public:
  Value *diffe(Value *val, IRBuilder<> &BuilderM) {
    if (auto arg = dyn_cast<Argument>(val))
      assert(arg->getParent() == oldFunc);
    if (auto inst = dyn_cast<Instruction>(val))
      assert(inst->getParent()->getParent() == oldFunc);

    if (isConstantValue(val)) {
      llvm::errs() << *newFunc << "\n";
      llvm::errs() << *val << "\n";
    }
    if (val->getType()->isPointerTy()) {
      llvm::errs() << *newFunc << "\n";
      llvm::errs() << *val << "\n";
    }
    assert(!val->getType()->isPointerTy());
    assert(!val->getType()->isVoidTy());
    return BuilderM.CreateLoad(getDifferential(val));
  }

  // Returns created select instructions, if any
  std::vector<SelectInst *>
  addToDiffe(Value *val, Value *dif, IRBuilder<> &BuilderM, Type *addingType) {
    if (auto arg = dyn_cast<Argument>(val))
      assert(arg->getParent() == oldFunc);
    if (auto inst = dyn_cast<Instruction>(val))
      assert(inst->getParent()->getParent() == oldFunc);

    std::vector<SelectInst *> addedSelects;

    auto faddForNeg = [&](Value *old, Value *inc) {
      if (auto bi = dyn_cast<BinaryOperator>(inc)) {
        if (auto ci = dyn_cast<ConstantFP>(bi->getOperand(0))) {
          if (bi->getOpcode() == BinaryOperator::FSub && ci->isZero()) {
            return BuilderM.CreateFSub(old, bi->getOperand(1));
          }
        }
      }
      return BuilderM.CreateFAdd(old, inc);
    };

    auto faddForSelect = [&](Value *old, Value *dif) -> Value * {
      //! optimize fadd of select to select of fadd
      if (SelectInst *select = dyn_cast<SelectInst>(dif)) {
        if (Constant *ci = dyn_cast<Constant>(select->getTrueValue())) {
          if (ci->isZeroValue()) {
            SelectInst *res = cast<SelectInst>(BuilderM.CreateSelect(
                select->getCondition(), old,
                faddForNeg(old, select->getFalseValue())));
            addedSelects.emplace_back(res);
            return res;
          }
        }
        if (Constant *ci = dyn_cast<Constant>(select->getFalseValue())) {
          if (ci->isZeroValue()) {
            SelectInst *res = cast<SelectInst>(BuilderM.CreateSelect(
                select->getCondition(), faddForNeg(old, select->getTrueValue()),
                old));
            addedSelects.emplace_back(res);
            return res;
          }
        }
      }

      //! optimize fadd of bitcast select to select of bitcast fadd
      if (BitCastInst *bc = dyn_cast<BitCastInst>(dif)) {
        if (SelectInst *select = dyn_cast<SelectInst>(bc->getOperand(0))) {
          if (Constant *ci = dyn_cast<Constant>(select->getTrueValue())) {
            if (ci->isZeroValue()) {
              SelectInst *res = cast<SelectInst>(BuilderM.CreateSelect(
                  select->getCondition(), old,
                  faddForNeg(old, BuilderM.CreateCast(bc->getOpcode(),
                                                      select->getFalseValue(),
                                                      bc->getDestTy()))));
              addedSelects.emplace_back(res);
              return res;
            }
          }
          if (Constant *ci = dyn_cast<Constant>(select->getFalseValue())) {
            if (ci->isZeroValue()) {
              SelectInst *res = cast<SelectInst>(BuilderM.CreateSelect(
                  select->getCondition(),
                  faddForNeg(old, BuilderM.CreateCast(bc->getOpcode(),
                                                      select->getTrueValue(),
                                                      bc->getDestTy())),
                  old));
              addedSelects.emplace_back(res);
              return res;
            }
          }
        }
      }

      // fallback
      return faddForNeg(old, dif);
    };

    if (val->getType()->isPointerTy()) {
      llvm::errs() << *newFunc << "\n";
      llvm::errs() << *val << "\n";
    }
    if (isConstantValue(val)) {
      llvm::errs() << *newFunc << "\n";
      llvm::errs() << *val << "\n";
    }
    assert(!val->getType()->isPointerTy());
    assert(!isConstantValue(val));
    if (val->getType() != dif->getType()) {
      llvm::errs() << "val: " << *val << " dif: " << *dif << "\n";
    }
    assert(val->getType() == dif->getType());
    auto old = diffe(val, BuilderM);
    assert(val->getType() == old->getType());
    Value *res = nullptr;
    if (val->getType()->isIntOrIntVectorTy()) {
      if (!addingType) {
        llvm::errs() << "module: " << *oldFunc->getParent() << "\n";
        llvm::errs() << "oldFunc: " << *oldFunc << "\n";
        llvm::errs() << "newFunc: " << *newFunc << "\n";
        llvm::errs() << "val: " << *val << "\n";
      }
      assert(addingType);
      assert(addingType->isFPOrFPVectorTy());

      auto oldBitSize = oldFunc->getParent()->getDataLayout().getTypeSizeInBits(
          old->getType());
      auto newBitSize =
          oldFunc->getParent()->getDataLayout().getTypeSizeInBits(addingType);

      if (oldBitSize > newBitSize && oldBitSize % newBitSize == 0 &&
          !addingType->isVectorTy()) {
        #if LLVM_VERSION_MAJOR >= 11
        addingType = VectorType::get(addingType, oldBitSize / newBitSize, false);
        #else
        addingType = VectorType::get(addingType, oldBitSize / newBitSize);
        #endif
      }

      Value *bcold = BuilderM.CreateBitCast(old, addingType);
      Value *bcdif = BuilderM.CreateBitCast(dif, addingType);

      res = faddForSelect(bcold, bcdif);
      if (Instruction *oldinst = dyn_cast<Instruction>(bcold)) {
        if (oldinst->getNumUses() == 0) {
          // if (oldinst == &*BuilderM.GetInsertPoint())
          // BuilderM.SetInsertPoint(oldinst->getNextNode());
          // oldinst->eraseFromParent();
        }
      }
      if (Instruction *difinst = dyn_cast<Instruction>(bcdif)) {
        if (difinst->getNumUses() == 0) {
          // if (difinst == &*BuilderM.GetInsertPoint())
          // BuilderM.SetInsertPoint(difinst->getNextNode());
          // difinst->eraseFromParent();
        }
      }
      if (SelectInst *select = dyn_cast<SelectInst>(res)) {
        assert(addedSelects.back() == select);
        addedSelects.erase(addedSelects.end() - 1);
        res = BuilderM.CreateSelect(
            select->getCondition(),
            BuilderM.CreateBitCast(select->getTrueValue(), val->getType()),
            BuilderM.CreateBitCast(select->getFalseValue(), val->getType()));
        assert(select->getNumUses() == 0);
        // if (select == &*BuilderM.GetInsertPoint())
        // BuilderM.SetInsertPoint(select->getNextNode());
        // select->eraseFromParent();
      } else {
        res = BuilderM.CreateBitCast(res, val->getType());
      }
      BuilderM.CreateStore(res, getDifferential(val));
      // store->setAlignment(align);
      return addedSelects;
    } else if (val->getType()->isFPOrFPVectorTy()) {
      // TODO consider adding type
      res = faddForSelect(old, dif);

      BuilderM.CreateStore(res, getDifferential(val));
      // store->setAlignment(align);
      return addedSelects;
    } else if (val->getType()->isStructTy()) {
      auto st = cast<StructType>(val->getType());
      for (unsigned i = 0; i < st->getNumElements(); ++i) {
        Value *v = ConstantInt::get(Type::getInt32Ty(st->getContext()), i);
        SelectInst *addedSelect = addToDiffeIndexed(
            val, BuilderM.CreateExtractValue(dif, {i}), {v}, BuilderM);
        if (addedSelect) {
          addedSelects.push_back(addedSelect);
        }
      }
      return addedSelects;
    } else {
      llvm_unreachable("unknown type to add to diffe");
      exit(1);
    }
  }

  void setDiffe(Value *val, Value *toset, IRBuilder<> &BuilderM) {
    if (auto arg = dyn_cast<Argument>(val))
      assert(arg->getParent() == oldFunc);
    if (auto inst = dyn_cast<Instruction>(val))
      assert(inst->getParent()->getParent() == oldFunc);
    if (isConstantValue(val)) {
      llvm::errs() << *newFunc << "\n";
      llvm::errs() << *val << "\n";
    }
    assert(!isConstantValue(val));
    Value *tostore = getDifferential(val);
    if (toset->getType() !=
        cast<PointerType>(tostore->getType())->getElementType()) {
      llvm::errs() << "toset:" << *toset << "\n";
      llvm::errs() << "tostore:" << *tostore << "\n";
    }
    assert(toset->getType() ==
           cast<PointerType>(tostore->getType())->getElementType());
    BuilderM.CreateStore(toset, tostore);
  }

  SelectInst *addToDiffeIndexed(Value *val, Value *dif, ArrayRef<Value *> idxs,
                                IRBuilder<> &BuilderM) {
    if (auto arg = dyn_cast<Argument>(val))
      assert(arg->getParent() == oldFunc);
    if (auto inst = dyn_cast<Instruction>(val))
      assert(inst->getParent()->getParent() == oldFunc);
    assert(!isConstantValue(val));
    SmallVector<Value *, 4> sv;
    sv.push_back(ConstantInt::get(Type::getInt32Ty(val->getContext()), 0));
    for (auto i : idxs)
      sv.push_back(i);
    Value *ptr = BuilderM.CreateGEP(getDifferential(val), sv);
    cast<GetElementPtrInst>(ptr)->setIsInBounds(true);
    Value *old = BuilderM.CreateLoad(ptr);

    Value *res = nullptr;

    if (old->getType()->isIntOrIntVectorTy()) {
      res = BuilderM.CreateFAdd(
          BuilderM.CreateBitCast(old, IntToFloatTy(old->getType())),
          BuilderM.CreateBitCast(dif, IntToFloatTy(dif->getType())));
      res = BuilderM.CreateBitCast(res, old->getType());
    } else if (old->getType()->isFPOrFPVectorTy()) {
      res = BuilderM.CreateFAdd(old, dif);
    } else {
      assert(old);
      assert(dif);
      llvm::errs() << *newFunc << "\n"
                   << "cannot handle type " << *old << "\n"
                   << *dif;
      assert(0 && "cannot handle type");
      report_fatal_error("cannot handle type");
    }

    SelectInst *addedSelect = nullptr;

    //! optimize fadd of select to select of fadd
    // TODO: Handle Selects of ints
    if (SelectInst *select = dyn_cast<SelectInst>(dif)) {
      if (ConstantFP *ci = dyn_cast<ConstantFP>(select->getTrueValue())) {
        if (ci->isZero()) {
          cast<Instruction>(res)->eraseFromParent();
          res = BuilderM.CreateSelect(
              select->getCondition(), old,
              BuilderM.CreateFAdd(old, select->getFalseValue()));
          addedSelect = cast<SelectInst>(res);
          goto endselect;
        }
      }
      if (ConstantFP *ci = dyn_cast<ConstantFP>(select->getFalseValue())) {
        if (ci->isZero()) {
          cast<Instruction>(res)->eraseFromParent();
          res = BuilderM.CreateSelect(
              select->getCondition(),
              BuilderM.CreateFAdd(old, select->getTrueValue()), old);
          addedSelect = cast<SelectInst>(res);
          goto endselect;
        }
      }
    }
  endselect:;

    BuilderM.CreateStore(res, ptr);
    return addedSelect;
  }

//! align is the alignment that should be specified for load/store to pointer
#if LLVM_VERSION_MAJOR >= 10
  void addToInvertedPtrDiffe(Value *ptr, Value *dif, IRBuilder<> &BuilderM,
                             MaybeAlign align){
#else
  void addToInvertedPtrDiffe(Value *ptr, Value *dif, IRBuilder<> &BuilderM,
                             unsigned align) {
#endif
      if (!(ptr->getType()->isPointerTy()) ||
          !(cast<PointerType>(ptr->getType())->getElementType() ==
            dif->getType())) {
        llvm::errs() << *oldFunc << "\n";
  llvm::errs() << *newFunc << "\n";
  llvm::errs() << "Ptr: " << *ptr << "\n";
  llvm::errs() << "Diff: " << *dif << "\n";
} assert(ptr->getType()->isPointerTy());
assert(cast<PointerType>(ptr->getType())->getElementType() == dif->getType());

assert(ptr->getType()->isPointerTy());
assert(cast<PointerType>(ptr->getType())->getElementType() == dif->getType());

Value *res;
LoadInst *old = BuilderM.CreateLoad(ptr);
#if LLVM_VERSION_MAJOR >= 10
if (align)
  old->setAlignment(align.getValue());
#else
    old->setAlignment(align);
#endif

if (old->getType()->isIntOrIntVectorTy()) {
  res = BuilderM.CreateFAdd(
      BuilderM.CreateBitCast(old, IntToFloatTy(old->getType())),
      BuilderM.CreateBitCast(dif, IntToFloatTy(dif->getType())));
  res = BuilderM.CreateBitCast(res, old->getType());
} else if (old->getType()->isFPOrFPVectorTy()) {
  res = BuilderM.CreateFAdd(old, dif);
} else {
  assert(old);
  assert(dif);
  llvm::errs() << *newFunc << "\n"
               << "cannot handle type " << *old << "\n"
               << *dif;
  assert(0 && "cannot handle type");
  report_fatal_error("cannot handle type");
}
StoreInst *st = BuilderM.CreateStore(res, ptr);
#if LLVM_VERSION_MAJOR >= 10
if (align)
  st->setAlignment(align.getValue());
#else
    st->setAlignment(align);
#endif
}
}
;
#endif
