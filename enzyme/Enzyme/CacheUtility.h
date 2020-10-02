//===- CacheUtility.h - Caching values in the forward pass for later use  ---===//
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
// This file declares a base helper class CacheUtility that manages the cache
// of values from the forward pass for later use.
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_CACHE_UTILITY_H
#define ENZYME_CACHE_UTILITY_H

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Instructions.h"

#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/ValueMapper.h"


#include "MustExitScalarEvolution.h"

/// Container for all loop information to synthesize gradients
struct LoopContext {
  /// Canonical induction variable of the loop
  llvm::PHINode *var;

  /// Increment of the induction
  llvm::Instruction *incvar;

  /// Allocation of induction variable of reverse pass
  llvm::AllocaInst *antivaralloc;

  /// Header of this loop
  llvm::BasicBlock *header;

  /// Preheader of this loop
  llvm::BasicBlock *preheader;

  /// Whether this loop has a statically analyzable number of iterations
  bool dynamic;

  /// limit is last value of a canonical induction variable
  /// iters is number of times loop is run (thus iters = limit + 1)
  llvm::Value *limit;

  /// All blocks this loop exits too
  llvm::SmallPtrSet<llvm::BasicBlock *, 8> exitBlocks;

  /// Parent loop of this loop
  llvm::Loop *parent;
};
static inline bool operator==(const LoopContext &lhs, const LoopContext &rhs) {
  return lhs.parent == rhs.parent;
}

/// Pack 8 bools together in a single byte
extern llvm::cl::opt<bool> EfficientBoolCache;

/// Modes of potential unwraps
enum class UnwrapMode {
  // It is already known that it is legal to fully unwrap
  // this instruction. This means unwrap this instruction,
  // its operands, etc
  LegalFullUnwrap,
  // Attempt to fully unwrap this, looking up whenever it
  // is not legal to unwrap
  AttemptFullUnwrapWithLookup,
  // Attempt to fully unwrap this
  AttemptFullUnwrap,
  // Unwrap the current instruction but not its operand
  AttemptSingleUnwrap,
};


class CacheUtility {
public:
  llvm::Function *const newFunc;
  std::map<llvm::Loop *, LoopContext> loopContexts;

public:
  llvm::TargetLibraryInfo &TLI;
  llvm::DominatorTree DT;
  llvm::LoopInfo LI;
  llvm::AssumptionCache AC;
  MustExitScalarEvolution SE;

  llvm::BasicBlock *inversionAllocs;

protected:
  std::map<std::pair<llvm::Value *, int>, llvm::MDNode *> invariantGroups;
  std::map<llvm::Value *, llvm::MDNode *> valueInvariantGroups;

public:
  // Context information to request calculation of loop limit information
  struct LimitContext {
      // A block inside of the loop, defining the location
      llvm::BasicBlock* Block;
      // Currently unused experimental information (see getSubLimit/lookupM)
      bool Experimental;

      LimitContext(llvm::BasicBlock* Block, bool Experimental=false) : Block(Block), Experimental(Experimental) {}
  };
  std::map<llvm::Value *, std::pair<llvm::AllocaInst *, LimitContext>> scopeMap;

  std::map<llvm::AllocaInst *, std::vector<llvm::Value *>> scopeStores;
protected:
  std::map<llvm::AllocaInst *, std::set<llvm::CallInst *>> scopeFrees;
  std::map<llvm::AllocaInst *, std::vector<llvm::CallInst *>> scopeAllocs;

protected:
  CacheUtility(llvm::TargetLibraryInfo &TLI, llvm::Function* newFunc) : newFunc(newFunc),
    TLI(TLI), DT(*newFunc), LI(DT), AC(*newFunc), SE(*newFunc, TLI, AC, DT, LI) {
    inversionAllocs = llvm::BasicBlock::Create(newFunc->getContext(),
                                         "allocsForInversion", newFunc);
  }
public:
  virtual ~CacheUtility();
  
  bool getContext(llvm::BasicBlock *BB, LoopContext &loopContext);
  
  void dumpScope() {
    llvm::errs() << "scope:\n";
    for (auto a : scopeMap) {
      llvm::errs() << "   scopeMap[" << *a.first << "] = " << *a.second.first
                   << " ctx:" << a.second.second.Block->getName() << "\n";
    }
    llvm::errs() << "end scope\n";
  }

  virtual void erase(llvm::Instruction* I) {
    using namespace llvm;
    assert(I);

    for (auto v : scopeMap) {
      if (v.second.first == I) {
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
          llvm::errs() << *newFunc << "\n";
          llvm::errs() << *v.first << "\n";
          llvm::errs() << *I << "\n";
          assert(0 && "erasing something in scopeFrees map");
        }
      }
    if (auto ci = dyn_cast<CallInst>(I))
      for (auto v : scopeAllocs) {
        if (std::find(v.second.begin(), v.second.end(), ci) != v.second.end()) {
          llvm::errs() << *newFunc << "\n";
          llvm::errs() << *v.first << "\n";
          llvm::errs() << *I << "\n";
          assert(0 && "erasing something in scopeAllocs map");
        }
      }
    for (auto v : scopeStores) {
      if (std::find(v.second.begin(), v.second.end(), I) != v.second.end()) {
        llvm::errs() << *newFunc << "\n";
        llvm::errs() << *v.first << "\n";
        llvm::errs() << *I << "\n";
        assert(0 && "erasing something in scopeStores map");
      }
    }

    auto found = scopeMap.find(I);
    if (found != scopeMap.end()) {
      scopeFrees.erase(found->second.first);
      scopeAllocs.erase(found->second.first);
      scopeStores.erase(found->second.first);
    }
    if (auto ai = dyn_cast<AllocaInst>(I)) {
      scopeFrees.erase(ai);
      scopeAllocs.erase(ai);
      scopeStores.erase(ai);
    }
    scopeMap.erase(I);
    SE.eraseValueFromMap(I);

    if (!I->use_empty()) {
      llvm::errs() << *newFunc << "\n";
      llvm::errs() << *I << "\n";
    }
    assert(I->use_empty());
    I->eraseFromParent();
  }
  
  //! returns true indices
  typedef std::vector<std::pair</*sublimit*/ llvm::Value *, /*loop limits*/ std::vector<
                        std::pair<LoopContext, llvm::Value *>>>> SubLimitType;
  SubLimitType getSubLimits(LimitContext ctx);

  llvm::AllocaInst *createCacheForScope(LimitContext ctx, llvm::Type *T,llvm::StringRef name,
                                bool shouldFree, bool allocateInternal = true,
                                llvm::Value *extraSize = nullptr);


    /// if full unwrap, don't just unwrap this instruction, but also its operands, etc
    virtual llvm::Value * unwrapM(llvm::Value *const val, llvm::IRBuilder<> &BuilderM,
                    const llvm::ValueToValueMapTy &available, UnwrapMode mode) = 0;

    virtual llvm::Value * lookupM(llvm::Value *val, llvm::IRBuilder<> &BuilderM,
            const llvm::ValueToValueMapTy &incoming_availalble = llvm::ValueToValueMapTy()) = 0;

    virtual void freeCache(llvm::BasicBlock* forwardPreheader, const SubLimitType& antimap, int i, llvm::AllocaInst* alloc, llvm::ConstantInt* byteSizeOfType, llvm::Value* storeInto) {
        assert(0 && "freeing cache not handled in this scenario");
        llvm_unreachable("freeing cache not handled in this scenario");
    } 

    void storeInstructionInCache(LimitContext ctx, llvm::IRBuilder<> &BuilderM,
                llvm::Value *val, llvm::AllocaInst *cache);

    void storeInstructionInCache(LimitContext ctx, llvm::Instruction *inst,
                                llvm::AllocaInst *cache);

    llvm::Value *getCachePointer(bool inForwardPass, llvm::IRBuilder<> &BuilderM, LimitContext ctx, llvm::Value *cache,
                            bool isi1, bool storeInStoresMap = false,
                            llvm::Value *extraSize = nullptr);

    llvm::Value *lookupValueFromCache(bool inForwardPass, llvm::IRBuilder<> &BuilderM, LimitContext ctx,
                            llvm::Value *cache, bool isi1,
                            llvm::Value *extraSize = nullptr,
                            llvm::Value *extraOffset = nullptr);
};

#endif