//===- FunctionUtils.h - Declaration of function utilities ---------------===//
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
// This file declares utilities on LLVM Functions that are used as part of the
// AD process.
//
//===----------------------------------------------------------------------===//
#ifndef ENZYME_FUNCTION_UTILS_H
#define ENZYME_FUNCTION_UTILS_H

#include <deque>
#include <set>

#include "SCEV/ScalarEvolution.h"
#include "SCEV/ScalarEvolutionExpander.h"

#include "Utils.h"

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/TargetLibraryInfo.h"

#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"

#include "llvm/IR/Instructions.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

//;

class PreProcessCache {
public:
  PreProcessCache();

  llvm::FunctionAnalysisManager FAM;
  llvm::ModuleAnalysisManager MAM;

  std::map<std::pair<llvm::Function *, DerivativeMode>, llvm::Function *> cache;
  std::map<llvm::Function *, llvm::Function *> CloneOrigin;

  llvm::Function *preprocessForClone(llvm::Function *F, DerivativeMode mode);

  llvm::AAResults &getAAResultsFromFunction(llvm::Function *NewF);

  llvm::Function *CloneFunctionWithReturns(
      DerivativeMode mode, unsigned width, llvm::Function *&F,
      llvm::ValueToValueMapTy &ptrInputs,
      llvm::ArrayRef<DIFFE_TYPE> constant_args,
      llvm::SmallPtrSetImpl<llvm::Value *> &constants,
      llvm::SmallPtrSetImpl<llvm::Value *> &nonconstant,
      llvm::SmallPtrSetImpl<llvm::Value *> &returnvals, ReturnType returnValue,
      DIFFE_TYPE returnType, llvm::Twine name, llvm::ValueToValueMapTy *VMapO,
      bool diffeReturnArg, llvm::Type *additionalArg = nullptr);

  void ReplaceReallocs(llvm::Function *NewF, bool mem2reg = false);
  void LowerAllocAddr(llvm::Function *NewF);
  void AlwaysInline(llvm::Function *NewF);
  void optimizeIntermediate(llvm::Function *F);

  void clear();
};

class GradientUtils;

static inline void
getExitBlocks(const llvm::Loop *L,
              llvm::SmallPtrSetImpl<llvm::BasicBlock *> &ExitBlocks) {
  llvm::SmallVector<llvm::BasicBlock *, 8> PotentialExitBlocks;
  L->getExitBlocks(PotentialExitBlocks);
  for (auto a : PotentialExitBlocks) {

    llvm::SmallVector<llvm::BasicBlock *, 4> tocheck;
    llvm::SmallPtrSet<llvm::BasicBlock *, 4> checked;
    tocheck.push_back(a);

    bool isExit = false;

    while (tocheck.size()) {
      auto foo = tocheck.back();
      tocheck.pop_back();
      if (checked.count(foo)) {
        isExit = true;
        goto exitblockcheck;
      }
      checked.insert(foo);
      if (auto bi = llvm::dyn_cast<llvm::BranchInst>(foo->getTerminator())) {
        for (auto nb : bi->successors()) {
          if (L->contains(nb))
            continue;
          tocheck.push_back(nb);
        }
      } else if (llvm::isa<llvm::UnreachableInst>(foo->getTerminator())) {
        continue;
      } else {
        isExit = true;
        goto exitblockcheck;
      }
    }

  exitblockcheck:
    if (isExit) {
      ExitBlocks.insert(a);
    }
  }
}

static inline llvm::SmallVector<llvm::BasicBlock *, 3>
getLatches(const llvm::Loop *L,
           const llvm::SmallPtrSetImpl<llvm::BasicBlock *> &ExitBlocks) {
  llvm::BasicBlock *Preheader = L->getLoopPreheader();
  if (!Preheader) {
    llvm::errs() << *L->getHeader()->getParent() << "\n";
    llvm::errs() << *L->getHeader() << "\n";
    llvm::errs() << *L << "\n";
  }
  assert(Preheader && "requires preheader");

  // Find latch, defined as a (perhaps unique) block in loop that branches to
  // exit block
  llvm::SmallVector<llvm::BasicBlock *, 3> Latches;
  for (llvm::BasicBlock *ExitBlock : ExitBlocks) {
    for (llvm::BasicBlock *pred : llvm::predecessors(ExitBlock)) {
      if (L->contains(pred)) {
        if (std::find(Latches.begin(), Latches.end(), pred) != Latches.end())
          continue;
        Latches.push_back(pred);
      }
    }
  }
  return Latches;
}

// TODO note this doesn't go through [loop, unreachable], and we could get more
// performance by doing this can consider doing some domtree magic potentially
static inline llvm::SmallPtrSet<llvm::BasicBlock *, 4>
getGuaranteedUnreachable(llvm::Function *F) {
  llvm::SmallPtrSet<llvm::BasicBlock *, 4> knownUnreachables;
  std::deque<llvm::BasicBlock *> todo;
  for (auto &BB : *F) {
    todo.push_back(&BB);
  }

  while (!todo.empty()) {
    llvm::BasicBlock *next = todo.front();
    todo.pop_front();

    if (knownUnreachables.find(next) != knownUnreachables.end())
      continue;

    if (llvm::isa<llvm::ReturnInst>(next->getTerminator()))
      continue;

    if (llvm::isa<llvm::UnreachableInst>(next->getTerminator())) {
      knownUnreachables.insert(next);
      for (llvm::BasicBlock *Pred : predecessors(next)) {
        todo.push_back(Pred);
      }
      continue;
    }

    // Assume resumes don't happen
    // TODO consider EH
    if (llvm::isa<llvm::ResumeInst>(next->getTerminator())) {
      knownUnreachables.insert(next);
      for (llvm::BasicBlock *Pred : predecessors(next)) {
        todo.push_back(Pred);
      }
      continue;
    }

    bool unreachable = true;
    for (llvm::BasicBlock *Succ : llvm::successors(next)) {
      if (knownUnreachables.find(Succ) == knownUnreachables.end()) {
        unreachable = false;
        break;
      }
    }

    if (!unreachable)
      continue;
    knownUnreachables.insert(next);
    for (llvm::BasicBlock *Pred : llvm::predecessors(next)) {
      todo.push_back(Pred);
    }
    continue;
  }

  return knownUnreachables;
}

enum class UseReq {
  Need,
  Recur,
  Cached,
};
static inline void calculateUnusedValues(
    const llvm::Function &oldFunc,
    llvm::SmallPtrSetImpl<const llvm::Value *> &unnecessaryValues,
    llvm::SmallPtrSetImpl<const llvm::Instruction *> &unnecessaryInstructions,
    bool returnValue, std::function<bool(const llvm::Value *)> valneeded,
    std::function<UseReq(const llvm::Instruction *)> instneeded) {

  std::deque<const llvm::Instruction *> todo;

  for (const llvm::BasicBlock &BB : oldFunc) {
    if (auto ri = llvm::dyn_cast<llvm::ReturnInst>(BB.getTerminator())) {
      if (!returnValue) {
        unnecessaryInstructions.insert(ri);
      }
    }
    for (auto &inst : BB) {
      if (&inst == BB.getTerminator())
        continue;
      todo.push_back(&inst);
    }
  }

  while (!todo.empty()) {
    auto inst = todo.front();
    todo.pop_front();

    if (unnecessaryInstructions.count(inst)) {
      assert(unnecessaryValues.count(inst));
      continue;
    }

    if (unnecessaryValues.count(inst))
      continue;

    if (valneeded(inst))
      continue;

    bool necessaryUse = false;

    llvm::SmallPtrSet<const llvm::Instruction *, 4> seen;
    std::deque<const llvm::Instruction *> users;

    for (auto user_dtx : inst->users()) {
      if (auto cst = llvm::dyn_cast<llvm::Instruction>(user_dtx)) {
        users.push_back(cst);
      }
    }

    while (users.size()) {
      auto val = users.front();
      users.pop_front();

      if (seen.count(val))
        continue;
      seen.insert(val);

      if (unnecessaryInstructions.count(val))
        continue;

      switch (instneeded(val)) {
      case UseReq::Need:
        necessaryUse = true;
        break;
      case UseReq::Recur:
        for (auto user_dtx : val->users()) {
          if (auto cst = llvm::dyn_cast<llvm::Instruction>(user_dtx)) {
            users.push_back(cst);
          }
        }
        break;
      case UseReq::Cached:
        break;
      }
      if (necessaryUse)
        break;
    }

    if (necessaryUse)
      continue;

    unnecessaryValues.insert(inst);

    if (instneeded(inst) == UseReq::Need)
      continue;

    unnecessaryInstructions.insert(inst);

    for (auto &operand : inst->operands()) {
      if (auto usedinst = llvm::dyn_cast<llvm::Instruction>(operand.get())) {
        todo.push_back(usedinst);
      }
    }
  }

  if (false && oldFunc.getName().endswith("subfn")) {
    llvm::errs() << "Prepping values for: " << oldFunc.getName()
                 << " returnValue: " << returnValue << "\n";
    for (auto v : unnecessaryInstructions) {
      llvm::errs() << "+ unnecessaryInstructions: " << *v << "\n";
    }
    for (auto v : unnecessaryValues) {
      llvm::errs() << "+ unnecessaryValues: " << *v << "\n";
    }
    llvm::errs() << "</end>\n";
  }
}

static inline void calculateUnusedStores(
    const llvm::Function &oldFunc,
    llvm::SmallPtrSetImpl<const llvm::Instruction *> &unnecessaryStores,
    std::function<bool(const llvm::Instruction *)> needStore) {

  std::deque<const llvm::Instruction *> todo;

  for (const llvm::BasicBlock &BB : oldFunc) {
    for (auto &inst : BB) {
      if (&inst == BB.getTerminator())
        continue;
      todo.push_back(&inst);
    }
  }

  while (!todo.empty()) {
    auto inst = todo.front();
    todo.pop_front();

    if (unnecessaryStores.count(inst)) {
      continue;
    }

    if (needStore(inst))
      continue;

    unnecessaryStores.insert(inst);
  }
}

void RecursivelyReplaceAddressSpace(llvm::Value *AI, llvm::Value *rep,
                                    bool legal);

void ReplaceFunctionImplementation(llvm::Module &M);

/// Is the use of value val as an argument of call CI potentially captured
bool couldFunctionArgumentCapture(llvm::CallInst *CI, llvm::Value *val);

llvm::FunctionType *getFunctionTypeForClone(
    llvm::FunctionType *FTy, DerivativeMode mode, unsigned width,
    llvm::Type *additionalArg, llvm::ArrayRef<DIFFE_TYPE> constant_args,
    bool diffeReturnArg, ReturnType returnValue, DIFFE_TYPE returnType);

#endif
