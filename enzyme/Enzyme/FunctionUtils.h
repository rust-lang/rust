/*
 * FunctionUtils.h
 *
 * Copyright (C) 2019 William S. Moses (enzyme@wsmoses.com) - All Rights Reserved
 *
 * For commercial use of this code please contact the author(s) above.
 *
 * For research use of the code please use the following citation.
 *
 * \misc{mosesenzyme,
    author = {William S. Moses, Tim Kaler},
    title = {Enzyme: LLVM Automatic Differentiation},
    year = {2019},
    howpublished = {\url{https://github.com/wsmoses/Enzyme/}},
    note = {commit xxxxxxx}
 */

#ifndef ENZYME_FUNCTION_UTILS_H
#define ENZYME_FUNCTION_UTILS_H

#include <set>

#include "SCEV/ScalarEvolutionExpander.h"

#include "Utils.h"

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/TargetLibraryInfo.h"

#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"

#include "llvm/Transforms/Utils/ValueMapper.h"
#include "llvm/IR/Instructions.h"

llvm::Function* preprocessForClone(llvm::Function *F, llvm::AAResults &AA, llvm::TargetLibraryInfo &TLI, bool topLevel);

llvm::Function *CloneFunctionWithReturns(bool topLevel, llvm::Function *&F, llvm::AAResults &AA, llvm::TargetLibraryInfo &TLI, llvm::ValueToValueMapTy& ptrInputs,
                                   const std::set<unsigned>& constant_args, llvm::SmallPtrSetImpl<llvm::Value*> &constants, llvm::SmallPtrSetImpl<llvm::Value*> &nonconstant,
                                   llvm::SmallPtrSetImpl<llvm::Value*> &returnvals, ReturnType returnValue, bool differentialReturn, llvm::Twine name,
                                   llvm::ValueToValueMapTy *VMapO, bool diffeReturnArg, llvm::Type* additionalArg = nullptr);

class GradientUtils;

llvm::PHINode* canonicalizeIVs(llvm::fake::SCEVExpander &exp, llvm::Type *Ty, llvm::Loop *L, llvm::DominatorTree &DT, GradientUtils* gutils);

void forceRecursiveInlining(llvm::Function *NewF, const llvm::Function* F);

void optimizeIntermediate(GradientUtils* gutils, bool topLevel, llvm::Function *F);

static inline void getExitBlocks(const llvm::Loop *L, llvm::SmallPtrSetImpl<llvm::BasicBlock*>& ExitBlocks) {
    llvm::SmallVector<llvm::BasicBlock *, 8> PotentialExitBlocks;
    L->getExitBlocks(PotentialExitBlocks);
    for(auto a:PotentialExitBlocks) {

        llvm::SmallVector<llvm::BasicBlock*, 4> tocheck;
        llvm::SmallPtrSet<llvm::BasicBlock*, 4> checked;
        tocheck.push_back(a);

        bool isExit = false;

        while(tocheck.size()) {
            auto foo = tocheck.back();
            tocheck.pop_back();
            if (checked.count(foo)) {
                isExit = true;
                goto exitblockcheck;
            }
            checked.insert(foo);
            if(auto bi = llvm::dyn_cast<llvm::BranchInst>(foo->getTerminator())) {
                for(auto nb : bi->successors()) {
                    if (L->contains(nb)) continue;
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

static inline llvm::SmallVector<llvm::BasicBlock*, 3> getLatches(const llvm::Loop *L, const llvm::SmallPtrSetImpl<llvm::BasicBlock*>& ExitBlocks ) {
    llvm::BasicBlock *Preheader = L->getLoopPreheader();
    if (!Preheader) {
        llvm::errs() << *L->getHeader()->getParent() << "\n";
        llvm::errs() << *L->getHeader() << "\n";
        llvm::errs() << *L << "\n";
    }
    assert(Preheader && "requires preheader");

    // Find latch, defined as a (perhaps unique) block in loop that branches to exit block
    llvm::SmallVector<llvm::BasicBlock *, 3> Latches;
    for (llvm::BasicBlock* ExitBlock : ExitBlocks) {
        for (llvm::BasicBlock* pred : llvm::predecessors(ExitBlock)) {
            if (L->contains(pred)) {
                if (std::find(Latches.begin(), Latches.end(), pred) != Latches.end()) continue;
                Latches.push_back(pred);
            }
        }
    }
    return Latches;
}
#endif
