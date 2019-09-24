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

llvm::Function* preprocessForClone(llvm::Function *F, llvm::AAResults &AA, llvm::TargetLibraryInfo &TLI);

llvm::Function *CloneFunctionWithReturns(llvm::Function *&F, llvm::AAResults &AA, llvm::TargetLibraryInfo &TLI, llvm::ValueToValueMapTy& ptrInputs,
                                   const std::set<unsigned>& constant_args, llvm::SmallPtrSetImpl<llvm::Value*> &constants, llvm::SmallPtrSetImpl<llvm::Value*> &nonconstant,
                                   llvm::SmallPtrSetImpl<llvm::Value*> &returnvals, ReturnType returnValue, bool differentialReturn, llvm::Twine name,
                                   llvm::ValueToValueMapTy *VMapO, bool diffeReturnArg, llvm::Type* additionalArg = nullptr);

class GradientUtils;

llvm::PHINode* canonicalizeIVs(llvm::fake::SCEVExpander &exp, llvm::Type *Ty, llvm::Loop *L, llvm::DominatorTree &DT, GradientUtils* gutils);

void forceRecursiveInlining(llvm::Function *NewF, const llvm::Function* F);

void optimizeIntermediate(GradientUtils* gutils, bool topLevel, llvm::Function *F);

#endif
