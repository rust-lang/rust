/*
 * EnzymeLogic.h
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

#ifndef ENZYME_LOGIC_H
#define ENZYME_LOGIC_H

#include <set>
#include <utility>

#include "SCEV/ScalarEvolutionExpander.h"

#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/CommandLine.h"

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/TargetLibraryInfo.h"

extern llvm::cl::opt<bool> enzyme_print;

//! return structtype if recursive function
std::pair<llvm::Function*,llvm::StructType*> CreateAugmentedPrimal(llvm::Function* todiff, llvm::AAResults &AA, const std::set<unsigned>& constant_args, llvm::TargetLibraryInfo &TLI, bool differentialReturn);

llvm::Function* CreatePrimalAndGradient(llvm::Function* todiff, const std::set<unsigned>& constant_args, llvm::TargetLibraryInfo &TLI, llvm::AAResults &AA, bool returnValue, bool differentialReturn, bool dretPtr, bool topLevel, llvm::Type* additionalArg, std::set<unsigned> volatile_args);

#endif
