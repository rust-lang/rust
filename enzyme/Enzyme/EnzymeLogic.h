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
class AugmentedReturn {
public:
    llvm::Function *fn;
    //! return structtype if recursive function
    llvm::StructType* tapeType;
    std::map<std::pair<llvm::Instruction*, std::string>, unsigned> tapeIndices;
    //! Map from original call to sub augmentation data
    std::map<llvm::CallInst*, const AugmentedReturn*> subaugmentations;
    AugmentedReturn(llvm::Function* fn, llvm::StructType* tapeType) : fn(fn), tapeType(tapeType), tapeIndices() {}
    AugmentedReturn(llvm::Function* fn, llvm::StructType* tapeType, std::map<std::pair<llvm::Instruction*, std::string>, unsigned> tapeIndices) : fn(fn), tapeType(tapeType), tapeIndices(tapeIndices) {}
};

const AugmentedReturn& CreateAugmentedPrimal(llvm::Function* todiff, llvm::AAResults &global_AA, const std::set<unsigned>& constant_args, llvm::TargetLibraryInfo &TLI, bool differentialReturn, bool returnUsed, const std::set<unsigned> _uncacheable_args);

llvm::Function* CreatePrimalAndGradient(llvm::Function* todiff, const std::set<unsigned>& constant_args, llvm::TargetLibraryInfo &TLI, llvm::AAResults &global_AA, bool returnValue, bool differentialReturn, bool dretPtr, bool topLevel, llvm::Type* additionalArg, std::set<unsigned> _uncacheable_args, llvm::Optional<std::map<std::pair<llvm::Instruction*, std::string>, unsigned>> index_map);

#endif
