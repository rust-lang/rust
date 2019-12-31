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

#include "ActiveVariable.h"

extern llvm::cl::opt<bool> enzyme_print;

enum class AugmentedStruct {
    Tape,
    Return,
    DifferentialReturn
};

//! return structtype if recursive function
class AugmentedReturn {
public:
    llvm::Function *fn;
    //! return structtype if recursive function
    llvm::StructType* tapeType;
    std::map<std::pair<llvm::Instruction*, std::string>, unsigned> tapeIndices;
    //! Map from original call to sub augmentation data
    std::map<const llvm::CallInst*, const AugmentedReturn*> subaugmentations;
    //! Map from information desired from a augmented return to its index in the returned struct
    std::map<AugmentedStruct, unsigned> returns;

    std::map<llvm::CallInst*, const std::map<llvm::Argument*, bool> > uncacheable_args_map;
  
    std::map<llvm::Instruction*, bool> can_modref_map;
    
    AugmentedReturn(llvm::Function* fn, llvm::StructType* tapeType, std::map<std::pair<llvm::Instruction*, std::string>, unsigned> tapeIndices, std::map<AugmentedStruct, unsigned> returns, std::map<llvm::CallInst*, const std::map<llvm::Argument*, bool>> uncacheable_args_map, std::map<llvm::Instruction*, bool> can_modref_map)
        : fn(fn), tapeType(tapeType), tapeIndices(tapeIndices), returns(returns), uncacheable_args_map(uncacheable_args_map), can_modref_map(can_modref_map) {}
};

const AugmentedReturn& CreateAugmentedPrimal(llvm::Function* todiff, llvm::AAResults &global_AA, const std::set<unsigned>& constant_args, llvm::TargetLibraryInfo &TLI, bool differentialReturn, bool returnUsed, const std::map<llvm::Argument*, DataType> typeInfo, const std::map<llvm::Argument*, bool> _uncacheable_args, bool forceAnonymousTape);

llvm::Function* CreatePrimalAndGradient(llvm::Function* todiff, const std::set<unsigned>& constant_args, llvm::TargetLibraryInfo &TLI, llvm::AAResults &global_AA, bool returnValue, bool differentialReturn, bool dretPtr, bool topLevel, llvm::Type* additionalArg, const std::map<llvm::Argument*, DataType> typeInfo, const std::map<llvm::Argument*, bool> _uncacheable_args, const AugmentedReturn* augmented);

#endif
