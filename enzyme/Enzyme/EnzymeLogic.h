/*
 * EnzymeLogic.h
 *
 * Copyright (C) 2020 William S. Moses (enzyme@wsmoses.com) - All Rights
 * Reserved
 *
 * For commercial use of this code please contact the author(s) above.
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
#include "TypeAnalysis/TypeAnalysis.h"
#include "Utils.h"

extern llvm::cl::opt<bool> enzyme_print;

enum class AugmentedStruct { Tape, Return, DifferentialReturn };

static inline std::string str(AugmentedStruct c) {
  switch (c) {
  case AugmentedStruct::Tape:
    return "tape";
  case AugmentedStruct::Return:
    return "return";
  case AugmentedStruct::DifferentialReturn:
    return "DifferentialReturn";
  default:
    llvm_unreachable("unknown cache type");
  }
}

static inline llvm::raw_ostream &operator<<(llvm::raw_ostream &o,
                                            AugmentedStruct c) {
  return o << str(c);
}

enum class CacheType { Self, Shadow, Tape };

static inline std::string str(CacheType c) {
  switch (c) {
  case CacheType::Self:
    return "self";
  case CacheType::Shadow:
    return "shadow";
  case CacheType::Tape:
    return "tape";
  default:
    llvm_unreachable("unknown cache type");
  }
}

static inline llvm::raw_ostream &operator<<(llvm::raw_ostream &o, CacheType c) {
  return o << str(c);
}

//! return structtype if recursive function
class AugmentedReturn {
public:
  llvm::Function *fn;
  //! return structtype if recursive function
  llvm::Type *tapeType;

  std::map<std::pair<llvm::Instruction *, CacheType>, int> tapeIndices;

  //! Map from original call to sub augmentation data
  std::map<const llvm::CallInst *, const AugmentedReturn *> subaugmentations;

  //! Map from information desired from a augmented return to its index in the
  //! returned struct
  std::map<AugmentedStruct, int> returns;

  std::map<llvm::CallInst *, const std::map<llvm::Argument *, bool>>
      uncacheable_args_map;

  std::map<llvm::Instruction *, bool> can_modref_map;

  AugmentedReturn(
      llvm::Function *fn, llvm::StructType *tapeType,
      std::map<std::pair<llvm::Instruction *, CacheType>, int> tapeIndices,
      std::map<AugmentedStruct, int> returns,
      std::map<llvm::CallInst *, const std::map<llvm::Argument *, bool>>
          uncacheable_args_map,
      std::map<llvm::Instruction *, bool> can_modref_map)
      : fn(fn), tapeType(tapeType), tapeIndices(tapeIndices), returns(returns),
        uncacheable_args_map(uncacheable_args_map),
        can_modref_map(can_modref_map) {}
  AugmentedReturn(const AugmentedReturn &) = default;
  AugmentedReturn& operator=(const AugmentedReturn&) = default;
};

const AugmentedReturn &
CreateAugmentedPrimal(llvm::Function *todiff, DIFFE_TYPE retType,
                      const std::vector<DIFFE_TYPE> &constant_args,
                      llvm::TargetLibraryInfo &TLI, TypeAnalysis &TA,
                      llvm::AAResults &global_AA, bool returnUsed,
                      const FnTypeInfo &typeInfo,
                      const std::map<llvm::Argument *, bool> _uncacheable_args,
                      bool forceAnonymousTape);

llvm::Function *CreatePrimalAndGradient(
    llvm::Function *todiff, DIFFE_TYPE retType,
    const std::vector<DIFFE_TYPE> &constant_args, llvm::TargetLibraryInfo &TLI,
    TypeAnalysis &TA, llvm::AAResults &global_AA, bool returnValue,
    bool dretUsed, bool topLevel, llvm::Type *additionalArg,
    const FnTypeInfo &typeInfo,
    const std::map<llvm::Argument *, bool> _uncacheable_args,
    const AugmentedReturn *augmented);

extern llvm::cl::opt<bool> looseTypeAnalysis;

extern llvm::cl::opt<bool> cache_reads_always;

extern llvm::cl::opt<bool> cache_reads_never;

extern llvm::cl::opt<bool> nonmarkedglobals_inactiveloads;

class GradientUtils;
bool shouldAugmentCall(llvm::CallInst *op, const GradientUtils *gutils,
                                     TypeResults &TR);

bool legalCombinedForwardReverse(
    llvm::CallInst *origop,
    const std::map<llvm::ReturnInst *, llvm::StoreInst *> &replacedReturns,
    std::vector<llvm::Instruction *> &postCreate,
    std::vector<llvm::Instruction *> &userReplace, GradientUtils *gutils,
    TypeResults &TR,
    const llvm::SmallPtrSetImpl<const llvm::Instruction *> &unnecessaryInstructions,
    const bool subretused);

std::pair<llvm::SmallVector<llvm::Type *, 4>, llvm::SmallVector<llvm::Type *, 4>>
getDefaultFunctionTypeForAugmentation(llvm::FunctionType *called, bool returnUsed,
                                      DIFFE_TYPE retType);

std::pair<llvm::SmallVector<llvm::Type *, 4>, llvm::SmallVector<llvm::Type *, 4>>
getDefaultFunctionTypeForGradient(llvm::FunctionType *called, DIFFE_TYPE retType);
#endif
