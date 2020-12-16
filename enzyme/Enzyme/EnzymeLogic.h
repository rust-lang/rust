//===- EnzymeLogic.h - Implementation of forward and reverse pass generation==//
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
// This file declares two functions CreatePrimalAndGradient and
// CreateAugmentedPrimal. CreatePrimalAndGradient takes a function, known
// TypeResults of the calling context, known activity analysis of the
// arguments and a bool `topLevel`. It creates a corresponding gradient
// function, computing the forward pass as well if at `topLevel`.
// CreateAugmentedPrimal takes similar arguments and creates an augmented
// forward pass.
//
//===----------------------------------------------------------------------===//
#ifndef ENZYME_LOGIC_H
#define ENZYME_LOGIC_H

#include <set>
#include <utility>

#include "SCEV/ScalarEvolutionExpander.h"
#include "SCEV/TargetLibraryInfo.h"

#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/CommandLine.h"

#include "llvm/Analysis/AliasAnalysis.h"

#include "ActivityAnalysis.h"
#include "TypeAnalysis/TypeAnalysis.h"
#include "Utils.h"

extern llvm::cl::opt<bool> EnzymePrint;

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

  std::set<size_t> tapeIndiciesToFree;

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
};

/// Create an augmented forward pass.
///  \p todiff is the function to differentiate
///  \p retType is the activity info of the return
///  \p constant_args is the activity info of the arguments
///  \p returnUsed is whether the primal's return should also be returned
///  \p typeInfo is the type info information about the calling context
///  \p _uncacheable_args marks whether an argument may be rewritten before
///  loads in the generated function (and thus cannot be cached). \p
///  forceAnonymousTape forces the tape to be an i8* rather than the true tape
///  structure \p AtomicAdd is whether to perform all adjoint updates to memory
///  in an atomic way \p PostOpt is whether to perform basic optimization of the
///  function after synthesis
const AugmentedReturn &CreateAugmentedPrimal(
    llvm::Function *todiff, DIFFE_TYPE retType,
    const std::vector<DIFFE_TYPE> &constant_args, llvm::TargetLibraryInfo &TLI,
    TypeAnalysis &TA, llvm::AAResults &global_AA, bool returnUsed,
    const FnTypeInfo &typeInfo,
    const std::map<llvm::Argument *, bool> _uncacheable_args,
    bool forceAnonymousTape, bool AtomicAdd, bool PostOpt, bool omp = false);

/// Create the derivative function itself.
///  \p todiff is the function to differentiate
///  \p retType is the activity info of the return
///  \p constant_args is the activity info of the arguments
///  \p returnValue is whether the primal's return should also be returned
///  \p dretUsed is whether the shadow return value should also be returned
///  \p additionalArg is the type (or null) of an additional type in the
///  signature to hold the tape. \p typeInfo is the type info information about
///  the calling context \p _uncacheable_args marks whether an argument may be
///  rewritten before loads in the generated function (and thus cannot be
///  cached). \p augmented is the data structure created by prior call to an
///  augmented forward pass \p AtomicAdd is whether to perform all adjoint
///  updates to memory in an atomic way \p PostOpt is whether to perform basic
///  optimization of the function after synthesis
llvm::Function *CreatePrimalAndGradient(
    llvm::Function *todiff, DIFFE_TYPE retType,
    const std::vector<DIFFE_TYPE> &constant_args, llvm::TargetLibraryInfo &TLI,
    TypeAnalysis &TA, llvm::AAResults &global_AA, bool returnValue,
    bool dretUsed, bool topLevel, llvm::Type *additionalArg,
    const FnTypeInfo &typeInfo,
    const std::map<llvm::Argument *, bool> _uncacheable_args,
    const AugmentedReturn *augmented, bool AtomicAdd, bool PostOpt = false,
    bool omp = false);

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
    const llvm::SmallPtrSetImpl<const llvm::Instruction *>
        &unnecessaryInstructions,
    const bool subretused);

std::pair<llvm::SmallVector<llvm::Type *, 4>,
          llvm::SmallVector<llvm::Type *, 4>>
getDefaultFunctionTypeForAugmentation(llvm::FunctionType *called,
                                      bool returnUsed, DIFFE_TYPE retType);

std::pair<llvm::SmallVector<llvm::Type *, 4>,
          llvm::SmallVector<llvm::Type *, 4>>
getDefaultFunctionTypeForGradient(llvm::FunctionType *called,
                                  DIFFE_TYPE retType);
#endif
