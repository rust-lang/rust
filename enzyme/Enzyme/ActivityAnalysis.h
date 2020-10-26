//===- ActivityAnalysis.h - Declaration of Activity Analysis  -----------===//
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
// This file contains the declaration of Activity Analysis -- an AD-specific
// analysis that deduces if a given instruction or value can impact the
// calculation of a derivative. This file consists of two mutually recurive
// functions that compute this for values and instructions, respectively.
//
//===----------------------------------------------------------------------===//
#ifndef ENZYME_ACTIVE_VAR_H
#define ENZYME_ACTIVE_VAR_H 1

#include <cstdint>
#include <deque>

#include <llvm/Config/llvm-config.h>

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"

#include "llvm/Support/CommandLine.h"

#include "llvm/IR/InstVisitor.h"

#include "TypeAnalysis/TypeAnalysis.h"

extern llvm::cl::opt<bool> printconst;
extern llvm::cl::opt<bool> nonmarkedglobals_inactive;

/// Helper class to analyze the differential activity
class ActivityAnalyzer {
  /// Aliasing Information
  llvm::AAResults &AA;
  /// Library Information
  llvm::TargetLibraryInfo &TLI;
  /// Whether the returns of the function being analyzed are active
  const bool ActiveReturns;

  /// Direction of current analysis
  const uint8_t directions;
  /// Analyze up based off of operands
  static constexpr uint8_t UP = 1;
  /// Analyze down based off uses
  static constexpr uint8_t DOWN = 2;

  /// Instructions that don't propagate adjoints
  /// These instructions could return an active pointer, but
  /// do not propagate adjoints themselves
  llvm::SmallPtrSet<llvm::Instruction *, 4> ConstantInstructions;

  /// Instructions that could propagate adjoints
  llvm::SmallPtrSet<llvm::Instruction *, 20> ActiveInstructions;

  /// Values that do not contain derivative information, either
  /// directly or as a pointer to
  llvm::SmallPtrSet<llvm::Value *, 4> ConstantValues;

  /// Values that may contain derivative information
  llvm::SmallPtrSet<llvm::Value *, 2> ActiveValues;

public:
  /// Construct the analyzer from the a previous set of constant and active
  /// values and whether returns are active. The all arguments of the functions
  /// being analyzed must be in the set of constant and active values, lest an
  /// error occur during analysis
  ActivityAnalyzer(llvm::AAResults &AA_, llvm::TargetLibraryInfo &TLI_,
                   const llvm::SmallPtrSetImpl<llvm::Value *> &ConstantValues,
                   const llvm::SmallPtrSetImpl<llvm::Value *> &ActiveValues,
                   bool ActiveReturns)
      : AA(AA_), TLI(TLI_), ActiveReturns(ActiveReturns), directions(UP | DOWN),
        ConstantValues(ConstantValues.begin(), ConstantValues.end()),
        ActiveValues(ActiveValues.begin(), ActiveValues.end()) {}

  /// Return whether this instruction is known not to propagate adjoints
  /// Note that instructions could return an active pointer, but
  /// do not propagate adjoints themselves
  bool isConstantInstruction(TypeResults &TR, llvm::Instruction *inst);

  /// Return whether this values is known not to contain derivative
  // information, either directly or as a pointer to
  bool isConstantValue(TypeResults &TR, llvm::Value *val);

private:
  /// Create a new analyzer starting from an existing Analyzer
  /// This is used to perform inductive assumptions
  ActivityAnalyzer(ActivityAnalyzer &Other, uint8_t directions)
      : AA(Other.AA), TLI(Other.TLI), ActiveReturns(Other.ActiveReturns),
        directions(directions),
        ConstantInstructions(Other.ConstantInstructions),
        ActiveInstructions(Other.ActiveInstructions),
        ConstantValues(Other.ConstantValues), ActiveValues(Other.ActiveValues) {
    assert(directions != 0);
    assert((directions & Other.directions) == directions);
    assert((directions & Other.directions) != 0);
  }

  /// Import known constants from an existing analyzer
  void insertConstantsFrom(ActivityAnalyzer &Hypothesis) {
    ConstantInstructions.insert(Hypothesis.ConstantInstructions.begin(),
                                Hypothesis.ConstantInstructions.end());
    ConstantValues.insert(Hypothesis.ConstantValues.begin(),
                          Hypothesis.ConstantValues.end());
  }

  /// Import known data from an existing analyzer
  void insertAllFrom(ActivityAnalyzer &Hypothesis) {
    ConstantInstructions.insert(Hypothesis.ConstantInstructions.begin(),
                                Hypothesis.ConstantInstructions.end());
    ConstantValues.insert(Hypothesis.ConstantValues.begin(),
                          Hypothesis.ConstantValues.end());
    ActiveInstructions.insert(Hypothesis.ActiveInstructions.begin(),
                              Hypothesis.ActiveInstructions.end());
    ActiveValues.insert(Hypothesis.ActiveValues.begin(),
                        Hypothesis.ActiveValues.end());
  }

  /// Is the use of value val as an argument of call CI known to be inactive
  bool isFunctionArgumentConstant(llvm::CallInst *CI, llvm::Value *val);

  /// Is the instruction guaranteed to be inactive because of its operands
  bool isInstructionInactiveFromOrigin(TypeResults &TR, llvm::Value *val);

public:
  /// Is the value free of any active uses
  bool isValueInactiveFromUsers(TypeResults &TR, llvm::Value *val);

private:
  /// Is the value potentially actively returned or stored
  bool isValueActivelyStoredOrReturned(TypeResults &TR, llvm::Value *val);
  /// StoredOrReturnedCache acts as an inductive cache of results for
  /// isValueActivelyStoredOrReturned
  std::map<llvm::Value *, bool> StoredOrReturnedCache;
};

#endif
