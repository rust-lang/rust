//===- ActivityAnalysis.h - Declaration of Activity Analysis  -----------===//
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
#include <memory>

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
class ActivityAnalyzer : std::enable_shared_from_this<ActivityAnalyzer> {
public:
    llvm::AAResults &AA;
    llvm::TargetLibraryInfo &TLI;
    bool ActiveReturns;
    uint8_t directions;

    llvm::SmallPtrSet<llvm::Instruction *, 4> constants;
    llvm::SmallPtrSet<llvm::Instruction *, 20> nonconstant;
    llvm::SmallPtrSet<llvm::Value *, 4> constantvals;
    llvm::SmallPtrSet<llvm::Value *, 2> retvals;


    ActivityAnalyzer(llvm::AAResults &AA_, llvm::TargetLibraryInfo &TLI_, bool ActiveReturns, uint8_t directions) :
        AA(AA_), TLI(TLI_), ActiveReturns(ActiveReturns), directions(directions) {
        assert(directions <= 3);
        assert(directions != 0);
    }

    bool isconstantValueM(TypeResults &TR, llvm::Value *val);

    bool isconstantM(TypeResults &TR, llvm::Instruction *inst);

private:
    ActivityAnalyzer(ActivityAnalyzer & Other, uint8_t directions)  : AA(Other.AA), TLI(Other.TLI), 
        ActiveReturns(Other.ActiveReturns), directions(directions),
        constants(Other.constants.begin(), Other.constants.end()), nonconstant(Other.nonconstant.begin(), Other.nonconstant.end()), 
        constantvals(Other.constantvals.begin(), Other.constantvals.end()), retvals(Other.retvals.begin(), Other.retvals.end())
        {
            assert(directions != 0);
            //assert(directions != 3);

            assert((directions & Other.directions) == directions);
            assert((directions & Other.directions) != 0);

        }
    void insertConstantsFrom(ActivityAnalyzer &Hypothesis) {
        constants.insert(Hypothesis.constants.begin(), Hypothesis.constants.end());
        constantvals.insert(Hypothesis.constantvals.begin(), Hypothesis.constantvals.end());
    }
    void insertAllFrom(ActivityAnalyzer &Hypothesis) {
        constants.insert(Hypothesis.constants.begin(), Hypothesis.constants.end());
        constantvals.insert(Hypothesis.constantvals.begin(), Hypothesis.constantvals.end());
        nonconstant.insert(Hypothesis.nonconstant.begin(), Hypothesis.nonconstant.end());
        retvals.insert(Hypothesis.retvals.begin(), Hypothesis.retvals.end());
    }
    bool isFunctionArgumentConstant(llvm::CallInst *CI, llvm::Value *val);

    bool isInstructionInactiveFromOrigin(TypeResults &TR, llvm::Value* val);
    bool isValueInactiveFromUsers(TypeResults &TR, llvm::Value* val);

    std::map<llvm::Value*, bool> StoredOrReturnedCache;
    bool isValueActivelyStoredOrReturned(TypeResults &TR, llvm::Value* val);
};

#endif
