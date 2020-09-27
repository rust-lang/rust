//===- ActiveVariable.h - Declaration of Activity Analysis  -----------===//
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

bool isconstantValueM(TypeResults &TA, llvm::Value *val,
                      llvm::SmallPtrSetImpl<llvm::Value *> &constants,
                      llvm::SmallPtrSetImpl<llvm::Value *> &nonconstant,
                      llvm::SmallPtrSetImpl<llvm::Value *> &constantvals,
                      llvm::SmallPtrSetImpl<llvm::Value *> &retvals,
                      llvm::AAResults &AA, uint8_t directions = 3);

bool isconstantM(TypeResults &TA, llvm::Instruction *inst,
                 llvm::SmallPtrSetImpl<llvm::Value *> &constants,
                 llvm::SmallPtrSetImpl<llvm::Value *> &nonconstant,
                 llvm::SmallPtrSetImpl<llvm::Value *> &constantvals,
                 llvm::SmallPtrSetImpl<llvm::Value *> &retvals,
                 llvm::AAResults &AA, uint8_t directions = 3);

#endif
