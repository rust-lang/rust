/*
 * ActiveVariable.h - Active Varaible Detection Utilities
 *
 * Copyright (C) 2020 William S. Moses (enzyme@wsmoses.com) - All Rights
 * Reserved
 *
 * For commercial use of this code please contact the author(s) above.
 */
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

#include "TypeAnalysis.h"

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
