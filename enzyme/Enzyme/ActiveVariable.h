/*
 * ActiveVariable.h - Active Varaible Detection Utilities
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
#ifndef ENZYME_ACTIVE_VAR_H
#define ENZYME_ACTIVE_VAR_H 1

#include <cstdint>
#include <deque>

#include <llvm/Config/llvm-config.h>

#include "llvm/IR/Constants.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Analysis/AliasAnalysis.h"

#include "llvm/Support/CommandLine.h"

#include "llvm/IR/InstVisitor.h"

#include "TypeAnalysis.h"

extern llvm::cl::opt<bool> printconst;
extern llvm::cl::opt<bool> nonmarkedglobals_inactive;

//DataType isIntASecretFloat(const std::map<llvm::Argument*, DataType> typeInfo, llvm::Value* val, IntType defaultType=IntType::Unknown, bool errIfNotFound=false);

//! return the secret float type of val if found, otherwise nullptr
//!   if onlyFirst is set, consider only the first element of the pointer val (e.g. if we have {int, double, double}, consider only the int)
//    onlyFirst should typically be used for store operands, whereas not used for memcpy
//DataType isIntPointerASecretFloat(const std::map<llvm::Argument*, DataType> typeInfo, llvm::Value* val, bool onlyFirst, bool errIfNotFound=true);

bool isconstantValueM(TypeResults &TA, llvm::Value* val, llvm::SmallPtrSetImpl<llvm::Value*> &constants, llvm::SmallPtrSetImpl<llvm::Value*> &nonconstant, llvm::SmallPtrSetImpl<llvm::Value*> &constantvals, llvm::SmallPtrSetImpl<llvm::Value*> &retvals, llvm::AAResults &AA, uint8_t directions=3);

// TODO separate if the instruction is constant (i.e. could change things)
//    from if the value is constant (the value is something that could be differentiated)
bool isconstantM(TypeResults &TA, llvm::Instruction* inst, llvm::SmallPtrSetImpl<llvm::Value*> &constants, llvm::SmallPtrSetImpl<llvm::Value*> &nonconstant, llvm::SmallPtrSetImpl<llvm::Value*> &constantvals, llvm::SmallPtrSetImpl<llvm::Value*> &retvals, llvm::AAResults &AA, uint8_t directions=3);

#endif
