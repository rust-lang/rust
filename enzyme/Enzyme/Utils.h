/*
 * Utils.h
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

#ifndef ENZYME_UTILS_H
#define ENZYME_UTILS_H

#include "llvm/IR/Function.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"

static inline llvm::FastMathFlags getFast() {
    llvm::FastMathFlags f;
    f.set();
    return f;
}

//! Create function for type that performs the derivative memcpy on floating point memory
llvm::Function* getOrInsertDifferentialFloatMemcpy(llvm::Module& M, llvm::PointerType* T);

#endif
