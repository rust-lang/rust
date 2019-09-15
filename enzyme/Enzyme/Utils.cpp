/*
 * Utils.cpp
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

#include "Utils.h"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"

using namespace llvm;

static inline std::string tofltstr(Type* T) {
    switch (T->getTypeID()) {
        case Type::HalfTyID: return "half";
        case Type::FloatTyID: return "float";
        case Type::DoubleTyID: return "double";
        case Type::X86_FP80TyID: return "x87d";
        case Type::FP128TyID: return "quad";
        case Type::PPC_FP128TyID: return "ppcddouble";
        default: llvm_unreachable("Invalid floating type");
    }
}

//! Create function for type that is equivalent to memcpy but adds to destination rather 
//! than a direct copy; dst, src, numelems
Function* getOrInsertDifferentialFloatMemcpy(Module& M, PointerType* T) {
    Type* elementType = T->getElementType();
    assert(elementType->isFloatingPointTy());
    std::string name = "__enzyme_memcpyadd_" + tofltstr(elementType);
    FunctionType* FT = FunctionType::get(Type::getVoidTy(M.getContext()), { T, T, Type::getInt64Ty(M.getContext()) }, false);

    Function* F = cast<Function>(M.getOrInsertFunction(name, FT));

    if (!F->empty()) return F;
    
    F->setLinkage(Function::LinkageTypes::InternalLinkage);
    F->addFnAttr(Attribute::ArgMemOnly);
    F->addFnAttr(Attribute::NoUnwind);
    F->addParamAttr(0, Attribute::NoCapture);
    F->addParamAttr(1, Attribute::NoCapture);

    BasicBlock* entry = BasicBlock::Create(M.getContext(), "entry", F);
    BasicBlock* body = BasicBlock::Create(M.getContext(), "for.body", F);
    BasicBlock* end = BasicBlock::Create(M.getContext(), "for.end", F);

    auto dst = F->arg_begin();
    dst->setName("dst");
    auto src = dst+1;
    src->setName("src");
    auto num = src+1;
    num->setName("num");

    {
    IRBuilder<> B(entry);
    B.CreateCondBr(B.CreateICmpEQ(num, ConstantInt::get(num->getType(), 0)), end, body);
    }

    {
    IRBuilder<> B(body);
    B.setFastMathFlags(getFast());
    PHINode* idx = B.CreatePHI(num->getType(), 2, "idx");
    idx->addIncoming(ConstantInt::get(num->getType(), 0), entry);

    Value* dsti = B.CreateGEP(dst, { idx }, "dst.i");
    Value* dstl = B.CreateLoad(dsti, "dst.i.l");
    B.CreateStore(Constant::getNullValue(elementType), dsti);
    
    Value* srci = B.CreateGEP(src, { idx }, "src.i");
    Value* srcl = B.CreateLoad(srci, "src.i.l");
    B.CreateStore(B.CreateFAdd(srcl, dstl), srci);

    Value* next = B.CreateNUWAdd(idx, ConstantInt::get(num->getType(), 1), "idx.next");
    idx->addIncoming(next,  body);
    B.CreateCondBr(B.CreateICmpEQ(num, next), end, body);
    }
    
    {
    IRBuilder<> B(end);
    B.CreateRetVoid();
    }
    return F;
}
