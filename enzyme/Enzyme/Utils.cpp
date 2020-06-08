/*
 * Utils.cpp
 *
 * Copyright (C) 2020 Anonymous Author(s) - All Rights Reserved
 *
 * For commercial use of this code please contact the author(s) above.
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
Function* getOrInsertDifferentialFloatMemcpy(Module& M, PointerType* T, unsigned dstalign, unsigned srcalign) {
    Type* elementType = T->getElementType();
    assert(elementType->isFloatingPointTy());
    std::string name = "__enzyme_memcpyadd_" + tofltstr(elementType) + "da" + std::to_string(dstalign) + "sa" + std::to_string(srcalign);
    FunctionType* FT = FunctionType::get(Type::getVoidTy(M.getContext()), { T, T, Type::getInt64Ty(M.getContext()) }, false);

    #if LLVM_VERSION_MAJOR >= 9
    Function* F = cast<Function>(M.getOrInsertFunction(name, FT).getCallee());
    #else
    Function* F = cast<Function>(M.getOrInsertFunction(name, FT));
    #endif

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
    LoadInst* dstl = B.CreateLoad(dsti, "dst.i.l");
    dstl->setAlignment(dstalign);
    StoreInst* dsts = B.CreateStore(Constant::getNullValue(elementType), dsti);
    dsts->setAlignment(dstalign);

    Value* srci = B.CreateGEP(src, { idx }, "src.i");
    LoadInst* srcl = B.CreateLoad(srci, "src.i.l");
    srcl->setAlignment(srcalign);
    StoreInst* srcs = B.CreateStore(B.CreateFAdd(srcl, dstl), srci);
    srcs->setAlignment(srcalign);

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

//TODO implement differential memmove
Function* getOrInsertDifferentialFloatMemmove(Module& M, PointerType* T, unsigned dstalign, unsigned srcalign) {
    llvm::errs() << "warning: didn't implement memmove, using memcpy as fallback which can result in errors\n";
    return getOrInsertDifferentialFloatMemcpy(M, T, dstalign, srcalign);
}
