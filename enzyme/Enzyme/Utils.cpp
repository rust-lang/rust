//===- Utils.cpp - Definition of miscellaneous utilities ------------------===//
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
// This file defines miscellaneous utilities that are used as part of the
// AD process.
//
//===----------------------------------------------------------------------===//
#include "Utils.h"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"

using namespace llvm;

EnzymeFailure::EnzymeFailure(llvm::StringRef RemarkName,
                             const llvm::DiagnosticLocation &Loc,
                             const llvm::Instruction *CodeRegion)
    : DiagnosticInfoIROptimization(
          EnzymeFailure::ID(), DS_Error, "enzyme", RemarkName,
          *CodeRegion->getParent()->getParent(), Loc, CodeRegion) {}

llvm::DiagnosticKind EnzymeFailure::ID() {
  static auto id = llvm::getNextAvailablePluginDiagnosticKind();
  return (llvm::DiagnosticKind)id;
}

/// \see DiagnosticInfoOptimizationBase::isEnabled.
bool EnzymeFailure::isEnabled() const { return true; }

/// Convert a floating type to a string
static inline std::string tofltstr(Type *T) {
  switch (T->getTypeID()) {
  case Type::HalfTyID:
    return "half";
  case Type::FloatTyID:
    return "float";
  case Type::DoubleTyID:
    return "double";
  case Type::X86_FP80TyID:
    return "x87d";
  case Type::FP128TyID:
    return "quad";
  case Type::PPC_FP128TyID:
    return "ppcddouble";
  default:
    llvm_unreachable("Invalid floating type");
  }
}

/// Create function for type that is equivalent to memcpy but adds to
/// destination rather than a direct copy; dst, src, numelems
Function *getOrInsertDifferentialFloatMemcpy(Module &M, Type *elementType,
                                             unsigned dstalign,
                                             unsigned srcalign,
                                             unsigned dstaddr,
                                             unsigned srcaddr) {
  assert(elementType->isFloatingPointTy());
  std::string name = "__enzyme_memcpyadd_" + tofltstr(elementType) + "da" +
                     std::to_string(dstalign) + "sa" + std::to_string(srcalign);
  if (dstaddr)
    name += "dadd" + std::to_string(dstaddr);
  if (srcaddr)
    name += "sadd" + std::to_string(srcaddr);
  FunctionType *FT = FunctionType::get(Type::getVoidTy(M.getContext()),
                                       {PointerType::get(elementType, dstaddr),
                                        PointerType::get(elementType, srcaddr),
                                        Type::getInt64Ty(M.getContext())},
                                       false);

#if LLVM_VERSION_MAJOR >= 9
  Function *F = cast<Function>(M.getOrInsertFunction(name, FT).getCallee());
#else
  Function *F = cast<Function>(M.getOrInsertFunction(name, FT));
#endif

  if (!F->empty())
    return F;

  F->setLinkage(Function::LinkageTypes::InternalLinkage);
  F->addFnAttr(Attribute::ArgMemOnly);
  F->addFnAttr(Attribute::NoUnwind);
  F->addParamAttr(0, Attribute::NoCapture);
  F->addParamAttr(1, Attribute::NoCapture);

  BasicBlock *entry = BasicBlock::Create(M.getContext(), "entry", F);
  BasicBlock *body = BasicBlock::Create(M.getContext(), "for.body", F);
  BasicBlock *end = BasicBlock::Create(M.getContext(), "for.end", F);

  auto dst = F->arg_begin();
  dst->setName("dst");
  auto src = dst + 1;
  src->setName("src");
  auto num = src + 1;
  num->setName("num");

  {
    IRBuilder<> B(entry);
    B.CreateCondBr(B.CreateICmpEQ(num, ConstantInt::get(num->getType(), 0)),
                   end, body);
  }

  {
    IRBuilder<> B(body);
    B.setFastMathFlags(getFast());
    PHINode *idx = B.CreatePHI(num->getType(), 2, "idx");
    idx->addIncoming(ConstantInt::get(num->getType(), 0), entry);

#if LLVM_VERSION_MAJOR > 7
    Value *dsti = B.CreateGEP(
        cast<PointerType>(dst->getType())->getElementType(), dst, idx, "dst.i");
    LoadInst *dstl = B.CreateLoad(
        cast<PointerType>(dsti->getType())->getElementType(), dsti, "dst.i.l");
#else
    Value *dsti = B.CreateGEP(dst, idx, "dst.i");
    LoadInst *dstl = B.CreateLoad(dsti, "dst.i.l");
#endif
    StoreInst *dsts = B.CreateStore(Constant::getNullValue(elementType), dsti);
    if (dstalign) {
#if LLVM_VERSION_MAJOR >= 10
      dstl->setAlignment(Align(dstalign));
      dsts->setAlignment(Align(dstalign));
#else
      dstl->setAlignment(dstalign);
      dsts->setAlignment(dstalign);
#endif
    }

#if LLVM_VERSION_MAJOR > 7
    Value *srci = B.CreateGEP(
        cast<PointerType>(src->getType())->getElementType(), src, idx, "src.i");
    LoadInst *srcl = B.CreateLoad(
        cast<PointerType>(srci->getType())->getElementType(), srci, "src.i.l");
#else
    Value *srci = B.CreateGEP(src, idx, "src.i");
    LoadInst *srcl = B.CreateLoad(srci, "src.i.l");
#endif
    StoreInst *srcs = B.CreateStore(B.CreateFAdd(srcl, dstl), srci);
    if (srcalign) {
#if LLVM_VERSION_MAJOR >= 10
      srcl->setAlignment(Align(srcalign));
      srcs->setAlignment(Align(srcalign));
#else
      srcl->setAlignment(srcalign);
      srcs->setAlignment(srcalign);
#endif
    }

    Value *next =
        B.CreateNUWAdd(idx, ConstantInt::get(num->getType(), 1), "idx.next");
    idx->addIncoming(next, body);
    B.CreateCondBr(B.CreateICmpEQ(num, next), end, body);
  }

  {
    IRBuilder<> B(end);
    B.CreateRetVoid();
  }
  return F;
}

Function *getOrInsertMemcpyStrided(Module &M, PointerType *T, unsigned dstalign,
                                   unsigned srcalign) {
  Type *elementType = T->getElementType();
  assert(elementType->isFloatingPointTy());
  std::string name = "__enzyme_memcpy_" + tofltstr(elementType) + "da" +
                     std::to_string(dstalign) + "sa" +
                     std::to_string(srcalign) + "stride";
  FunctionType *FT = FunctionType::get(Type::getVoidTy(M.getContext()),
                                       {T, T, Type::getInt32Ty(M.getContext()),
                                        Type::getInt32Ty(M.getContext())},
                                       false);

#if LLVM_VERSION_MAJOR >= 9
  Function *F = cast<Function>(M.getOrInsertFunction(name, FT).getCallee());
#else
  Function *F = cast<Function>(M.getOrInsertFunction(name, FT));
#endif

  if (!F->empty())
    return F;

  F->setLinkage(Function::LinkageTypes::InternalLinkage);
  F->addFnAttr(Attribute::ArgMemOnly);
  F->addFnAttr(Attribute::NoUnwind);
  F->addParamAttr(0, Attribute::NoCapture);
  F->addParamAttr(1, Attribute::NoCapture);
  F->addParamAttr(0, Attribute::WriteOnly);
  F->addParamAttr(1, Attribute::ReadOnly);

  BasicBlock *entry = BasicBlock::Create(M.getContext(), "entry", F);
  BasicBlock *body = BasicBlock::Create(M.getContext(), "for.body", F);
  BasicBlock *end = BasicBlock::Create(M.getContext(), "for.end", F);

  auto dst = F->arg_begin();
  dst->setName("dst");
  auto src = dst + 1;
  src->setName("src");
  auto num = src + 1;
  num->setName("num");
  auto stride = num + 1;
  stride->setName("stride");

  {
    IRBuilder<> B(entry);
    B.CreateCondBr(B.CreateICmpEQ(num, ConstantInt::get(num->getType(), 0)),
                   end, body);
  }

  {
    IRBuilder<> B(body);
    B.setFastMathFlags(getFast());
    PHINode *idx = B.CreatePHI(num->getType(), 2, "idx");
    PHINode *sidx = B.CreatePHI(num->getType(), 2, "sidx");
    idx->addIncoming(ConstantInt::get(num->getType(), 0), entry);
    sidx->addIncoming(ConstantInt::get(num->getType(), 0), entry);

#if LLVM_VERSION_MAJOR > 7
    Value *dsti = B.CreateGEP(
        cast<PointerType>(dst->getType())->getElementType(), dst, idx, "dst.i");
    Value *srci =
        B.CreateGEP(cast<PointerType>(src->getType())->getElementType(), src,
                    sidx, "src.i");
    LoadInst *srcl = B.CreateLoad(
        cast<PointerType>(srci->getType())->getElementType(), srci, "src.i.l");
#else
    Value *dsti = B.CreateGEP(dst, idx, "dst.i");
    Value *srci = B.CreateGEP(src, sidx, "src.i");
    LoadInst *srcl = B.CreateLoad(srci, "src.i.l");
#endif

    StoreInst *dsts = B.CreateStore(srcl, dsti);

    if (dstalign) {
#if LLVM_VERSION_MAJOR >= 10
      dsts->setAlignment(Align(dstalign));
#else
      dsts->setAlignment(dstalign);
#endif
    }
    if (srcalign) {
#if LLVM_VERSION_MAJOR >= 10
      srcl->setAlignment(Align(srcalign));
#else
      srcl->setAlignment(srcalign);
#endif
    }

    Value *next =
        B.CreateNUWAdd(idx, ConstantInt::get(num->getType(), 1), "idx.next");
    Value *snext = B.CreateNUWAdd(sidx, stride, "sidx.next");
    idx->addIncoming(next, body);
    sidx->addIncoming(snext, body);
    B.CreateCondBr(B.CreateICmpEQ(num, next), end, body);
  }

  {
    IRBuilder<> B(end);
    B.CreateRetVoid();
  }

  return F;
}

// TODO implement differential memmove
Function *getOrInsertDifferentialFloatMemmove(Module &M, Type *T,
                                              unsigned dstalign,
                                              unsigned srcalign,
                                              unsigned dstaddr,
                                              unsigned srcaddr) {
  llvm::errs() << "warning: didn't implement memmove, using memcpy as fallback "
                  "which can result in errors\n";
  return getOrInsertDifferentialFloatMemcpy(M, T, dstalign, srcalign, dstaddr,
                                            srcaddr);
}

/// Create function to computer nearest power of two
llvm::Value *nextPowerOfTwo(llvm::IRBuilder<> &B, llvm::Value *V) {
  assert(V->getType()->isIntegerTy());
  IntegerType *T = cast<IntegerType>(V->getType());
  V = B.CreateAdd(V, ConstantInt::get(T, -1));
  for (size_t i = 1; i < T->getBitWidth(); i *= 2) {
    V = B.CreateOr(V, B.CreateLShr(V, ConstantInt::get(T, i)));
  }
  V = B.CreateAdd(V, ConstantInt::get(T, 1));
  return V;
}

llvm::Function *getOrInsertDifferentialMPI_Wait(llvm::Module &M,
                                                ArrayRef<llvm::Type *> T,
                                                Type *reqType) {
  std::vector<llvm::Type *> types(T.begin(), T.end());
  types.push_back(reqType);
  std::string name = "__enzyme_differential_mpi_wait";
  FunctionType *FT =
      FunctionType::get(Type::getVoidTy(M.getContext()), types, false);

#if LLVM_VERSION_MAJOR >= 9
  Function *F = cast<Function>(M.getOrInsertFunction(name, FT).getCallee());
#else
  Function *F = cast<Function>(M.getOrInsertFunction(name, FT));
#endif

  if (!F->empty())
    return F;

  BasicBlock *entry = BasicBlock::Create(M.getContext(), "entry", F);
  BasicBlock *isend = BasicBlock::Create(M.getContext(), "invertISend", F);
  BasicBlock *irecv = BasicBlock::Create(M.getContext(), "invertIRecv", F);

#if 0
    /*0 */Type::getInt8PtrTy(call.getContext())
    /*1 */i64
    /*2 */Type::getInt8PtrTy(call.getContext())
    /*3 */i64
    /*4 */i64
    /*5 */Type::getInt8PtrTy(call.getContext())
    /*6 */Type::getInt8Ty(call.getContext())
#endif

  auto buff = F->arg_begin();
  buff->setName("buf");
  Value *buf = buff;
  Value *count = buff + 1;
  count->setName("count");
  Value *datatype = buff + 2;
  datatype->setName("datatype");
  Value *source = buff + 3;
  source->setName("source");
  Value *tag = buff + 4;
  tag->setName("tag");
  Value *comm = buff + 5;
  comm->setName("comm");
  Value *fn = buff + 6;
  fn->setName("fn");
  Value *d_req = buff + 7;
  d_req->setName("d_req");

  auto isendfn = M.getFunction("PMPI_Isend");
  if (!isendfn)
    isendfn = M.getFunction("MPI_Isend");
  assert(isendfn);
  auto irecvfn = M.getFunction("PMPI_Irecv");
  if (!irecvfn)
    irecvfn = M.getFunction("MPI_Irecv");
  assert(irecvfn);

  IRBuilder<> B(entry);
  auto arg = isendfn->arg_begin();
  if (arg->getType()->isIntegerTy())
    buf = B.CreatePtrToInt(buf, arg->getType());
  arg++;
  count = B.CreateZExtOrTrunc(count, arg->getType());
  arg++;
  datatype = B.CreatePointerCast(datatype, arg->getType());
  arg++;
  source = B.CreateZExtOrTrunc(source, arg->getType());
  arg++;
  tag = B.CreateZExtOrTrunc(tag, arg->getType());
  arg++;
  comm = B.CreatePointerCast(comm, arg->getType());
  arg++;
  if (arg->getType()->isIntegerTy())
    d_req = B.CreatePtrToInt(d_req, arg->getType());
  Value *args[] = {
      buf, count, datatype, source, tag, comm, d_req,
  };

  B.CreateCondBr(B.CreateICmpEQ(fn, ConstantInt::get(fn->getType(),
                                                     (int)MPI_CallType::ISEND)),
                 isend, irecv);

  {
    B.SetInsertPoint(isend);
    auto fcall = B.CreateCall(irecvfn, args);
    fcall->setCallingConv(isendfn->getCallingConv());
    B.CreateRetVoid();
  }

  {
    B.SetInsertPoint(irecv);
    auto fcall = B.CreateCall(isendfn, args);
    fcall->setCallingConv(isendfn->getCallingConv());
    B.CreateRetVoid();
  }
  return F;
}

llvm::Value *getOrInsertOpFloatSum(llvm::Module &M, llvm::Type *OpPtr,
                                   ConcreteType CT, llvm::Type *intType,
                                   IRBuilder<> &B2) {
  std::string name = "__enzyme_mpi_sum" + CT.str();
  assert(CT.isFloat());
  auto FlT = CT.isFloat();

  if (auto Glob = M.getGlobalVariable(name)) {
#if LLVM_VERSION_MAJOR > 7
    return B2.CreateLoad(Glob->getValueType(), Glob);
#else
    return B2.CreateLoad(Glob);
#endif
  }

  std::vector<llvm::Type *> types = {PointerType::getUnqual(FlT),
                                     PointerType::getUnqual(FlT),
                                     PointerType::getUnqual(intType), OpPtr};
  FunctionType *FuT =
      FunctionType::get(Type::getVoidTy(M.getContext()), types, false);

#if LLVM_VERSION_MAJOR >= 9
  Function *F =
      cast<Function>(M.getOrInsertFunction(name + "_run", FuT).getCallee());
#else
  Function *F = cast<Function>(M.getOrInsertFunction(name + "_run", FuT));
#endif

  F->setLinkage(Function::LinkageTypes::InternalLinkage);
  F->addFnAttr(Attribute::ArgMemOnly);
  F->addFnAttr(Attribute::NoUnwind);
  F->addParamAttr(0, Attribute::NoCapture);
  F->addParamAttr(0, Attribute::ReadOnly);
  F->addParamAttr(1, Attribute::NoCapture);
  F->addParamAttr(2, Attribute::NoCapture);
  F->addParamAttr(2, Attribute::ReadOnly);
  F->addParamAttr(3, Attribute::NoCapture);
  F->addParamAttr(3, Attribute::ReadNone);

  BasicBlock *entry = BasicBlock::Create(M.getContext(), "entry", F);
  BasicBlock *body = BasicBlock::Create(M.getContext(), "for.body", F);
  BasicBlock *end = BasicBlock::Create(M.getContext(), "for.end", F);

  auto src = F->arg_begin();
  src->setName("src");
  auto dst = src + 1;
  dst->setName("dst");
  auto lenp = dst + 1;
  lenp->setName("lenp");
  Value *len;
  // TODO consider using datatype arg and asserting same size as assumed
  // by type analysis

  {
    IRBuilder<> B(entry);
#if LLVM_VERSION_MAJOR > 7
    len = B.CreateLoad(cast<PointerType>(lenp->getType())->getElementType(),
                       lenp);
#else
    len = B.CreateLoad(lenp);
#endif
    B.CreateCondBr(B.CreateICmpEQ(len, ConstantInt::get(len->getType(), 0)),
                   end, body);
  }

  {
    IRBuilder<> B(body);
    B.setFastMathFlags(getFast());
    PHINode *idx = B.CreatePHI(len->getType(), 2, "idx");
    idx->addIncoming(ConstantInt::get(len->getType(), 0), entry);

#if LLVM_VERSION_MAJOR > 7
    Value *dsti = B.CreateGEP(
        cast<PointerType>(dst->getType())->getElementType(), dst, idx, "dst.i");
    LoadInst *dstl = B.CreateLoad(
        cast<PointerType>(dsti->getType())->getElementType(), dsti, "dst.i.l");

    Value *srci = B.CreateGEP(
        cast<PointerType>(src->getType())->getElementType(), src, idx, "src.i");
    LoadInst *srcl = B.CreateLoad(
        cast<PointerType>(srci->getType())->getElementType(), srci, "src.i.l");
#else
    Value *dsti = B.CreateGEP(dst, idx, "dst.i");
    LoadInst *dstl = B.CreateLoad(dsti, "dst.i.l");

    Value *srci = B.CreateGEP(src, idx, "src.i");
    LoadInst *srcl = B.CreateLoad(srci, "src.i.l");
#endif

    B.CreateStore(B.CreateFAdd(srcl, dstl), dsti);

    Value *next =
        B.CreateNUWAdd(idx, ConstantInt::get(len->getType(), 1), "idx.next");
    idx->addIncoming(next, body);
    B.CreateCondBr(B.CreateICmpEQ(len, next), end, body);
  }

  {
    IRBuilder<> B(end);
    B.CreateRetVoid();
  }

  std::vector<llvm::Type *> rtypes = {Type::getInt8PtrTy(M.getContext()),
                                      intType, OpPtr};
  FunctionType *RFT = FunctionType::get(intType, rtypes, false);

  Constant *RF = M.getNamedValue("MPI_Op_create");
  if (!RF) {
#if LLVM_VERSION_MAJOR >= 9
    RF =
        cast<Function>(M.getOrInsertFunction("MPI_Op_create", RFT).getCallee());
#else
    RF = cast<Function>(M.getOrInsertFunction("MPI_Op_create", RFT));
#endif
  } else {
    RF = ConstantExpr::getBitCast(RF, PointerType::getUnqual(RFT));
  }

  GlobalVariable *GV = new GlobalVariable(
      M, cast<PointerType>(OpPtr)->getElementType(), false,
      GlobalVariable::InternalLinkage,
      UndefValue::get(cast<PointerType>(OpPtr)->getElementType()), name);

  Type *i1Ty = Type::getInt1Ty(M.getContext());
  GlobalVariable *initD = new GlobalVariable(
      M, i1Ty, false, GlobalVariable::InternalLinkage,
      ConstantInt::getFalse(M.getContext()), name + "_initd");

  // Finish initializing mpi sum
  // https://www.mpich.org/static/docs/v3.2/www3/MPI_Op_create.html
  FunctionType *IFT = FunctionType::get(Type::getVoidTy(M.getContext()),
                                        ArrayRef<Type *>(), false);

#if LLVM_VERSION_MAJOR >= 9
  Function *initializerFunction = cast<Function>(
      M.getOrInsertFunction(name + "initializer", IFT).getCallee());
#else
  Function *initializerFunction =
      cast<Function>(M.getOrInsertFunction(name + "initializer", IFT));
#endif

  initializerFunction->setLinkage(Function::LinkageTypes::InternalLinkage);
  initializerFunction->addFnAttr(Attribute::NoUnwind);

  {
    BasicBlock *entry =
        BasicBlock::Create(M.getContext(), "entry", initializerFunction);
    BasicBlock *run =
        BasicBlock::Create(M.getContext(), "run", initializerFunction);
    BasicBlock *end =
        BasicBlock::Create(M.getContext(), "end", initializerFunction);
    IRBuilder<> B(entry);
#if LLVM_VERSION_MAJOR > 7
    B.CreateCondBr(
        B.CreateLoad(cast<PointerType>(initD->getType())->getElementType(),
                     initD),
        end, run);
#else
    B.CreateCondBr(B.CreateLoad(initD), end, run);
#endif

    B.SetInsertPoint(run);
    Value *args[] = {ConstantExpr::getPointerCast(F, rtypes[0]),
                     ConstantInt::get(rtypes[1], 1, false),
                     ConstantExpr::getPointerCast(GV, rtypes[2])};
    B.CreateCall(RFT, RF, args);
    B.CreateStore(ConstantInt::getTrue(M.getContext()), initD);
    B.CreateBr(end);
    B.SetInsertPoint(end);
    B.CreateRetVoid();
  }

  B2.CreateCall(M.getFunction(name + "initializer"));
#if LLVM_VERSION_MAJOR > 7
  return B2.CreateLoad(GV->getValueType(), GV);
#else
  return B2.CreateLoad(GV);
#endif
}

Function *getOrInsertExponentialAllocator(Module &M, bool ZeroInit) {
  Type *BPTy = Type::getInt8PtrTy(M.getContext());
  Type *types[] = {BPTy, Type::getInt64Ty(M.getContext()),
                   Type::getInt64Ty(M.getContext())};
  std::string name = "__enzyme_exponentialallocation";
  if (ZeroInit)
    name += "zero";
  FunctionType *FT =
      FunctionType::get(Type::getInt8PtrTy(M.getContext()), types, false);

#if LLVM_VERSION_MAJOR >= 9
  Function *F = cast<Function>(M.getOrInsertFunction(name, FT).getCallee());
#else
  Function *F = cast<Function>(M.getOrInsertFunction(name, FT));
#endif

  if (!F->empty())
    return F;

  F->setLinkage(Function::LinkageTypes::InternalLinkage);
  F->addFnAttr(Attribute::AlwaysInline);
  F->addFnAttr(Attribute::NoUnwind);
  BasicBlock *entry = BasicBlock::Create(M.getContext(), "entry", F);
  BasicBlock *grow = BasicBlock::Create(M.getContext(), "grow", F);
  BasicBlock *ok = BasicBlock::Create(M.getContext(), "ok", F);

  IRBuilder<> B(entry);

  Argument *ptr = F->arg_begin();
  ptr->setName("ptr");
  Argument *size = ptr + 1;
  size->setName("size");
  Argument *tsize = size + 1;
  tsize->setName("tsize");

  Value *hasOne = B.CreateICmpNE(
      B.CreateAnd(size, ConstantInt::get(size->getType(), 1, false)),
      ConstantInt::get(size->getType(), 0, false));
  auto popCnt = Intrinsic::getDeclaration(&M, Intrinsic::ctpop,
                                          std::vector<Type *>({types[1]}));

  B.CreateCondBr(
      B.CreateAnd(
          B.CreateICmpULT(B.CreateCall(popCnt, std::vector<Value *>({size})),
                          ConstantInt::get(types[1], 3, false)),
          hasOne),
      grow, ok);

  B.SetInsertPoint(grow);

  auto lz = B.CreateCall(
      Intrinsic::getDeclaration(&M, Intrinsic::ctlz,
                                std::vector<Type *>({types[1]})),
      std::vector<Value *>({size, ConstantInt::getTrue(M.getContext())}));
  Value *next =
      B.CreateShl(tsize, B.CreateSub(ConstantInt::get(types[1], 64, false), lz,
                                     "", true, true));

  auto reallocF = M.getOrInsertFunction("realloc", BPTy, BPTy,
                                        Type::getInt64Ty(M.getContext()));

  Value *args[] = {B.CreatePointerCast(ptr, BPTy), next};
  Value *gVal =
      B.CreatePointerCast(B.CreateCall(reallocF, args), ptr->getType());
  if (ZeroInit) {
    Value *prevSize = B.CreateSelect(
        B.CreateICmpEQ(size, ConstantInt::get(size->getType(), 1)),
        ConstantInt::get(next->getType(), 0),
        B.CreateLShr(next, ConstantInt::get(next->getType(), 1)));

    Value *zeroSize = B.CreateSub(next, prevSize);

    Value *margs[] = {
#if LLVM_VERSION_MAJOR > 7
      B.CreateGEP(cast<PointerType>(gVal->getType())->getElementType(), gVal,
                  prevSize),
#else
      B.CreateGEP(gVal, prevSize),
#endif
      ConstantInt::get(Type::getInt8Ty(args[0]->getContext()), 0),
      zeroSize,
      ConstantInt::getFalse(args[0]->getContext())
    };
    Type *tys[] = {margs[0]->getType(), margs[2]->getType()};
    auto memsetF = Intrinsic::getDeclaration(&M, Intrinsic::memset, tys);
    B.CreateCall(memsetF, margs);
  }

  B.CreateBr(ok);
  B.SetInsertPoint(ok);
  auto phi = B.CreatePHI(ptr->getType(), 2);
  phi->addIncoming(gVal, grow);
  phi->addIncoming(ptr, entry);
  B.CreateRet(phi);
  return F;
}
