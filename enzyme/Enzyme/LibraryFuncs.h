//===- LibraryFuncs.h - Utilities for handling library functions ---------===//
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
// Automatically Synthesize Fast Gradients}, author = {Moses, William S. and
// Churavy, Valentin}, booktitle = {Advances in Neural Information Processing
// Systems 33}, year = {2020}, note = {To appear in},
// }
//
//===----------------------------------------------------------------------===//
//
// This file defines miscelaious utilities for handling library functions.
//
//===----------------------------------------------------------------------===//

#ifndef LIBRARYFUNCS_H_
#define LIBRARYFUNCS_H_

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"

extern std::map<std::string, std::function<llvm::Value *(
                                 llvm::IRBuilder<> &, llvm::CallInst *,
                                 llvm::ArrayRef<llvm::Value *>)>>
    shadowHandlers;
extern std::map<std::string, std::function<llvm::CallInst *(llvm::IRBuilder<> &,
                                                            llvm::Value *)>>
    shadowErasers;

/// Return whether a given function is a known C/C++ memory allocation function
/// For updating below one should read MemoryBuiltins.cpp, TargetLibraryInfo.cpp
static inline bool isAllocationFunction(const llvm::StringRef name,
                                        const llvm::TargetLibraryInfo &TLI) {
  if (name == "calloc" || name == "malloc")
    return true;
  if (name == "swift_allocObject")
    return true;
  if (name == "__rust_alloc" || name == "__rust_alloc_zeroed")
    return true;
  if (name == "julia.gc_alloc_obj")
    return true;
  if (shadowHandlers.find(name.str()) != shadowHandlers.end())
    return true;

  using namespace llvm;
  llvm::LibFunc libfunc;
  if (!TLI.getLibFunc(name, libfunc))
    return false;

  switch (libfunc) {
  case LibFunc_malloc: // malloc(unsigned int);
  case LibFunc_valloc: // valloc(unsigned int);

  case LibFunc_Znwj:               // new(unsigned int);
  case LibFunc_ZnwjRKSt9nothrow_t: // new(unsigned int, nothrow);
#if LLVM_VERSION_MAJOR > 6
  case LibFunc_ZnwjSt11align_val_t: // new(unsigned int, align_val_t)
  case LibFunc_ZnwjSt11align_val_tRKSt9nothrow_t: // new(unsigned int,
                                                  // align_val_t, nothrow)
#endif

  case LibFunc_Znwm:               // new(unsigned long);
  case LibFunc_ZnwmRKSt9nothrow_t: // new(unsigned long, nothrow);
#if LLVM_VERSION_MAJOR > 6
  case LibFunc_ZnwmSt11align_val_t: // new(unsigned long, align_val_t)
  case LibFunc_ZnwmSt11align_val_tRKSt9nothrow_t: // new(unsigned long,
                                                  // align_val_t, nothrow)
#endif

  case LibFunc_Znaj:               // new[](unsigned int);
  case LibFunc_ZnajRKSt9nothrow_t: // new[](unsigned int, nothrow);
#if LLVM_VERSION_MAJOR > 6
  case LibFunc_ZnajSt11align_val_t: // new[](unsigned int, align_val_t)
  case LibFunc_ZnajSt11align_val_tRKSt9nothrow_t: // new[](unsigned int,
                                                  // align_val_t, nothrow)
#endif

  case LibFunc_Znam:               // new[](unsigned long);
  case LibFunc_ZnamRKSt9nothrow_t: // new[](unsigned long, nothrow);
#if LLVM_VERSION_MAJOR > 6
  case LibFunc_ZnamSt11align_val_t: // new[](unsigned long, align_val_t)
  case LibFunc_ZnamSt11align_val_tRKSt9nothrow_t: // new[](unsigned long,
                                                  // align_val_t, nothrow)
#endif

  case LibFunc_msvc_new_int:               // new(unsigned int);
  case LibFunc_msvc_new_int_nothrow:       // new(unsigned int, nothrow);
  case LibFunc_msvc_new_longlong:          // new(unsigned long long);
  case LibFunc_msvc_new_longlong_nothrow:  // new(unsigned long long, nothrow);
  case LibFunc_msvc_new_array_int:         // new[](unsigned int);
  case LibFunc_msvc_new_array_int_nothrow: // new[](unsigned int, nothrow);
  case LibFunc_msvc_new_array_longlong:    // new[](unsigned long long);
  case LibFunc_msvc_new_array_longlong_nothrow: // new[](unsigned long long,
                                                // nothrow);

    // TODO strdup, strndup

    // TODO call, realloc, reallocf

    // TODO (perhaps) posix_memalign
    return true;
  default:
    return false;
  }
}

/// Return whether a given function is a known C/C++ memory deallocation
/// function For updating below one should read MemoryBuiltins.cpp,
/// TargetLibraryInfo.cpp
static inline bool isDeallocationFunction(const llvm::StringRef name,
                                          const llvm::TargetLibraryInfo &TLI) {
  using namespace llvm;
  llvm::LibFunc libfunc;
  if (!TLI.getLibFunc(name, libfunc)) {
    if (name == "free")
      return true;
    if (name == "__rust_dealloc")
      return true;
    if (name == "swift_release")
      return true;
    return false;
  }

  switch (libfunc) {
  // void free(void*);
  case LibFunc_free:

  // void operator delete[](void*);
  case LibFunc_ZdaPv:
  // void operator delete(void*);
  case LibFunc_ZdlPv:
  // void operator delete[](void*);
  case LibFunc_msvc_delete_array_ptr32:
  // void operator delete[](void*);
  case LibFunc_msvc_delete_array_ptr64:
  // void operator delete(void*);
  case LibFunc_msvc_delete_ptr32:
  // void operator delete(void*);
  case LibFunc_msvc_delete_ptr64:

  // void operator delete[](void*, nothrow);
  case LibFunc_ZdaPvRKSt9nothrow_t:
  // void operator delete[](void*, unsigned int);
  case LibFunc_ZdaPvj:
  // void operator delete[](void*, unsigned long);
  case LibFunc_ZdaPvm:
  // void operator delete(void*, nothrow);
  case LibFunc_ZdlPvRKSt9nothrow_t:
  // void operator delete(void*, unsigned int);
  case LibFunc_ZdlPvj:
  // void operator delete(void*, unsigned long);
  case LibFunc_ZdlPvm:
#if LLVM_VERSION_MAJOR > 6
  // void operator delete(void*, align_val_t)
  case LibFunc_ZdlPvSt11align_val_t:
  // void operator delete[](void*, align_val_t)
  case LibFunc_ZdaPvSt11align_val_t:
#endif
  // void operator delete[](void*, unsigned int);
  case LibFunc_msvc_delete_array_ptr32_int:
  // void operator delete[](void*, nothrow);
  case LibFunc_msvc_delete_array_ptr32_nothrow:
  // void operator delete[](void*, unsigned long long);
  case LibFunc_msvc_delete_array_ptr64_longlong:
  // void operator delete[](void*, nothrow);
  case LibFunc_msvc_delete_array_ptr64_nothrow:
  // void operator delete(void*, unsigned int);
  case LibFunc_msvc_delete_ptr32_int:
  // void operator delete(void*, nothrow);
  case LibFunc_msvc_delete_ptr32_nothrow:
  // void operator delete(void*, unsigned long long);
  case LibFunc_msvc_delete_ptr64_longlong:
  // void operator delete(void*, nothrow);
  case LibFunc_msvc_delete_ptr64_nothrow:

#if LLVM_VERSION_MAJOR > 6
  // void operator delete(void*, align_val_t, nothrow)
  case LibFunc_ZdlPvSt11align_val_tRKSt9nothrow_t:
  // void operator delete[](void*, align_val_t, nothrow)
  case LibFunc_ZdaPvSt11align_val_tRKSt9nothrow_t:
#endif
    return true;
  default:
    return false;
  }
}

static inline void zeroKnownAllocation(llvm::IRBuilder<> &bb,
                                       llvm::Value *toZero,
                                       llvm::ArrayRef<llvm::Value *> argValues,
                                       const llvm::StringRef funcName,
                                       const llvm::TargetLibraryInfo &TLI) {
  using namespace llvm;
  assert(isAllocationFunction(funcName, TLI));

  // Don't re-zero an already-zero buffer
  if (funcName == "calloc" || funcName == "__rust_alloc_zeroed")
    return;

  Value *allocSize = argValues[0];
  if (funcName == "julia.gc_alloc_obj") {
    Type *tys[] = {PointerType::get(StructType::get(toZero->getContext()), 10)};
    FunctionType *FT =
        FunctionType::get(Type::getVoidTy(toZero->getContext()), tys, true);
    bb.CreateCall(
        bb.GetInsertBlock()->getParent()->getParent()->getOrInsertFunction(
            "julia.write_barrier", FT),
        toZero);
    allocSize = argValues[1];
  }
  Value *dst_arg = toZero;

  if (dst_arg->getType()->isIntegerTy())
    dst_arg =
        bb.CreateIntToPtr(dst_arg, Type::getInt8PtrTy(toZero->getContext()));
  else
    dst_arg = bb.CreateBitCast(
        dst_arg,
        Type::getInt8PtrTy(toZero->getContext(),
                           toZero->getType()->getPointerAddressSpace()));

  auto val_arg = ConstantInt::get(Type::getInt8Ty(toZero->getContext()), 0);
  auto len_arg =
      bb.CreateZExtOrTrunc(allocSize, Type::getInt64Ty(toZero->getContext()));
  auto volatile_arg = ConstantInt::getFalse(toZero->getContext());

#if LLVM_VERSION_MAJOR == 6
  auto align_arg = ConstantInt::get(Type::getInt32Ty(toZero->getContext()), 1);
  Value *nargs[] = {dst_arg, val_arg, len_arg, align_arg, volatile_arg};
#else
  Value *nargs[] = {dst_arg, val_arg, len_arg, volatile_arg};
#endif

  Type *tys[] = {dst_arg->getType(), len_arg->getType()};

  auto memset = cast<CallInst>(bb.CreateCall(
      Intrinsic::getDeclaration(bb.GetInsertBlock()->getParent()->getParent(),
                                Intrinsic::memset, tys),
      nargs));
  memset->addParamAttr(0, Attribute::NonNull);
  if (auto CI = dyn_cast<ConstantInt>(allocSize)) {
    auto derefBytes = CI->getLimitedValue();
#if LLVM_VERSION_MAJOR >= 14
    memset->addDereferenceableParamAttr(0, derefBytes);
    memset->setAttributes(
        memset->getAttributes().addDereferenceableOrNullParamAttr(
            memset->getContext(), 0, derefBytes));
#else
    memset->addDereferenceableAttr(llvm::AttributeList::FirstArgIndex,
                                   derefBytes);
    memset->addDereferenceableOrNullAttr(llvm::AttributeList::FirstArgIndex,
                                         derefBytes);
#endif
  }
}

/// Perform the corresponding deallocation of tofree, given it was allocated by
/// allocationfn
// For updating below one should read MemoryBuiltins.cpp, TargetLibraryInfo.cpp
static inline llvm::CallInst *
freeKnownAllocation(llvm::IRBuilder<> &builder, llvm::Value *tofree,
                    const llvm::StringRef allocationfn,
                    const llvm::DebugLoc &debuglocation,
                    const llvm::TargetLibraryInfo &TLI) {
  using namespace llvm;
  assert(isAllocationFunction(allocationfn, TLI));

  if (allocationfn == "__rust_alloc" || allocationfn == "__rust_alloc_zeroed") {
    llvm_unreachable("todo - hook in rust allocation fns");
  }
  if (allocationfn == "julia.gc_alloc_obj")
    return nullptr;

  if (allocationfn == "swift_allocObject") {
    Type *VoidTy = Type::getVoidTy(tofree->getContext());
    Type *IntPtrTy = Type::getInt8PtrTy(tofree->getContext());

    auto FT = FunctionType::get(VoidTy, ArrayRef<Type *>(IntPtrTy), false);
#if LLVM_VERSION_MAJOR >= 9
    Value *freevalue = builder.GetInsertBlock()
                           ->getParent()
                           ->getParent()
                           ->getOrInsertFunction("swift_release", FT)
                           .getCallee();
#else
    Value *freevalue =
        builder.GetInsertBlock()->getParent()->getParent()->getOrInsertFunction(
            "swift_release", FT);
#endif
    CallInst *freecall = cast<CallInst>(
#if LLVM_VERSION_MAJOR >= 8
        CallInst::Create(
            FT, freevalue,
            ArrayRef<Value *>(builder.CreatePointerCast(tofree, IntPtrTy)),
#else
        CallInst::Create(
            freevalue,
            ArrayRef<Value *>(builder.CreatePointerCast(tofree, IntPtrTy)),
#endif
            "", builder.GetInsertBlock()));
    freecall->setDebugLoc(debuglocation);
    freecall->setTailCall();
    if (isa<CallInst>(tofree) &&
#if LLVM_VERSION_MAJOR >= 14
        cast<CallInst>(tofree)->getAttributes().hasAttributeAtIndex(
            AttributeList::ReturnIndex, Attribute::NonNull)
#else
        cast<CallInst>(tofree)->getAttributes().hasAttribute(
            AttributeList::ReturnIndex, Attribute::NonNull)
#endif
    ) {
#if LLVM_VERSION_MAJOR >= 14
      freecall->addAttributeAtIndex(AttributeList::FirstArgIndex,
                                    Attribute::NonNull);
#else
      freecall->addAttribute(AttributeList::FirstArgIndex, Attribute::NonNull);
#endif
    }
    if (Function *F = dyn_cast<Function>(freevalue))
      freecall->setCallingConv(F->getCallingConv());
    if (freecall->getParent() == nullptr)
      builder.Insert(freecall);
    return freecall;
  }

  if (shadowErasers.find(allocationfn.str()) != shadowErasers.end()) {
    return shadowErasers[allocationfn.str()](builder, tofree);
  }

  if (tofree->getType()->isIntegerTy())
    tofree = builder.CreateIntToPtr(tofree,
                                    Type::getInt8PtrTy(tofree->getContext()));

  llvm::LibFunc libfunc;
  if (allocationfn == "calloc" || allocationfn == "malloc") {
    libfunc = LibFunc_malloc;
  } else {
    bool res = TLI.getLibFunc(allocationfn, libfunc);
    assert(res && "ought find known allocation fn");
  }

  llvm::LibFunc freefunc;

  switch (libfunc) {
  case LibFunc_malloc: // malloc(unsigned int);
  case LibFunc_valloc: // valloc(unsigned int);
    freefunc = LibFunc_free;
    break;

  case LibFunc_Znwj:               // new(unsigned int);
  case LibFunc_ZnwjRKSt9nothrow_t: // new(unsigned int, nothrow);
#if LLVM_VERSION_MAJOR > 6
  case LibFunc_ZnwjSt11align_val_t: // new(unsigned int, align_val_t)
  case LibFunc_ZnwjSt11align_val_tRKSt9nothrow_t: // new(unsigned int,
                                                  // align_val_t, nothrow)
#endif

  case LibFunc_Znwm:               // new(unsigned long);
  case LibFunc_ZnwmRKSt9nothrow_t: // new(unsigned long, nothrow);
#if LLVM_VERSION_MAJOR > 6
  case LibFunc_ZnwmSt11align_val_t: // new(unsigned long, align_val_t)
  case LibFunc_ZnwmSt11align_val_tRKSt9nothrow_t: // new(unsigned long,
                                                  // align_val_t, nothrow)
#endif
    freefunc = LibFunc_ZdlPv;
    break;

  case LibFunc_Znaj:               // new[](unsigned int);
  case LibFunc_ZnajRKSt9nothrow_t: // new[](unsigned int, nothrow);
#if LLVM_VERSION_MAJOR > 6
  case LibFunc_ZnajSt11align_val_t: // new[](unsigned int, align_val_t)
  case LibFunc_ZnajSt11align_val_tRKSt9nothrow_t: // new[](unsigned int,
                                                  // align_val_t, nothrow)
#endif

  case LibFunc_Znam:               // new[](unsigned long);
  case LibFunc_ZnamRKSt9nothrow_t: // new[](unsigned long, nothrow);
#if LLVM_VERSION_MAJOR > 6
  case LibFunc_ZnamSt11align_val_t: // new[](unsigned long, align_val_t)
  case LibFunc_ZnamSt11align_val_tRKSt9nothrow_t: // new[](unsigned long,
                                                  // align_val_t, nothrow)
#endif
    freefunc = LibFunc_ZdaPv;
    break;

  case LibFunc_msvc_new_int:               // new(unsigned int);
  case LibFunc_msvc_new_int_nothrow:       // new(unsigned int, nothrow);
  case LibFunc_msvc_new_longlong:          // new(unsigned long long);
  case LibFunc_msvc_new_longlong_nothrow:  // new(unsigned long long, nothrow);
  case LibFunc_msvc_new_array_int:         // new[](unsigned int);
  case LibFunc_msvc_new_array_int_nothrow: // new[](unsigned int, nothrow);
  case LibFunc_msvc_new_array_longlong:    // new[](unsigned long long);
  case LibFunc_msvc_new_array_longlong_nothrow: // new[](unsigned long long,
                                                // nothrow);
    llvm_unreachable("msvc deletion not handled");

  default:
    llvm_unreachable("unknown allocation function");
  }
  llvm::StringRef freename = TLI.getName(freefunc);
  if (freefunc == LibFunc_free) {
    freename = "free";
    assert(freename == "free");
    if (freename != "free")
      llvm_unreachable("illegal free");
  }

  Type *VoidTy = Type::getVoidTy(tofree->getContext());
  Type *IntPtrTy = Type::getInt8PtrTy(tofree->getContext());

  auto FT = FunctionType::get(VoidTy, {IntPtrTy}, false);
#if LLVM_VERSION_MAJOR >= 9
  Value *freevalue = builder.GetInsertBlock()
                         ->getParent()
                         ->getParent()
                         ->getOrInsertFunction(freename, FT)
                         .getCallee();
#else
  Value *freevalue =
      builder.GetInsertBlock()->getParent()->getParent()->getOrInsertFunction(
          freename, FT);
#endif
  CallInst *freecall = cast<CallInst>(
#if LLVM_VERSION_MAJOR >= 8
      CallInst::Create(FT, freevalue,
                       {builder.CreatePointerCast(tofree, IntPtrTy)},
#else
      CallInst::Create(freevalue, {builder.CreatePointerCast(tofree, IntPtrTy)},
#endif
                       "", builder.GetInsertBlock()));
  freecall->setTailCall();
  freecall->setDebugLoc(debuglocation);
  if (isa<CallInst>(tofree) &&
#if LLVM_VERSION_MAJOR >= 14
      cast<CallInst>(tofree)->getAttributes().hasAttributeAtIndex(
          AttributeList::ReturnIndex, Attribute::NonNull)
#else
      cast<CallInst>(tofree)->getAttributes().hasAttribute(
          AttributeList::ReturnIndex, Attribute::NonNull)
#endif
  ) {
#if LLVM_VERSION_MAJOR >= 14
    freecall->addAttributeAtIndex(AttributeList::FirstArgIndex,
                                  Attribute::NonNull);
#else
    freecall->addAttribute(AttributeList::FirstArgIndex, Attribute::NonNull);
#endif
  }
  if (Function *F = dyn_cast<Function>(freevalue))
    freecall->setCallingConv(F->getCallingConv());
  if (freecall->getParent() == nullptr)
    builder.Insert(freecall);
  return freecall;
}

static inline bool isAllocationCall(llvm::Value *TmpOrig,
                                    llvm::TargetLibraryInfo &TLI) {
  if (auto *CI = llvm::dyn_cast<llvm::CallInst>(TmpOrig)) {
    return isAllocationFunction(getFuncNameFromCall(CI), TLI);
  }
  if (auto *CI = llvm::dyn_cast<llvm::InvokeInst>(TmpOrig)) {
    return isAllocationFunction(getFuncNameFromCall(CI), TLI);
  }
  return false;
}

static inline bool isDeallocationCall(llvm::Value *TmpOrig,
                                      llvm::TargetLibraryInfo &TLI) {
  if (auto *CI = llvm::dyn_cast<llvm::CallInst>(TmpOrig)) {
    return isDeallocationFunction(getFuncNameFromCall(CI), TLI);
  }
  if (auto *CI = llvm::dyn_cast<llvm::InvokeInst>(TmpOrig)) {
    return isDeallocationFunction(getFuncNameFromCall(CI), TLI);
  }
  return false;
}

#endif
