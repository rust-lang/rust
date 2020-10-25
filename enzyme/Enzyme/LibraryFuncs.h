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
#include "llvm/IR/Instructions.h"

/// Return whether a given function is a known C/C++ memory allocation function
/// For updating below one should read MemoryBuiltins.cpp, TargetLibraryInfo.cpp
static inline bool isAllocationFunction(const llvm::Function &F,
                                        const llvm::TargetLibraryInfo &TLI) {
  if (F.getName() == "__rust_alloc" || F.getName() == "__rust_alloc_zeroed")
    return true;                                  
  using namespace llvm;
  llvm::LibFunc libfunc;
  if (!TLI.getLibFunc(F, libfunc))
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
static inline bool isDeallocationFunction(const llvm::Function &F,
                                          const llvm::TargetLibraryInfo &TLI) {
  using namespace llvm;
  llvm::LibFunc libfunc;
  if (!TLI.getLibFunc(F, libfunc)) {
    if (F.getName() == "free")
      return true;
    if (F.getName() == "__rust_dealloc")
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

/// Perform the corresponding deallocation of tofree, given it was allocated by
/// allocationfn
// For updating below one should read MemoryBuiltins.cpp, TargetLibraryInfo.cpp
static inline llvm::CallInst *
freeKnownAllocation(llvm::IRBuilder<> &builder, llvm::Value *tofree,
                    llvm::Function &allocationfn,
                    const llvm::TargetLibraryInfo &TLI) {
  using namespace llvm;
  assert(isAllocationFunction(allocationfn, TLI));

  if (allocationfn.getName() == "__rust_alloc" || allocationfn.getName() == "__rust_alloc_zeroed") {
    llvm_unreachable("todo - hook in rust allocation fns");
  }

  llvm::LibFunc libfunc;
  bool res = TLI.getLibFunc(allocationfn, libfunc);
  assert(res && "ought find known allocation fn");

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

  Type *VoidTy = Type::getVoidTy(tofree->getContext());
  Type *IntPtrTy = Type::getInt8PtrTy(tofree->getContext());

  auto FT = FunctionType::get(VoidTy, {IntPtrTy}, false);
#if LLVM_VERSION_MAJOR >= 9
  Value *freevalue =
      allocationfn.getParent()->getOrInsertFunction(freename, FT).getCallee();
#else
  Value *freevalue =
      allocationfn.getParent()->getOrInsertFunction(freename, FT);
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
  if (isa<CallInst>(tofree) &&
      cast<CallInst>(tofree)->getAttributes().hasAttribute(
          AttributeList::ReturnIndex, Attribute::NonNull)) {
    freecall->addAttribute(AttributeList::FirstArgIndex, Attribute::NonNull);
  }
  if (Function *F = dyn_cast<Function>(freevalue))
    freecall->setCallingConv(F->getCallingConv());
  if (freecall->getParent() == nullptr)
    builder.Insert(freecall);
  return freecall;
}

/// Return whether maybeReader can read from memory written to by maybeWriter
static inline bool writesToMemoryReadBy(llvm::AAResults &AA,
                                        llvm::Instruction *maybeReader,
                                        llvm::Instruction *maybeWriter) {
  using namespace llvm;
  if (auto call = dyn_cast<CallInst>(maybeWriter)) {
    if (call->getCalledFunction() &&
        isCertainMallocOrFree(call->getCalledFunction())) {
      return false;
    }
  }
  if (auto call = dyn_cast<CallInst>(maybeReader)) {
    if (call->getCalledFunction() &&
        isCertainMallocOrFree(call->getCalledFunction())) {
      return false;
    }
  }
  if (auto call = dyn_cast<InvokeInst>(maybeWriter)) {
    if (call->getCalledFunction() &&
        isCertainMallocOrFree(call->getCalledFunction())) {
      return false;
    }
  }
  if (auto call = dyn_cast<InvokeInst>(maybeReader)) {
    if (call->getCalledFunction() &&
        isCertainMallocOrFree(call->getCalledFunction())) {
      return false;
    }
  }
  assert(maybeWriter->mayWriteToMemory());
  assert(maybeReader->mayReadFromMemory());

  if (auto li = dyn_cast<LoadInst>(maybeReader)) {
    return isModSet(AA.getModRefInfo(maybeWriter, MemoryLocation::get(li)));
  }
  if (auto rmw = dyn_cast<AtomicRMWInst>(maybeReader)) {
    return isModSet(AA.getModRefInfo(maybeWriter, MemoryLocation::get(rmw)));
  }
  if (auto xch = dyn_cast<AtomicCmpXchgInst>(maybeReader)) {
    return isModSet(AA.getModRefInfo(maybeWriter, MemoryLocation::get(xch)));
  }
  if (auto mti = dyn_cast<MemTransferInst>(maybeReader)) {
    return isModSet(
        AA.getModRefInfo(maybeWriter, MemoryLocation::getForSource(mti)));
  }

  if (auto si = dyn_cast<StoreInst>(maybeWriter)) {
    return isRefSet(AA.getModRefInfo(maybeReader, MemoryLocation::get(si)));
  }
  if (auto rmw = dyn_cast<AtomicRMWInst>(maybeWriter)) {
    return isRefSet(AA.getModRefInfo(maybeReader, MemoryLocation::get(rmw)));
  }
  if (auto xch = dyn_cast<AtomicCmpXchgInst>(maybeWriter)) {
    return isRefSet(AA.getModRefInfo(maybeReader, MemoryLocation::get(xch)));
  }
  if (auto mti = dyn_cast<MemIntrinsic>(maybeWriter)) {
    return isRefSet(
        AA.getModRefInfo(maybeReader, MemoryLocation::getForDest(mti)));
  }

  if (auto cb = dyn_cast<CallInst>(maybeReader)) {
    // llvm::errs() << " considering: " << *cb << " and: " << *maybeWriter <<
    // "\n";
    return isModOrRefSet(AA.getModRefInfo(maybeWriter, cb));
  }
  if (auto cb = dyn_cast<InvokeInst>(maybeReader)) {
    return isModOrRefSet(AA.getModRefInfo(maybeWriter, cb));
  }
  llvm::errs() << " maybeReader: " << *maybeReader
               << " maybeWriter: " << *maybeWriter << "\n";
  llvm_unreachable("unknown inst2");
}

#endif
