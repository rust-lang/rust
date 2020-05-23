/*
 * LibraryFuncs.h - Library Function Utilities
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

#ifndef LIBRARYFUNCS_H_
#define LIBRARYFUNCS_H_

#include "llvm/Analysis/TargetLibraryInfo.h"


//For updating below one should read MemoryBuiltins.cpp, TargetLibraryInfo.cpp
static inline
bool isAllocationFunction(const llvm::Function &F, const llvm::TargetLibraryInfo &TLI) {
	llvm::LibFunc libfunc;
	if (!TLI.getLibFunc(F, libfunc)) return false;

	switch (libfunc) {
  		case LibFunc_malloc: // malloc(unsigned int);
		case LibFunc_valloc: // valloc(unsigned int);

		case LibFunc_Znwj: // new(unsigned int);
		case LibFunc_ZnwjRKSt9nothrow_t: // new(unsigned int, nothrow);
#if LLVM_VERSION_MAJOR > 6
		case LibFunc_ZnwjSt11align_val_t: // new(unsigned int, align_val_t)
		case LibFunc_ZnwjSt11align_val_tRKSt9nothrow_t: // new(unsigned int, align_val_t, nothrow)
#endif

		case LibFunc_Znwm: // new(unsigned long);
		case LibFunc_ZnwmRKSt9nothrow_t: // new(unsigned long, nothrow);
#if LLVM_VERSION_MAJOR > 6
		case LibFunc_ZnwmSt11align_val_t: // new(unsigned long, align_val_t)
		case LibFunc_ZnwmSt11align_val_tRKSt9nothrow_t: // new(unsigned long, align_val_t, nothrow)
#endif

		case LibFunc_Znaj: // new[](unsigned int);
		case LibFunc_ZnajRKSt9nothrow_t: // new[](unsigned int, nothrow);
#if LLVM_VERSION_MAJOR > 6
		case LibFunc_ZnajSt11align_val_t: // new[](unsigned int, align_val_t)
		case LibFunc_ZnajSt11align_val_tRKSt9nothrow_t: // new[](unsigned int, align_val_t, nothrow)
#endif


		case LibFunc_Znam: // new[](unsigned long);
		case LibFunc_ZnamRKSt9nothrow_t: // new[](unsigned long, nothrow);
#if LLVM_VERSION_MAJOR > 6
		case LibFunc_ZnamSt11align_val_t: // new[](unsigned long, align_val_t)
		case LibFunc_ZnamSt11align_val_tRKSt9nothrow_t: // new[](unsigned long, align_val_t, nothrow)
#endif

		case LibFunc_msvc_new_int: // new(unsigned int);
		case LibFunc_msvc_new_int_nothrow: // new(unsigned int, nothrow);
		case LibFunc_msvc_new_longlong: // new(unsigned long long);
		case LibFunc_msvc_new_longlong_nothrow: // new(unsigned long long, nothrow);
		case LibFunc_msvc_new_array_int: // new[](unsigned int);
		case LibFunc_msvc_new_array_int_nothrow: // new[](unsigned int, nothrow);
		case LibFunc_msvc_new_array_longlong: // new[](unsigned long long);
		case LibFunc_msvc_new_array_longlong_nothrow: // new[](unsigned long long, nothrow);

		//TODO strdup, strndup

		//TODO call, realloc, reallocf

		//TODO (perhaps) posix_memalign
			return true;
		default:
		  	return false;
	}
}

//For updating below one should read MemoryBuiltins.cpp, TargetLibraryInfo.cpp
static inline
bool isDeallocationFunction(const llvm::Function &F, const llvm::TargetLibraryInfo &TLI) {
	llvm::LibFunc libfunc;
	if (!TLI.getLibFunc(F, libfunc)) {
		if (F.getName() == "free") return true;
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

//For updating below one should read MemoryBuiltins.cpp, TargetLibraryInfo.cpp
static inline
CallInst* freeKnownAllocation(llvm::IRBuilder <>& builder, llvm::Value* tofree, llvm::Function& allocationfn, const llvm::TargetLibraryInfo &TLI) {
	assert(isAllocationFunction(allocationfn, TLI));

	llvm::LibFunc libfunc;
	bool res = TLI.getLibFunc(allocationfn, libfunc);
	assert(res && "ought find known allocation fn");

	llvm::LibFunc freefunc;

	switch (libfunc) {
  		case LibFunc_malloc: // malloc(unsigned int);
		case LibFunc_valloc: // valloc(unsigned int);
			freefunc = LibFunc_free;
			break;

		case LibFunc_Znwj: // new(unsigned int);
		case LibFunc_ZnwjRKSt9nothrow_t: // new(unsigned int, nothrow);
#if LLVM_VERSION_MAJOR > 6
		case LibFunc_ZnwjSt11align_val_t: // new(unsigned int, align_val_t)
		case LibFunc_ZnwjSt11align_val_tRKSt9nothrow_t: // new(unsigned int, align_val_t, nothrow)
#endif

		case LibFunc_Znwm: // new(unsigned long);
		case LibFunc_ZnwmRKSt9nothrow_t: // new(unsigned long, nothrow);
#if LLVM_VERSION_MAJOR > 6
		case LibFunc_ZnwmSt11align_val_t: // new(unsigned long, align_val_t)
		case LibFunc_ZnwmSt11align_val_tRKSt9nothrow_t: // new(unsigned long, align_val_t, nothrow)
#endif
			freefunc = LibFunc_ZdlPv;
			break;

		case LibFunc_Znaj: // new[](unsigned int);
		case LibFunc_ZnajRKSt9nothrow_t: // new[](unsigned int, nothrow);
#if LLVM_VERSION_MAJOR > 6
		case LibFunc_ZnajSt11align_val_t: // new[](unsigned int, align_val_t)
		case LibFunc_ZnajSt11align_val_tRKSt9nothrow_t: // new[](unsigned int, align_val_t, nothrow)
#endif

		case LibFunc_Znam: // new[](unsigned long);
		case LibFunc_ZnamRKSt9nothrow_t: // new[](unsigned long, nothrow);
#if LLVM_VERSION_MAJOR > 6
		case LibFunc_ZnamSt11align_val_t: // new[](unsigned long, align_val_t)
		case LibFunc_ZnamSt11align_val_tRKSt9nothrow_t: // new[](unsigned long, align_val_t, nothrow)
#endif
			freefunc = LibFunc_ZdaPv;
			break;

		case LibFunc_msvc_new_int: // new(unsigned int);
		case LibFunc_msvc_new_int_nothrow: // new(unsigned int, nothrow);
		case LibFunc_msvc_new_longlong: // new(unsigned long long);
		case LibFunc_msvc_new_longlong_nothrow: // new(unsigned long long, nothrow);
		case LibFunc_msvc_new_array_int: // new[](unsigned int);
		case LibFunc_msvc_new_array_int_nothrow: // new[](unsigned int, nothrow);
		case LibFunc_msvc_new_array_longlong: // new[](unsigned long long);
		case LibFunc_msvc_new_array_longlong_nothrow: // new[](unsigned long long, nothrow);
			llvm_unreachable("msvc deletion not handled");

		default:
			llvm_unreachable("unknown allocation function");
	}

	llvm::StringRef freename = TLI.getName(freefunc);

    Type *VoidTy = Type::getVoidTy(tofree->getContext());
    Type *IntPtrTy = Type::getInt8PtrTy(tofree->getContext());

    #if LLVM_VERSION_MAJOR >= 9
    Value* freevalue = allocationfn.getParent()->getOrInsertFunction(freename, FunctionType::get(VoidTy, {IntPtrTy}, false)).getCallee();
    #else
    Value* freevalue = allocationfn.getParent()->getOrInsertFunction(freename, FunctionType::get(VoidTy, {IntPtrTy}, false));
    #endif


	CallInst* freecall = cast<CallInst>(CallInst::Create(freevalue, {builder.CreatePointerCast(tofree, IntPtrTy)}, "", builder.GetInsertBlock()));
    freecall->setTailCall();
    if (isa<CallInst>(tofree) && cast<CallInst>(tofree)->getAttributes().hasAttribute(AttributeList::ReturnIndex, Attribute::NonNull)) {
    	freecall->addAttribute(AttributeList::FirstArgIndex, Attribute::NonNull);
	}
    if (Function *F = dyn_cast<Function>(freevalue)) freecall->setCallingConv(F->getCallingConv());
  	if (freecall->getParent()==nullptr) builder.Insert(freecall);
  	return freecall;
}

static inline bool writesToMemoryReadBy(AAResults &AA, Instruction* maybeReader, Instruction* maybeWriter) {
	if (auto call = dyn_cast<CallInst>(maybeWriter)) {
	  if (call->getCalledFunction() && isCertainMallocOrFree(call->getCalledFunction())) {
	    return false;
	  }
	}
	if (auto call = dyn_cast<CallInst>(maybeReader)) {
	  if (call->getCalledFunction() && isCertainMallocOrFree(call->getCalledFunction())) {
	    return false;
	  }
	}
	if (auto call = dyn_cast<InvokeInst>(maybeWriter)) {
	  if (call->getCalledFunction() && isCertainMallocOrFree(call->getCalledFunction())) {
	    return false;
	  }
	}
	if (auto call = dyn_cast<InvokeInst>(maybeReader)) {
	  if (call->getCalledFunction() && isCertainMallocOrFree(call->getCalledFunction())) {
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

	if (auto si = dyn_cast<StoreInst>(maybeWriter)) {
	  return isRefSet(AA.getModRefInfo(maybeReader, MemoryLocation::get(si)));
	}
	if (auto rmw = dyn_cast<AtomicRMWInst>(maybeWriter)) {
	  return isRefSet(AA.getModRefInfo(maybeReader, MemoryLocation::get(rmw)));
	}
	if (auto xch = dyn_cast<AtomicCmpXchgInst>(maybeWriter)) {
	  return isRefSet(AA.getModRefInfo(maybeReader, MemoryLocation::get(xch)));
	}

	if (auto cb = dyn_cast<CallInst>(maybeReader)) {
	  //llvm::errs() << " considering: " << *cb << " and: " << *maybeWriter << "\n";
	  return isModOrRefSet(AA.getModRefInfo(maybeWriter, cb));
	}
	if (auto cb = dyn_cast<InvokeInst>(maybeReader)) {
	  return isModOrRefSet(AA.getModRefInfo(maybeWriter, cb));
	}
	llvm::errs() << " maybeReader: " << *maybeReader << " maybeWriter: " << *maybeWriter << "\n";
	llvm_unreachable("unknown inst2");
}

#endif
