// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//===----------------------------------------------------------------------===
//
// This file defines alternate interfaces to core functions that are more
// readily callable by Rust's FFI.
//
//===----------------------------------------------------------------------===

#include "llvm/LLVMContext.h"
#include "llvm/Linker.h"
#include "llvm/PassManager.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/ADT/Triple.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Assembly/Parser.h"
#include "llvm/Assembly/PrintModulePass.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Memory.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/JIT.h"
#include "llvm/ExecutionEngine/JITMemoryManager.h"
#include "llvm/ExecutionEngine/MCJIT.h"
#include "llvm/ExecutionEngine/Interpreter.h"
#include "llvm-c/Core.h"
#include "llvm-c/BitReader.h"
#include "llvm-c/Object.h"

// Used by RustMCJITMemoryManager::getPointerToNamedFunction()
// to get around glibc issues. See the function for more information.
#ifdef __linux__
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

using namespace llvm;
using namespace llvm::sys;

static const char *LLVMRustError;

extern "C" LLVMMemoryBufferRef
LLVMRustCreateMemoryBufferWithContentsOfFile(const char *Path) {
  LLVMMemoryBufferRef MemBuf = NULL;
  LLVMCreateMemoryBufferWithContentsOfFile(Path, &MemBuf,
    const_cast<char **>(&LLVMRustError));
  return MemBuf;
}

extern "C" const char *LLVMRustGetLastError(void) {
  return LLVMRustError;
}

extern "C" void LLVMAddBasicAliasAnalysisPass(LLVMPassManagerRef PM);

extern "C" void LLVMRustAddPrintModulePass(LLVMPassManagerRef PMR,
                                           LLVMModuleRef M,
                                           const char* path) {
  PassManager *PM = unwrap<PassManager>(PMR);
  std::string ErrorInfo;
  raw_fd_ostream OS(path, ErrorInfo, raw_fd_ostream::F_Binary);
  formatted_raw_ostream FOS(OS);
  PM->add(createPrintModulePass(&FOS));
  PM->run(*unwrap(M));
}

void LLVMInitializeX86TargetInfo();
void LLVMInitializeX86Target();
void LLVMInitializeX86TargetMC();
void LLVMInitializeX86AsmPrinter();
void LLVMInitializeX86AsmParser();


void LLVMInitializeARMTargetInfo();
void LLVMInitializeARMTarget();
void LLVMInitializeARMTargetMC();
void LLVMInitializeARMAsmPrinter();
void LLVMInitializeARMAsmParser();
// Only initialize the platforms supported by Rust here,
// because using --llvm-root will have multiple platforms
// that rustllvm doesn't actually link to and it's pointless to put target info
// into the registry that Rust can not generate machine code for.

void LLVMRustInitializeTargets() {
  LLVMInitializeX86TargetInfo();
  LLVMInitializeX86Target();
  LLVMInitializeX86TargetMC();
  LLVMInitializeX86AsmPrinter();
  LLVMInitializeX86AsmParser();
	
  LLVMInitializeARMTargetInfo();
  LLVMInitializeARMTarget();
  LLVMInitializeARMTargetMC();
  LLVMInitializeARMAsmPrinter();
  LLVMInitializeARMAsmParser();	
}

// Custom memory manager for MCJITting. It needs special features
// that the generic JIT memory manager doesn't entail. Based on
// code from LLI, change where needed for Rust.
class RustMCJITMemoryManager : public JITMemoryManager {
public:
  SmallVector<sys::MemoryBlock, 16> AllocatedDataMem;
  SmallVector<sys::MemoryBlock, 16> AllocatedCodeMem;
  SmallVector<sys::MemoryBlock, 16> FreeCodeMem;
  void* __morestack;
  DenseSet<DynamicLibrary*> crates;

  RustMCJITMemoryManager(void* sym) : __morestack(sym) { }
  ~RustMCJITMemoryManager();

  bool loadCrate(const char*, std::string*);

  virtual uint8_t *allocateCodeSection(uintptr_t Size, unsigned Alignment,
                                       unsigned SectionID);

  virtual uint8_t *allocateDataSection(uintptr_t Size, unsigned Alignment,
                                       unsigned SectionID);

  virtual void *getPointerToNamedFunction(const std::string &Name,
                                          bool AbortOnFailure = true);

  // Invalidate instruction cache for code sections. Some platforms with
  // separate data cache and instruction cache require explicit cache flush,
  // otherwise JIT code manipulations (like resolved relocations) will get to
  // the data cache but not to the instruction cache.
  virtual void invalidateInstructionCache();

  // The MCJITMemoryManager doesn't use the following functions, so we don't
  // need implement them.
  virtual void setMemoryWritable() {
    llvm_unreachable("Unimplemented call");
  }
  virtual void setMemoryExecutable() {
    llvm_unreachable("Unimplemented call");
  }
  virtual void setPoisonMemory(bool poison) {
    llvm_unreachable("Unimplemented call");
  }
  virtual void AllocateGOT() {
    llvm_unreachable("Unimplemented call");
  }
  virtual uint8_t *getGOTBase() const {
    llvm_unreachable("Unimplemented call");
    return 0;
  }
  virtual uint8_t *startFunctionBody(const Function *F,
                                     uintptr_t &ActualSize){
    llvm_unreachable("Unimplemented call");
    return 0;
  }
  virtual uint8_t *allocateStub(const GlobalValue* F, unsigned StubSize,
                                unsigned Alignment) {
    llvm_unreachable("Unimplemented call");
    return 0;
  }
  virtual void endFunctionBody(const Function *F, uint8_t *FunctionStart,
                               uint8_t *FunctionEnd) {
    llvm_unreachable("Unimplemented call");
  }
  virtual uint8_t *allocateSpace(intptr_t Size, unsigned Alignment) {
    llvm_unreachable("Unimplemented call");
    return 0;
  }
  virtual uint8_t *allocateGlobal(uintptr_t Size, unsigned Alignment) {
    llvm_unreachable("Unimplemented call");
    return 0;
  }
  virtual void deallocateFunctionBody(void *Body) {
    llvm_unreachable("Unimplemented call");
  }
  virtual uint8_t* startExceptionTable(const Function* F,
                                       uintptr_t &ActualSize) {
    llvm_unreachable("Unimplemented call");
    return 0;
  }
  virtual void endExceptionTable(const Function *F, uint8_t *TableStart,
                                 uint8_t *TableEnd, uint8_t* FrameRegister) {
    llvm_unreachable("Unimplemented call");
  }
  virtual void deallocateExceptionTable(void *ET) {
    llvm_unreachable("Unimplemented call");
  }
};

bool RustMCJITMemoryManager::loadCrate(const char* file, std::string* err) {
  DynamicLibrary crate = DynamicLibrary::getPermanentLibrary(file,
                                                             err);

  if(crate.isValid()) {
    crates.insert(&crate);

    return true;
  }

  return false;
}

uint8_t *RustMCJITMemoryManager::allocateDataSection(uintptr_t Size,
                                                    unsigned Alignment,
                                                    unsigned SectionID) {
  if (!Alignment)
    Alignment = 16;
  uint8_t *Addr = (uint8_t*)calloc((Size + Alignment - 1)/Alignment, Alignment);
  AllocatedDataMem.push_back(sys::MemoryBlock(Addr, Size));
  return Addr;
}

uint8_t *RustMCJITMemoryManager::allocateCodeSection(uintptr_t Size,
                                                    unsigned Alignment,
                                                    unsigned SectionID) {
  if (!Alignment)
    Alignment = 16;
  unsigned NeedAllocate = Alignment * ((Size + Alignment - 1)/Alignment + 1);
  uintptr_t Addr = 0;
  // Look in the list of free code memory regions and use a block there if one
  // is available.
  for (int i = 0, e = FreeCodeMem.size(); i != e; ++i) {
    sys::MemoryBlock &MB = FreeCodeMem[i];
    if (MB.size() >= NeedAllocate) {
      Addr = (uintptr_t)MB.base();
      uintptr_t EndOfBlock = Addr + MB.size();
      // Align the address.
      Addr = (Addr + Alignment - 1) & ~(uintptr_t)(Alignment - 1);
      // Store cutted free memory block.
      FreeCodeMem[i] = sys::MemoryBlock((void*)(Addr + Size),
                                        EndOfBlock - Addr - Size);
      return (uint8_t*)Addr;
    }
  }

  // No pre-allocated free block was large enough. Allocate a new memory region.
  sys::MemoryBlock MB = sys::Memory::AllocateRWX(NeedAllocate, 0, 0);

  AllocatedCodeMem.push_back(MB);
  Addr = (uintptr_t)MB.base();
  uintptr_t EndOfBlock = Addr + MB.size();
  // Align the address.
  Addr = (Addr + Alignment - 1) & ~(uintptr_t)(Alignment - 1);
  // The AllocateRWX may allocate much more memory than we need. In this case,
  // we store the unused memory as a free memory block.
  unsigned FreeSize = EndOfBlock-Addr-Size;
  if (FreeSize > 16)
    FreeCodeMem.push_back(sys::MemoryBlock((void*)(Addr + Size), FreeSize));

  // Return aligned address
  return (uint8_t*)Addr;
}

void RustMCJITMemoryManager::invalidateInstructionCache() {
  for (int i = 0, e = AllocatedCodeMem.size(); i != e; ++i)
    sys::Memory::InvalidateInstructionCache(AllocatedCodeMem[i].base(),
                                            AllocatedCodeMem[i].size());
}

void *RustMCJITMemoryManager::getPointerToNamedFunction(const std::string &Name,
                                                       bool AbortOnFailure) {
#ifdef __linux__
  // Force the following functions to be linked in to anything that uses the
  // JIT. This is a hack designed to work around the all-too-clever Glibc
  // strategy of making these functions work differently when inlined vs. when
  // not inlined, and hiding their real definitions in a separate archive file
  // that the dynamic linker can't see. For more info, search for
  // 'libc_nonshared.a' on Google, or read http://llvm.org/PR274.
  if (Name == "stat") return (void*)(intptr_t)&stat;
  if (Name == "fstat") return (void*)(intptr_t)&fstat;
  if (Name == "lstat") return (void*)(intptr_t)&lstat;
  if (Name == "stat64") return (void*)(intptr_t)&stat64;
  if (Name == "fstat64") return (void*)(intptr_t)&fstat64;
  if (Name == "lstat64") return (void*)(intptr_t)&lstat64;
  if (Name == "atexit") return (void*)(intptr_t)&atexit;
  if (Name == "mknod") return (void*)(intptr_t)&mknod;
#endif

  if (Name == "__morestack" || Name == "___morestack") return &__morestack;

  const char *NameStr = Name.c_str();

  // Look through loaded crates and main for symbols.

  void *Ptr = sys::DynamicLibrary::SearchForAddressOfSymbol(NameStr);
  if (Ptr) return Ptr;

  // If it wasn't found and if it starts with an underscore ('_') character,
  // try again without the underscore.
  if (NameStr[0] == '_') {
    Ptr = sys::DynamicLibrary::SearchForAddressOfSymbol(NameStr+1);
    if (Ptr) return Ptr;
  }

  if (AbortOnFailure)
    report_fatal_error("Program used external function '" + Name +
                      "' which could not be resolved!");
  return 0;
}

RustMCJITMemoryManager::~RustMCJITMemoryManager() {
  for (unsigned i = 0, e = AllocatedCodeMem.size(); i != e; ++i)
    sys::Memory::ReleaseRWX(AllocatedCodeMem[i]);
  for (unsigned i = 0, e = AllocatedDataMem.size(); i != e; ++i)
    free(AllocatedDataMem[i].base());
}

extern "C" void*
LLVMRustPrepareJIT(void* __morestack) {
  // An execution engine will take ownership of this later
  // and clean it up for us.

  return (void*) new RustMCJITMemoryManager(__morestack);
}

extern "C" bool
LLVMRustLoadCrate(void* mem, const char* crate) {
  RustMCJITMemoryManager* manager = (RustMCJITMemoryManager*) mem;
  std::string Err;

  assert(manager);

  if(!manager->loadCrate(crate, &Err)) {
    LLVMRustError = Err.c_str();
    return false;
  }

  return true;
}

extern "C" void*
LLVMRustExecuteJIT(void* mem,
                   LLVMPassManagerRef PMR,
                   LLVMModuleRef M,
                   CodeGenOpt::Level OptLevel,
                   bool EnableSegmentedStacks) {

  InitializeNativeTarget();
  InitializeNativeTargetAsmPrinter();
  InitializeNativeTargetAsmParser();

  std::string Err;
  TargetOptions Options;
  Options.JITExceptionHandling = true;
  Options.JITEmitDebugInfo = true;
  Options.NoFramePointerElim = true;
  Options.EnableSegmentedStacks = EnableSegmentedStacks;
  PassManager *PM = unwrap<PassManager>(PMR);
  RustMCJITMemoryManager* MM = (RustMCJITMemoryManager*) mem;

  assert(MM);

  PM->add(createBasicAliasAnalysisPass());
  PM->add(createInstructionCombiningPass());
  PM->add(createReassociatePass());
  PM->add(createGVNPass());
  PM->add(createCFGSimplificationPass());
  PM->add(createFunctionInliningPass());
  PM->add(createPromoteMemoryToRegisterPass());
  PM->run(*unwrap(M));

  ExecutionEngine* EE = EngineBuilder(unwrap(M))
    .setErrorStr(&Err)
    .setTargetOptions(Options)
    .setJITMemoryManager(MM)
    .setOptLevel(OptLevel)
    .setUseMCJIT(true)
    .setAllocateGVsWithCode(false)
    .create();

  if(!EE || Err != "") {
    LLVMRustError = Err.c_str();
    return 0;
  }

  MM->invalidateInstructionCache();
  Function* func = EE->FindFunctionNamed("_rust_main");

  if(!func || Err != "") {
    LLVMRustError = Err.c_str();
    return 0;
  }

  void* entry = EE->getPointerToFunction(func);
  assert(entry);

  return entry;
}

extern "C" bool
LLVMRustWriteOutputFile(LLVMPassManagerRef PMR,
                        LLVMModuleRef M,
                        const char *triple,
                        const char *path,
                        TargetMachine::CodeGenFileType FileType,
                        CodeGenOpt::Level OptLevel,
			bool EnableSegmentedStacks) {

  LLVMRustInitializeTargets();

  TargetOptions Options;
  Options.NoFramePointerElim = true;
  Options.EnableSegmentedStacks = EnableSegmentedStacks;

  std::string Err;
  const Target *TheTarget = TargetRegistry::lookupTarget(triple, Err);
  std::string FeaturesStr;
  std::string Trip(triple);
  std::string CPUStr("generic");
  TargetMachine *Target =
    TheTarget->createTargetMachine(Trip, CPUStr, FeaturesStr,
				   Options, Reloc::PIC_,
				   CodeModel::Default, OptLevel);
  bool NoVerify = false;
  PassManager *PM = unwrap<PassManager>(PMR);
  std::string ErrorInfo;
  raw_fd_ostream OS(path, ErrorInfo,
                    raw_fd_ostream::F_Binary);
  if (ErrorInfo != "") {
    LLVMRustError = ErrorInfo.c_str();
    return false;
  }
  formatted_raw_ostream FOS(OS);

  bool foo = Target->addPassesToEmitFile(*PM, FOS, FileType, NoVerify);
  assert(!foo);
  (void)foo;
  PM->run(*unwrap(M));
  delete Target;
  return true;
}

extern "C" LLVMModuleRef LLVMRustParseAssemblyFile(const char *Filename) {

  SMDiagnostic d;
  Module *m = ParseAssemblyFile(Filename, d, getGlobalContext());
  if (m) {
    return wrap(m);
  } else {
    LLVMRustError = d.getMessage().c_str();
    return NULL;
  }
}

extern "C" LLVMModuleRef LLVMRustParseBitcode(LLVMMemoryBufferRef MemBuf) {
  LLVMModuleRef M;
  return LLVMParseBitcode(MemBuf, &M, const_cast<char **>(&LLVMRustError))
         ? NULL : M;
}

extern "C" LLVMValueRef LLVMRustConstSmallInt(LLVMTypeRef IntTy, unsigned N,
                                              LLVMBool SignExtend) {
  return LLVMConstInt(IntTy, (unsigned long long)N, SignExtend);
}

extern "C" LLVMValueRef LLVMRustConstInt(LLVMTypeRef IntTy, 
					 unsigned N_hi,
					 unsigned N_lo,
					 LLVMBool SignExtend) {
  unsigned long long N = N_hi;
  N <<= 32;
  N |= N_lo;
  return LLVMConstInt(IntTy, N, SignExtend);
}

extern bool llvm::TimePassesIsEnabled;
extern "C" void LLVMRustEnableTimePasses() {
  TimePassesIsEnabled = true;
}

extern "C" void LLVMRustPrintPassTimings() {
  raw_fd_ostream OS (2, false); // stderr.
  TimerGroup::printAll(OS);
}

extern "C" LLVMValueRef LLVMGetOrInsertFunction(LLVMModuleRef M,
                                                const char* Name,
                                                LLVMTypeRef FunctionTy) {
  return wrap(unwrap(M)->getOrInsertFunction(Name,
                                             unwrap<FunctionType>(FunctionTy)));
}

extern "C" LLVMTypeRef LLVMMetadataTypeInContext(LLVMContextRef C) {
  return wrap(Type::getMetadataTy(*unwrap(C)));
}
extern "C" LLVMTypeRef LLVMMetadataType(void) {
  return LLVMMetadataTypeInContext(LLVMGetGlobalContext());
}

extern "C" LLVMValueRef LLVMBuildAtomicCmpXchg(LLVMBuilderRef B,
                                               LLVMValueRef target,
                                               LLVMValueRef old,
                                               LLVMValueRef source,
                                               AtomicOrdering order) {
    return wrap(unwrap(B)->CreateAtomicCmpXchg(unwrap(target), unwrap(old),
                                               unwrap(source), order));
}
extern "C" LLVMValueRef LLVMBuildAtomicRMW(LLVMBuilderRef B,
                                           AtomicRMWInst::BinOp op,
                                           LLVMValueRef target,
                                           LLVMValueRef source,
                                           AtomicOrdering order) {
    return wrap(unwrap(B)->CreateAtomicRMW(op,
                                           unwrap(target), unwrap(source),
                                           order));
}

extern "C" void LLVMSetDebug(int Enabled) {
#ifndef NDEBUG
  DebugFlag = Enabled;
#endif
}
