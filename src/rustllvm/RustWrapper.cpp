// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#include "rustllvm.h"

//===----------------------------------------------------------------------===
//
// This file defines alternate interfaces to core functions that are more
// readily callable by Rust's FFI.
//
//===----------------------------------------------------------------------===

using namespace llvm;
using namespace llvm::sys;

static const char *LLVMRustError;

extern cl::opt<bool> EnableARMEHABI;

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

void LLVMInitializeMipsTargetInfo();
void LLVMInitializeMipsTarget();
void LLVMInitializeMipsTargetMC();
void LLVMInitializeMipsAsmPrinter();
void LLVMInitializeMipsAsmParser();
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

  LLVMInitializeMipsTargetInfo();
  LLVMInitializeMipsTarget();
  LLVMInitializeMipsTargetMC();
  LLVMInitializeMipsAsmPrinter();
  LLVMInitializeMipsAsmParser();
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
                                       unsigned SectionID, bool isReadOnly);
  bool finalizeMemory(std::string *ErrMsg) { return false; }

  virtual bool applyPermissions(std::string *Str);

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
                                                     unsigned SectionID,
                                                     bool isReadOnly) {
  if (!Alignment)
    Alignment = 16;
  uint8_t *Addr = (uint8_t*)calloc((Size + Alignment - 1)/Alignment, Alignment);
  AllocatedDataMem.push_back(sys::MemoryBlock(Addr, Size));
  return Addr;
}

bool RustMCJITMemoryManager::applyPermissions(std::string *Str) {
    // Empty.
    return true;
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

extern "C" LLVMExecutionEngineRef
LLVMRustBuildJIT(void* mem,
                 LLVMModuleRef M,
                 bool EnableSegmentedStacks) {

  InitializeNativeTarget();
  InitializeNativeTargetAsmPrinter();
  InitializeNativeTargetAsmParser();

  std::string Err;
  TargetOptions Options;
  Options.JITEmitDebugInfo = true;
  Options.NoFramePointerElim = true;
  Options.EnableSegmentedStacks = EnableSegmentedStacks;
  RustMCJITMemoryManager* MM = (RustMCJITMemoryManager*) mem;
  assert(MM);

  ExecutionEngine* EE = EngineBuilder(unwrap(M))
    .setErrorStr(&Err)
    .setTargetOptions(Options)
    .setJITMemoryManager(MM)
    .setUseMCJIT(true)
    .setAllocateGVsWithCode(false)
    .create();

  if(!EE || Err != "") {
    LLVMRustError = Err.c_str();
    // The EngineBuilder only takes ownership of these two structures if the
    // create() call is successful, but here it wasn't successful.
    LLVMDisposeModule(M);
    delete MM;
    return NULL;
  }

  MM->invalidateInstructionCache();
  return wrap(EE);
}

extern "C" bool
LLVMRustWriteOutputFile(LLVMPassManagerRef PMR,
                        LLVMModuleRef M,
                        const char *triple,
                        const char *cpu,
                        const char *feature,
                        const char *path,
                        TargetMachine::CodeGenFileType FileType,
                        CodeGenOpt::Level OptLevel,
      bool EnableSegmentedStacks) {

  LLVMRustInitializeTargets();

  // Initializing the command-line options more than once is not
  // allowed. So, check if they've already been initialized.
  // (This could happen if we're being called from rustpkg, for
  // example.)
  if (!EnableARMEHABI) {
    int argc = 3;
    const char* argv[] = {"rustc", "-arm-enable-ehabi",
        "-arm-enable-ehabi-descriptors"};
    cl::ParseCommandLineOptions(argc, argv);
  }

  TargetOptions Options;
  Options.NoFramePointerElim = true;
  Options.EnableSegmentedStacks = EnableSegmentedStacks;
  Options.FixedStackSegmentSize = 2 * 1024 * 1024;  // XXX: This is too big.

  PassManager *PM = unwrap<PassManager>(PMR);

  std::string Err;
  std::string Trip(Triple::normalize(triple));
  std::string FeaturesStr(feature);
  std::string CPUStr(cpu);
  const Target *TheTarget = TargetRegistry::lookupTarget(Trip, Err);
  TargetMachine *Target =
    TheTarget->createTargetMachine(Trip, CPUStr, FeaturesStr,
           Options, Reloc::PIC_,
           CodeModel::Default, OptLevel);
  Target->addAnalysisPasses(*PM);

  bool NoVerify = false;
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

extern "C" LLVMModuleRef LLVMRustParseAssemblyFile(LLVMContextRef C,
                                                   const char *Filename) {
  SMDiagnostic d;
  Module *m = ParseAssemblyFile(Filename, d, *unwrap(C));
  if (m) {
    return wrap(m);
  } else {
    LLVMRustError = d.getMessage().str().c_str();
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

extern "C" LLVMValueRef LLVMBuildAtomicLoad(LLVMBuilderRef B,
                                            LLVMValueRef source,
                                            const char* Name,
                                            AtomicOrdering order,
                                            unsigned alignment) {
    LoadInst* li = new LoadInst(unwrap(source),0);
    li->setVolatile(true);
    li->setAtomic(order);
    li->setAlignment(alignment);
    return wrap(unwrap(B)->Insert(li, Name));
}

extern "C" LLVMValueRef LLVMBuildAtomicStore(LLVMBuilderRef B,
                                             LLVMValueRef val,
                                             LLVMValueRef target,
                                             AtomicOrdering order,
                                             unsigned alignment) {
    StoreInst* si = new StoreInst(unwrap(val),unwrap(target));
    si->setVolatile(true);
    si->setAtomic(order);
    si->setAlignment(alignment);
    return wrap(unwrap(B)->Insert(si));
}

extern "C" LLVMValueRef LLVMBuildAtomicCmpXchg(LLVMBuilderRef B,
                                               LLVMValueRef target,
                                               LLVMValueRef old,
                                               LLVMValueRef source,
                                               AtomicOrdering order) {
    return wrap(unwrap(B)->CreateAtomicCmpXchg(unwrap(target), unwrap(old),
                                               unwrap(source), order));
}
extern "C" LLVMValueRef LLVMBuildAtomicFence(LLVMBuilderRef B, AtomicOrdering order) {
    return wrap(unwrap(B)->CreateFence(order));
}

extern "C" void LLVMSetDebug(int Enabled) {
#ifndef NDEBUG
  DebugFlag = Enabled;
#endif
}

extern "C" LLVMValueRef LLVMInlineAsm(LLVMTypeRef Ty,
                                      char *AsmString,
                                      char *Constraints,
                                      LLVMBool HasSideEffects,
                                      LLVMBool IsAlignStack,
                                      unsigned Dialect) {
    return wrap(InlineAsm::get(unwrap<FunctionType>(Ty), AsmString,
                               Constraints, HasSideEffects,
                               IsAlignStack, (InlineAsm::AsmDialect) Dialect));
}

/**
 * This function is intended to be a threadsafe interface into enabling a
 * multithreaded LLVM. This is invoked at the start of the translation phase of
 * compilation to ensure that LLVM is ready.
 *
 * All of trans properly isolates LLVM with the use of a different
 * LLVMContextRef per task, thus allowing parallel compilation of different
 * crates in the same process. At the time of this writing, the use case for
 * this is unit tests for rusti, but there are possible other applications.
 */
extern "C" bool LLVMRustStartMultithreading() {
    static Mutex lock;
    bool ret = true;
    assert(lock.acquire());
    if (!LLVMIsMultithreaded()) {
        ret = LLVMStartMultithreaded();
    }
    assert(lock.release());
    return ret;
}


typedef DIBuilder* DIBuilderRef;

template<typename DIT>
DIT unwrapDI(LLVMValueRef ref) {
    return DIT(ref ? unwrap<MDNode>(ref) : NULL);
}

extern "C" DIBuilderRef LLVMDIBuilderCreate(LLVMModuleRef M) {
    return new DIBuilder(*unwrap(M));
}

extern "C" void LLVMDIBuilderDispose(DIBuilderRef Builder) {
    delete Builder;
}

extern "C" void LLVMDIBuilderFinalize(DIBuilderRef Builder) {
    Builder->finalize();
}

extern "C" void LLVMDIBuilderCreateCompileUnit(
    DIBuilderRef Builder,
    unsigned Lang,
    const char* File,
    const char* Dir,
    const char* Producer,
    bool isOptimized,
    const char* Flags,
    unsigned RuntimeVer,
    const char* SplitName) {
    Builder->createCompileUnit(Lang, File, Dir, Producer, isOptimized,
        Flags, RuntimeVer, SplitName);
}

extern "C" LLVMValueRef LLVMDIBuilderCreateFile(
    DIBuilderRef Builder,
    const char* Filename,
    const char* Directory) {
    return wrap(Builder->createFile(Filename, Directory));
}

extern "C" LLVMValueRef LLVMDIBuilderCreateSubroutineType(
    DIBuilderRef Builder,
    LLVMValueRef File,
    LLVMValueRef ParameterTypes) {
    return wrap(Builder->createSubroutineType(
        unwrapDI<DIFile>(File),
        unwrapDI<DIArray>(ParameterTypes)));
}

extern "C" LLVMValueRef LLVMDIBuilderCreateFunction(
    DIBuilderRef Builder,
    LLVMValueRef Scope,
    const char* Name,
    const char* LinkageName,
    LLVMValueRef File,
    unsigned LineNo,
    LLVMValueRef Ty,
    bool isLocalToUnit,
    bool isDefinition,
    unsigned ScopeLine,
    unsigned Flags,
    bool isOptimized,
    LLVMValueRef Fn,
    LLVMValueRef TParam,
    LLVMValueRef Decl) {
    return wrap(Builder->createFunction(
        unwrapDI<DIScope>(Scope), Name, LinkageName,
        unwrapDI<DIFile>(File), LineNo,
        unwrapDI<DIType>(Ty), isLocalToUnit, isDefinition, ScopeLine,
        Flags, isOptimized,
        unwrap<Function>(Fn),
        unwrapDI<MDNode*>(TParam),
        unwrapDI<MDNode*>(Decl)));
}

extern "C" LLVMValueRef LLVMDIBuilderCreateBasicType(
    DIBuilderRef Builder,
    const char* Name,
    uint64_t SizeInBits,
    uint64_t AlignInBits,
    unsigned Encoding) {
    return wrap(Builder->createBasicType(
        Name, SizeInBits,
        AlignInBits, Encoding));
}

extern "C" LLVMValueRef LLVMDIBuilderCreatePointerType(
    DIBuilderRef Builder,
    LLVMValueRef PointeeTy,
    uint64_t SizeInBits,
    uint64_t AlignInBits,
    const char* Name) {
    return wrap(Builder->createPointerType(
        unwrapDI<DIType>(PointeeTy), SizeInBits, AlignInBits, Name));
}

extern "C" LLVMValueRef LLVMDIBuilderCreateStructType(
    DIBuilderRef Builder,
    LLVMValueRef Scope,
    const char* Name,
    LLVMValueRef File,
    unsigned LineNumber,
    uint64_t SizeInBits,
    uint64_t AlignInBits,
    unsigned Flags,
    LLVMValueRef DerivedFrom,
    LLVMValueRef Elements,
    unsigned RunTimeLang,
    LLVMValueRef VTableHolder) {
    return wrap(Builder->createStructType(
        unwrapDI<DIDescriptor>(Scope), Name,
        unwrapDI<DIFile>(File), LineNumber,
        SizeInBits, AlignInBits, Flags,
        unwrapDI<DIType>(DerivedFrom),
        unwrapDI<DIArray>(Elements), RunTimeLang,
        unwrapDI<MDNode*>(VTableHolder)));
}

extern "C" LLVMValueRef LLVMDIBuilderCreateMemberType(
    DIBuilderRef Builder,
    LLVMValueRef Scope,
    const char* Name,
    LLVMValueRef File,
    unsigned LineNo,
    uint64_t SizeInBits,
    uint64_t AlignInBits,
    uint64_t OffsetInBits,
    unsigned Flags,
    LLVMValueRef Ty) {
    return wrap(Builder->createMemberType(
        unwrapDI<DIDescriptor>(Scope), Name,
        unwrapDI<DIFile>(File), LineNo,
        SizeInBits, AlignInBits, OffsetInBits, Flags,
        unwrapDI<DIType>(Ty)));
}

extern "C" LLVMValueRef LLVMDIBuilderCreateLexicalBlock(
    DIBuilderRef Builder,
    LLVMValueRef Scope,
    LLVMValueRef File,
    unsigned Line,
    unsigned Col) {
    return wrap(Builder->createLexicalBlock(
        unwrapDI<DIDescriptor>(Scope),
        unwrapDI<DIFile>(File), Line, Col));
}

extern "C" LLVMValueRef LLVMDIBuilderCreateLocalVariable(
    DIBuilderRef Builder,
    unsigned Tag,
    LLVMValueRef Scope,
    const char* Name,
    LLVMValueRef File,
    unsigned LineNo,
    LLVMValueRef Ty,
    bool AlwaysPreserve,
    unsigned Flags,
    unsigned ArgNo) {
    return wrap(Builder->createLocalVariable(Tag,
        unwrapDI<DIDescriptor>(Scope), Name,
        unwrapDI<DIFile>(File),
        LineNo,
        unwrapDI<DIType>(Ty), AlwaysPreserve, Flags, ArgNo));
}

extern "C" LLVMValueRef LLVMDIBuilderCreateArrayType(
    DIBuilderRef Builder,
    uint64_t Size,
    uint64_t AlignInBits,
    LLVMValueRef Ty,
    LLVMValueRef Subscripts) {
    return wrap(Builder->createArrayType(Size, AlignInBits,
        unwrapDI<DIType>(Ty),
        unwrapDI<DIArray>(Subscripts)));
}

extern "C" LLVMValueRef LLVMDIBuilderCreateVectorType(
    DIBuilderRef Builder,
    uint64_t Size,
    uint64_t AlignInBits,
    LLVMValueRef Ty,
    LLVMValueRef Subscripts) {
    return wrap(Builder->createVectorType(Size, AlignInBits,
        unwrapDI<DIType>(Ty),
        unwrapDI<DIArray>(Subscripts)));
}

extern "C" LLVMValueRef LLVMDIBuilderGetOrCreateSubrange(
    DIBuilderRef Builder,
    int64_t Lo,
    int64_t Count) {
    return wrap(Builder->getOrCreateSubrange(Lo, Count));
}

extern "C" LLVMValueRef LLVMDIBuilderGetOrCreateArray(
    DIBuilderRef Builder,
    LLVMValueRef* Ptr,
    unsigned Count) {
    return wrap(Builder->getOrCreateArray(
        ArrayRef<Value*>(reinterpret_cast<Value**>(Ptr), Count)));
}

extern "C" LLVMValueRef LLVMDIBuilderInsertDeclareAtEnd(
    DIBuilderRef Builder,
    LLVMValueRef Val,
    LLVMValueRef VarInfo,
    LLVMBasicBlockRef InsertAtEnd) {
    return wrap(Builder->insertDeclare(
        unwrap(Val),
        unwrapDI<DIVariable>(VarInfo),
        unwrap(InsertAtEnd)));
}

extern "C" LLVMValueRef LLVMDIBuilderInsertDeclareBefore(
    DIBuilderRef Builder,
    LLVMValueRef Val,
    LLVMValueRef VarInfo,
    LLVMValueRef InsertBefore) {
    return wrap(Builder->insertDeclare(
        unwrap(Val),
        unwrapDI<DIVariable>(VarInfo),
        unwrap<Instruction>(InsertBefore)));
}

extern "C" LLVMValueRef LLVMDIBuilderCreateEnumerator(
    DIBuilderRef Builder,
    const char* Name,
    uint64_t Val)
{
    return wrap(Builder->createEnumerator(Name, Val));
}

extern "C" LLVMValueRef LLVMDIBuilderCreateEnumerationType(
    DIBuilderRef Builder,
    LLVMValueRef Scope,
    const char* Name,
    LLVMValueRef File,
    unsigned LineNumber,
    uint64_t SizeInBits,
    uint64_t AlignInBits,
    LLVMValueRef Elements,
    LLVMValueRef ClassType)
{
    return wrap(Builder->createEnumerationType(
        unwrapDI<DIDescriptor>(Scope),
        Name,
        unwrapDI<DIFile>(File),
        LineNumber,
        SizeInBits,
        AlignInBits,
        unwrapDI<DIArray>(Elements),
        unwrapDI<DIType>(ClassType)));
}

extern "C" LLVMValueRef LLVMDIBuilderCreateUnionType(
    DIBuilderRef Builder,
    LLVMValueRef Scope,
    const char* Name,
    LLVMValueRef File,
    unsigned LineNumber,
    uint64_t SizeInBits,
    uint64_t AlignInBits,
    unsigned Flags,
    LLVMValueRef Elements,
    unsigned RunTimeLang)
{
    return wrap(Builder->createUnionType(
        unwrapDI<DIDescriptor>(Scope),
        Name,
        unwrapDI<DIFile>(File),
        LineNumber,
        SizeInBits,
        AlignInBits,
        Flags,
        unwrapDI<DIArray>(Elements),
        RunTimeLang));
}

extern "C" void LLVMSetUnnamedAddr(LLVMValueRef Value, LLVMBool Unnamed) {
    unwrap<GlobalValue>(Value)->setUnnamedAddr(Unnamed);
}
