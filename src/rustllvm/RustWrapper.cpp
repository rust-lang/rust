//===- RustWrapper.cpp - Rust wrapper for core functions --------*- C++ -*-===
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===
//
// This file defines alternate interfaces to core functions that are more
// readily callable by Rust's FFI.
//
//===----------------------------------------------------------------------===

#include "llvm/LLVMContext.h"
#include "llvm/Linker.h"
#include "llvm/PassManager.h"
#include "llvm/ADT/Triple.h"
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
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/JIT.h"
#include "llvm/ExecutionEngine/Interpreter.h"
#include "llvm/ExecutionEngine/GenericValue.h"
#include "llvm-c/Core.h"
#include "llvm-c/BitReader.h"
#include "llvm-c/Object.h"
#include <cstdlib>

using namespace llvm;

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

// Only initialize the platforms supported by Rust here,
// because using --llvm-root will have multiple platforms
// that rustllvm doesn't actually link to and it's pointless to put target info
// into the registry that Rust can not generate machine code for.

#define INITIALIZE_TARGETS() LLVMInitializeX86TargetInfo(); \
                             LLVMInitializeX86Target(); \
                             LLVMInitializeX86TargetMC(); \
                             LLVMInitializeX86AsmPrinter(); \
                             LLVMInitializeX86AsmParser();

extern "C" bool
LLVMRustJIT(LLVMPassManagerRef PMR,
            LLVMModuleRef M,
            CodeGenOpt::Level OptLevel,
            bool EnableSegmentedStacks) {

  INITIALIZE_TARGETS();
  InitializeNativeTarget();
  InitializeNativeTargetAsmPrinter();

  std::string Err;
  TargetOptions Options;
  Options.NoFramePointerElim = true;
  Options.EnableSegmentedStacks = EnableSegmentedStacks;

  PassManager *PM = unwrap<PassManager>(PMR);

  PM->run(*unwrap(M));

  ExecutionEngine* EE = EngineBuilder(unwrap(M))
    .setTargetOptions(Options)
    .setOptLevel(OptLevel)
    .setUseMCJIT(true)
    .create();

  if(!EE || Err != "") {
    LLVMRustError = Err.c_str();
    return false;
  }

  Function* func = EE->FindFunctionNamed("main");

  if(!func || Err != "") {
    LLVMRustError = Err.c_str();
    return false;
  }

  std::vector<GenericValue> args;

  EE->runFunction(func, args);

  return true;
}

extern "C" bool
LLVMRustWriteOutputFile(LLVMPassManagerRef PMR,
                        LLVMModuleRef M,
                        const char *triple,
                        const char *path,
                        TargetMachine::CodeGenFileType FileType,
                        CodeGenOpt::Level OptLevel,
			bool EnableSegmentedStacks) {

  INITIALIZE_TARGETS();

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
  DebugFlag = Enabled;
}
