//===- RustWrapper.cpp - Rust wrapper for core functions --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines alternate interfaces to core functions that are more
// readily callable by Rust's FFI.
//
//===----------------------------------------------------------------------===//

#include "llvm/Linker.h"
#include "llvm/PassManager.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetSelect.h"
#include "llvm/Target/TargetRegistry.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/Host.h"
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
extern "C" void LLVMAddStandardModulePasses(LLVMPassManagerRef PM,
    unsigned int OptimizationLevel, bool OptimizeSize, bool UnitAtATime,
    bool UnrollLoops, bool SimplifyLibCalls,
    unsigned int InliningThreshold);

int *RustHackToFetchPassesO = (int*)LLVMAddBasicAliasAnalysisPass;
int *RustHackToFetchPasses2O = (int*)LLVMAddStandardModulePasses;


extern "C" bool LLVMLinkModules(LLVMModuleRef Dest, LLVMModuleRef Src) {
  static std::string err;

  // For some strange reason, unwrap() doesn't work here. "No matching
  // function" error.
  Module *DM = reinterpret_cast<Module *>(Dest);
  Module *SM = reinterpret_cast<Module *>(Src);
  if (Linker::LinkModules(DM, SM, &err)) {
    LLVMRustError = err.c_str();
    return false;
  }
  return true;
}

extern "C" void LLVMRustWriteOutputFile(LLVMPassManagerRef PMR,
                                        LLVMModuleRef M,
                                        const char *triple,
                                        const char *path,
                                        TargetMachine::CodeGenFileType FileType,
                                        CodeGenOpt::Level OptLevel) {

  // Set compilation options.
  llvm::NoFramePointerElim = true;

  InitializeAllTargets();
  InitializeAllAsmPrinters();
  InitializeAllAsmParsers();
  TargetMachine::setRelocationModel(Reloc::PIC_);
  std::string Err;
  const Target *TheTarget = TargetRegistry::lookupTarget(triple, Err);
  std::string FeaturesStr;
  std::string Trip(triple);
  std::string CPUStr = llvm::sys::getHostCPUName();
  TargetMachine *Target = TheTarget->createTargetMachine(Trip, CPUStr, FeaturesStr);
  bool NoVerify = false;
  PassManager *PM = unwrap<PassManager>(PMR);
  std::string ErrorInfo;
  raw_fd_ostream OS(path, ErrorInfo,
                    raw_fd_ostream::F_Binary);
  formatted_raw_ostream FOS(OS);

  bool foo = Target->addPassesToEmitFile(*PM, FOS, FileType, OptLevel,
                                         NoVerify);
  assert(!foo);
  (void)foo;
  PM->run(*unwrap(M));
  delete Target;
}

extern "C" LLVMModuleRef LLVMRustParseBitcode(LLVMMemoryBufferRef MemBuf) {
  LLVMModuleRef M;
  return LLVMParseBitcode(MemBuf, &M, const_cast<char **>(&LLVMRustError))
         ? NULL : M;
}

extern "C" const char *LLVMRustGetHostTriple(void)
{
  static std::string str = llvm::sys::getHostTriple();
  return str.c_str();
}

extern "C" LLVMValueRef LLVMRustConstSmallInt(LLVMTypeRef IntTy, unsigned N,
                                              LLVMBool SignExtend) {
  return LLVMConstInt(IntTy, (unsigned long long)N, SignExtend);
}

extern bool llvm::TimePassesIsEnabled;
extern "C" void LLVMRustEnableTimePasses() {
  TimePassesIsEnabled = true;
}

extern "C" void LLVMRustPrintPassTimings() {
  raw_fd_ostream OS (2, false); // stderr.
  TimerGroup::printAll(OS);
}
