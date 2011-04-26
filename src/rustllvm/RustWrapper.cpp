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

#include "llvm/PassManager.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetSelect.h"
#include "llvm/Target/TargetRegistry.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm-c/Core.h"
#include "llvm-c/Object.h"
#include <cstdlib>

using namespace llvm;

static char *LLVMRustError;

extern "C" LLVMMemoryBufferRef
LLVMRustCreateMemoryBufferWithContentsOfFile(const char *Path) {
  LLVMMemoryBufferRef MemBuf = NULL;
  LLVMCreateMemoryBufferWithContentsOfFile(Path, &MemBuf, &LLVMRustError);
  return MemBuf;
}

extern "C" const char *LLVMRustGetLastError(void) {
  return LLVMRustError;
}

extern "C" void LLVMAddBasicAliasAnalysisPass(LLVMPassManagerRef PM);

void (*RustHackToFetchPassesO)(LLVMPassManagerRef PM) =
  LLVMAddBasicAliasAnalysisPass;

enum LLVMCodeGenFileType {
  LLVMAssemblyFile,
  LLVMObjectFile,
  LLVMNullFile         // Do not emit any output.
};

extern "C" void LLVMRustWriteOutputFile(LLVMPassManagerRef PMR, LLVMModuleRef M,
                                        const char *triple, const char *path,
                                        LLVMCodeGenFileType FileType) {

  // Set compilation options.
  llvm::NoFramePointerElim = true;

  InitializeAllTargets();
  InitializeAllAsmPrinters();
  InitializeAllAsmParsers();
  TargetMachine::setRelocationModel(Reloc::PIC_);
  std::string Err;
  const Target *TheTarget = TargetRegistry::lookupTarget(triple, Err);
  std::string FeaturesStr;
  TargetMachine &Target = *TheTarget->createTargetMachine(triple, FeaturesStr);
  bool NoVerify = false;
  CodeGenOpt::Level OLvl = CodeGenOpt::Default;
  PassManager *PM = unwrap<PassManager>(PMR);
  std::string ErrorInfo;
  raw_fd_ostream OS(path, ErrorInfo,
                    raw_fd_ostream::F_Binary);
  formatted_raw_ostream FOS(OS);
  TargetMachine::CodeGenFileType FileType2 =
    static_cast<TargetMachine::CodeGenFileType>(FileType);

  bool foo = Target.addPassesToEmitFile(*PM, FOS, FileType2, OLvl, NoVerify);
  assert(!foo);
  (void)foo;
  PM->run(*unwrap(M));
}
