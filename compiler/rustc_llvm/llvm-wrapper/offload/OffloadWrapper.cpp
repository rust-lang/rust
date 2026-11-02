#include "../SuppressLLVMWarnings.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Frontend/Offloading/OffloadWrapper.h"
#include "llvm/Frontend/Offloading/Utility.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileOutputBuffer.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

#include <optional>
#include <string>
#include <system_error>
#include <utility>

using namespace llvm;

static Error writeFile(StringRef Filename, StringRef Data) {
  Expected<std::unique_ptr<FileOutputBuffer>> OutputOrErr =
      FileOutputBuffer::create(Filename, Data.size());
  if (!OutputOrErr)
    return OutputOrErr.takeError();
  std::unique_ptr<FileOutputBuffer> Output = std::move(*OutputOrErr);
  llvm::copy(Data, Output->getBufferStart());
  if (Error E = Output->commit())
    return E;
  return Error::success();
}

// Clone OldFn into NewFn, remapping its arguments to RebuiltArgs.
// Each arg of OldFn is replaced with the corresponding value in RebuiltArgs.
// For scalars, RebuiltArgs contains the value cast and/or truncated to the
// original type.
extern "C" void LLVMRustOffloadMapper(LLVMValueRef OldFn, LLVMValueRef NewFn,
                                      const LLVMValueRef *RebuiltArgs) {
  llvm::Function *oldFn = llvm::unwrap<llvm::Function>(OldFn);
  llvm::Function *newFn = llvm::unwrap<llvm::Function>(NewFn);

  // Map old arguments to new arguments. We skip the first dyn_ptr argument,
  // since it can't be used directly by user code.
  llvm::ValueToValueMapTy vmap;
  auto newArgIt = newFn->arg_begin();
  newArgIt->setName("dyn_ptr");

  unsigned i = 0;
  for (auto &oldArg : oldFn->args()) {
    vmap[&oldArg] = unwrap<Value>(RebuiltArgs[i++]);
  }

  llvm::SmallVector<llvm::ReturnInst *, 8> returns;
  llvm::CloneFunctionInto(newFn, oldFn, vmap,
                          llvm::CloneFunctionChangeType::LocalChangesOnly,
                          returns);

  BasicBlock &entry = newFn->getEntryBlock();
  BasicBlock &clonedEntry = *std::next(newFn->begin());

  if (entry.getTerminator())
    entry.getTerminator()->eraseFromParent();

  IRBuilder<> B(&entry);
  B.CreateBr(&clonedEntry);
}

static bool hasOffloadEntries(Module &M) {
  for (GlobalVariable &GV : M.globals())
    if (GV.hasSection() && GV.getSection() == "llvm_offload_entries")
      return true;
  return false;
}

static bool reportAndFailWrappingImages(Error E, const char *What) {
  handleAllErrors(std::move(E), [&](const ErrorInfoBase &EI) {
    errs() << "LLVMRustOffloadWrapImages: " << What << ": " << EI.message()
           << "\n";
  });
  return false;
}

static Expected<std::unique_ptr<MemoryBuffer>>
assembleWithPtxas(StringRef Ptx, StringRef Arch) {
  const ErrorOr<std::string> Ptxas = sys::findProgramByName("ptxas");
  if (!Ptxas)
    return createStringError(Ptxas.getError(), "ptxas not found in PATH");

  SmallString<128> PtxFilePath;
  if (std::error_code E =
          sys::fs::createTemporaryFile("rust-offload", "ptx", PtxFilePath))
    return errorCodeToError(E);

  SmallString<128> CubinFilePath;
  if (std::error_code E =
          sys::fs::createTemporaryFile("rust-offload", "cubin", CubinFilePath))
    return errorCodeToError(E);

  auto Cleanup = scope_exit([&] {
    if (std::error_code E = sys::fs::remove(PtxFilePath))
      (void)reportAndFailWrappingImages(
          errorCodeToError(E), "assembleWithPtxas: PtxFilePath cleanup");
    if (std::error_code E = sys::fs::remove(CubinFilePath))
      (void)reportAndFailWrappingImages(
          errorCodeToError(E), "assembleWithPtxas: CubinFilePath cleanup");
  });

  if (Error E = writeFile(PtxFilePath, Ptx))
    return std::move(E);

  const StringRef Args[] = {
      *Ptxas, "-m64",          "-O3",         "--gpu-name",
      Arch,   "--output-file", CubinFilePath, PtxFilePath,
  };

  std::string ErrorMsg;
  const int Status =
      sys::ExecuteAndWait(*Ptxas, Args, std::nullopt, {}, 0, 0, &ErrorMsg);

  if (Status != 0)
    return createStringError("assembleWithPtxas: status %d: %s", Status,
                             ErrorMsg.c_str());

  ErrorOr<std::unique_ptr<MemoryBuffer>> CubinOrError =
      MemoryBuffer::getFileAsStream(CubinFilePath);
  if (!CubinOrError)
    return errorCodeToError(CubinOrError.getError());

  return std::move(*CubinOrError);
}

static Expected<std::unique_ptr<MemoryBuffer>>
linkWithRustLld(StringRef Obj, StringRef LldPath) {
  SmallString<128> ObjFilePath;
  if (std::error_code E =
          sys::fs::createTemporaryFile("rust-offload", "o", ObjFilePath))
    return errorCodeToError(E);

  SmallString<128> SoFilePath;
  if (std::error_code E =
          sys::fs::createTemporaryFile("rust-offload", "so", SoFilePath))
    return errorCodeToError(E);

  auto Cleanup = scope_exit([&] {
    if (std::error_code E = sys::fs::remove(ObjFilePath))
      (void)reportAndFailWrappingImages(errorCodeToError(E),
                                        "linkWithRustLld: ObjFilePath cleanup");
    if (std::error_code E = sys::fs::remove(SoFilePath))
      (void)reportAndFailWrappingImages(errorCodeToError(E),
                                        "linkWithRustLld: SoFilePath cleanup");
  });

  if (Error E = writeFile(ObjFilePath, Obj))
    return std::move(E);

  const StringRef Args[] = {
      LldPath,          "-flavor", "gnu",      "-shared",
      "--no-undefined", "-o",      SoFilePath, ObjFilePath,
  };

  std::string ErrorMsg;
  const int Status =
      sys::ExecuteAndWait(LldPath, Args, std::nullopt, {}, 0, 0, &ErrorMsg);

  if (Status != 0)
    return createStringError("linkWithRustLld: status %d: %s", Status,
                             ErrorMsg.c_str());

  ErrorOr<std::unique_ptr<MemoryBuffer>> ElfOrError =
      MemoryBuffer::getFileAsStream(SoFilePath);
  if (!ElfOrError)
    return errorCodeToError(ElfOrError.getError());

  return std::move(*ElfOrError);
}

static Expected<std::unique_ptr<MemoryBuffer>>
compileDeviceImage(MemoryBufferRef Input, const char *LldPath) {
  LLVMContext Ctx;
  Expected<std::unique_ptr<Module>> ModOrErr = parseBitcodeFile(Input, Ctx);
  if (!ModOrErr)
    return ModOrErr.takeError();
  Module &M = **ModOrErr;

  const Triple &DeviceTriple = M.getTargetTriple();
  StringRef Arch;
  for (Function &F : M)
    if (F.hasFnAttribute("target-cpu")) {
      Arch = F.getFnAttribute("target-cpu").getValueAsString();
      break;
    }

  if (Arch.empty())
    return createStringError("no target-cpu attribute in device bitcode");

  std::string ErrorMsg;
  const Target *DeviceTarget =
      TargetRegistry::lookupTarget(DeviceTriple, ErrorMsg);
  if (!DeviceTarget)
    return createStringError(ErrorMsg);

  std::unique_ptr<TargetMachine> TM(DeviceTarget->createTargetMachine(
      DeviceTriple, Arch, /*Features=*/"", TargetOptions(), Reloc::PIC_));
  if (!TM)
    return createStringError("createTargetMachine failed for %s",
                             DeviceTriple.str().c_str());

  const bool IsNvptx = DeviceTriple.isNVPTX();

  legacy::PassManager PM;
  SmallString<0> Emitted;
  raw_svector_ostream OS(Emitted);
  const CodeGenFileType FileType =
      IsNvptx ? CodeGenFileType::AssemblyFile : CodeGenFileType::ObjectFile;
  if (TM->addPassesToEmitFile(PM, OS, nullptr, FileType))
    return createStringError("target %s cannot emit %s",
                             DeviceTriple.str().c_str(),
                             IsNvptx ? "assembly" : "object");

  PM.run(**ModOrErr);

  if (IsNvptx)
    return assembleWithPtxas(Emitted, Arch);
  if (DeviceTriple.isAMDGPU()) {
    if (!LldPath || !*LldPath)
      return createStringError("rust-lld path was not provided for %s",
                               DeviceTriple.str().c_str());
    return linkWithRustLld(Emitted, LldPath);
  }

  return createStringError("unsupported offload target %s",
                           DeviceTriple.str().c_str());
}

extern "C" bool LLVMRustOffloadWrapImages(LLVMModuleRef HostMRef,
                                          const char *LldPath,
                                          const char *DeviceBitcodePath) {
  Module &M = *unwrap(HostMRef);
  if (!hasOffloadEntries(M))
    return true;

  ErrorOr<std::unique_ptr<MemoryBuffer>> BufOrErr =
      MemoryBuffer::getFile(DeviceBitcodePath);
  if (std::error_code EC = BufOrErr.getError())
    return reportAndFailWrappingImages(createFileError(DeviceBitcodePath, EC),
                                       "device bitcode");
  std::unique_ptr<MemoryBuffer> Buf = std::move(*BufOrErr);

  Expected<std::unique_ptr<MemoryBuffer>> ImageOrErr =
      compileDeviceImage(Buf->getMemBufferRef(), LldPath);
  if (!ImageOrErr)
    return reportAndFailWrappingImages(ImageOrErr.takeError(),
                                       "device compile");

  StringRef ImageBuf = (*ImageOrErr)->getBuffer();
  ArrayRef<char> Image(ImageBuf.data(), ImageBuf.size());

  if (Error E = offloading::wrapOpenMPBinaries(
          M, {Image}, offloading::getOffloadEntryArray(M), /*Suffix=*/"",
          /*Relocatable=*/
          false))
    return reportAndFailWrappingImages(std::move(E), "wrap");
  return true;
}
