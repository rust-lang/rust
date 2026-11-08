#include "../SuppressLLVMWarnings.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Frontend/Offloading/OffloadWrapper.h"
#include "llvm/Frontend/Offloading/Utility.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

#include <optional>
#include <string>
#include <system_error>
#include <utility>

using namespace llvm;

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

/// Returns the target architecture (e.g. "sm_86") named by the `.target`
/// directive of the given PTX text. The result points into `PtxBuf`.
static Expected<StringRef> getPtxArch(MemoryBufferRef PtxBuf) {
  const Regex TargetRe("^\\.target[[:space:]]+([^,[:space:]]+)",
                       Regex::Newline);

  SmallVector<StringRef, 2> M;
  if (!TargetRe.match(PtxBuf.getBuffer(), &M))
    return createStringError("no .target directive in %s",
                             PtxBuf.getBufferIdentifier().str().c_str());

  return M[1];
}

static Expected<std::unique_ptr<MemoryBuffer>>
assembleWithPtxas(MemoryBufferRef PtxBuf) {
  Expected<StringRef> ArchOrErr = getPtxArch(PtxBuf);
  if (!ArchOrErr)
    return ArchOrErr.takeError();

  const ErrorOr<std::string> Ptxas = sys::findProgramByName("ptxas");
  if (!Ptxas)
    return createStringError(Ptxas.getError(), "ptxas not found in PATH");

  SmallString<128> CubinFilePath;
  if (std::error_code E =
          sys::fs::createTemporaryFile("rust-offload", "cubin", CubinFilePath))
    return errorCodeToError(E);

  auto Cleanup = scope_exit([&] {
    if (std::error_code E = sys::fs::remove(CubinFilePath))
      (void)reportAndFailWrappingImages(
          errorCodeToError(E), "assembleWithPtxas: CubinFilePath cleanup");
  });

  const StringRef Args[] = {
      *Ptxas,     "-m64",          "-O3",         "--gpu-name",
      *ArchOrErr, "--output-file", CubinFilePath, PtxBuf.getBufferIdentifier(),
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

extern "C" bool LLVMRustOffloadWrapImages(LLVMModuleRef HostMRef,
                                          const char *DeviceImagePath) {
  Module &M = *unwrap(HostMRef);
  if (!hasOffloadEntries(M))
    return true;

  ErrorOr<std::unique_ptr<MemoryBuffer>> BufOrErr =
      MemoryBuffer::getFile(DeviceImagePath);
  if (std::error_code EC = BufOrErr.getError())
    return reportAndFailWrappingImages(createFileError(DeviceImagePath, EC),
                                       "device image");
  std::unique_ptr<MemoryBuffer> Buf = std::move(*BufOrErr);

  std::unique_ptr<MemoryBuffer> DeviceImage;
  StringRef Ext = sys::path::extension(DeviceImagePath);
  if (Ext == ".ptx") {
    Expected<std::unique_ptr<MemoryBuffer>> CubinOrErr =
        assembleWithPtxas(Buf->getMemBufferRef());
    if (!CubinOrErr)
      return reportAndFailWrappingImages(CubinOrErr.takeError(), "ptxas");
    DeviceImage = std::move(*CubinOrErr);
  } else if (Ext == ".elf") {
    DeviceImage = std::move(Buf);
  } else {
    return reportAndFailWrappingImages(
        createStringError("unsupported device image %s: expected .ptx or .elf",
                          DeviceImagePath),
        "device image path");
  }

  StringRef ImageBuf = DeviceImage->getBuffer();
  ArrayRef<char> Image(ImageBuf.data(), ImageBuf.size());

  if (Error E = offloading::wrapOpenMPBinaries(
          M, {Image}, offloading::getOffloadEntryArray(M), /*Suffix=*/"",
          /*Relocatable=*/
          false))
    return reportAndFailWrappingImages(std::move(E), "wrap");
  return true;
}
