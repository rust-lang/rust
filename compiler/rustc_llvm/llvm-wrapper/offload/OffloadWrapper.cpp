#include "../SuppressLLVMWarnings.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Object/OffloadBinary.h"
#include "llvm/Support/CBindingWrapping.h"
#include "llvm/Support/FileOutputBuffer.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

using namespace llvm;
using namespace llvm::object;

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

// This is the first of many steps in creating a binary using llvm offload,
// to run code on the gpu. Concrete, it replaces the following binary use:
// clang-offload-packager -o device.bin
//  --image=file=device.bc,triple=amdgcn-amd-amdhsa,arch=gfx90a,kind=openmp
// The input module is the rust code compiled for a gpu target like amdgpu.
// Based on clang/tools/clang-offload-packager/ClangOffloadPackager.cpp
extern "C" bool LLVMRustBundleImages(LLVMModuleRef M, TargetMachine &TM,
                                     const char *HostOutPath) {
  std::string Storage;
  llvm::raw_string_ostream OS1(Storage);
  llvm::WriteBitcodeToFile(*unwrap(M), OS1);
  OS1.flush();
  auto MB = llvm::MemoryBuffer::getMemBufferCopy(Storage, "device.bc");

  SmallVector<char, 1024> BinaryData;
  raw_svector_ostream OS2(BinaryData);

  OffloadBinary::OffloadingImage ImageBinary{};
  ImageBinary.TheImageKind = object::IMG_Bitcode;
  ImageBinary.Image = std::move(MB);
  ImageBinary.TheOffloadKind = object::OFK_OpenMP;

  std::string TripleStr = TM.getTargetTriple().str();
  llvm::StringRef CPURef = TM.getTargetCPU();
  ImageBinary.StringData["triple"] = TripleStr;
  ImageBinary.StringData["arch"] = CPURef;
  llvm::SmallString<0> Buffer = OffloadBinary::write(ImageBinary);
  if (Buffer.size() % OffloadBinary::getAlignment() != 0)
    // Offload binary has invalid size alignment
    return false;
  OS2 << Buffer;
  if (Error E = writeFile(HostOutPath,
                          StringRef(BinaryData.begin(), BinaryData.size())))
    return false;
  return true;
}

extern "C" bool LLVMRustOffloadEmbedBufferInModule(LLVMModuleRef HostM,
                                                   const char *HostOutPath) {
  auto MBOrErr = MemoryBuffer::getFile(HostOutPath);
  if (!MBOrErr) {
    auto E = MBOrErr.getError();
    auto _B = errorCodeToError(E);
    return false;
  }
  MemoryBufferRef Buf = (*MBOrErr)->getMemBufferRef();
  Module *M = unwrap(HostM);
  StringRef SectionName = ".llvm.offloading";
  Align Alignment = Align(8);
  llvm::embedBufferInModule(*M, Buf, SectionName, Alignment);
  return true;
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
