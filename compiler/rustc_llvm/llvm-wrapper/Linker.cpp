#include "LLVMWrapper.h"

#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/Module.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace llvm;

struct RustLinker {
  Linker L;
  LLVMContext &Ctx;

  RustLinker(Module &M) : L(M), Ctx(M.getContext()) {}
};

extern "C" RustLinker *LLVMRustLinkerNew(LLVMModuleRef DstRef) {
  Module *Dst = unwrap(DstRef);

  return new RustLinker(*Dst);
}

extern "C" void LLVMRustLinkerFree(RustLinker *L) { delete L; }

extern "C" bool LLVMRustLinkerAdd(RustLinker *L, char *BC, size_t Len) {
  std::unique_ptr<MemoryBuffer> Buf =
      MemoryBuffer::getMemBufferCopy(StringRef(BC, Len));

  Expected<std::unique_ptr<Module>> SrcOrError =
      llvm::getLazyBitcodeModule(Buf->getMemBufferRef(), L->Ctx);
  if (!SrcOrError) {
    LLVMRustSetLastError(toString(SrcOrError.takeError()).c_str());
    return false;
  }

  auto Src = std::move(*SrcOrError);

  if (L->L.linkInModule(std::move(Src))) {
    LLVMRustSetLastError("");
    return false;
  }
  return true;
}
