// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#include "llvm/Linker/Linker.h"

#include "rustllvm.h"

using namespace llvm;

struct RustLinker {
  Linker L;
  LLVMContext &Ctx;

  RustLinker(Module &M) :
    L(M),
    Ctx(M.getContext())
  {}
};

extern "C" RustLinker*
LLVMRustLinkerNew(LLVMModuleRef DstRef) {
  Module *Dst = unwrap(DstRef);

  auto Ret = llvm::make_unique<RustLinker>(*Dst);
  return Ret.release();
}

extern "C" void
LLVMRustLinkerFree(RustLinker *L) {
  delete L;
}

extern "C" bool
LLVMRustLinkerAdd(RustLinker *L, char *BC, size_t Len) {
  std::unique_ptr<MemoryBuffer> Buf =
      MemoryBuffer::getMemBufferCopy(StringRef(BC, Len));

#if LLVM_VERSION_GE(4, 0)
  Expected<std::unique_ptr<Module>> SrcOrError =
      llvm::getLazyBitcodeModule(Buf->getMemBufferRef(), L->Ctx);
  if (!SrcOrError) {
    LLVMRustSetLastError(toString(SrcOrError.takeError()).c_str());
    return false;
  }

  auto Src = std::move(*SrcOrError);
#else
  ErrorOr<std::unique_ptr<Module>> Src =
      llvm::getLazyBitcodeModule(std::move(Buf), L->Ctx);
  if (!Src) {
    LLVMRustSetLastError(Src.getError().message().c_str());
    return false;
  }
#endif

#if LLVM_VERSION_GE(4, 0)
  if (L->L.linkInModule(std::move(Src))) {
#else
  if (L->L.linkInModule(std::move(Src.get()))) {
#endif
    LLVMRustSetLastError("");
    return false;
  }
  return true;
}
