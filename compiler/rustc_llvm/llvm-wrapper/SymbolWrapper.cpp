// Derived from code in LLVM, which is:
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Derived from:
// * https://github.com/llvm/llvm-project/blob/8ef3e895ad8ab1724e2b87cabad1dacdc7a397a3/llvm/include/llvm/Object/ArchiveWriter.h
// * https://github.com/llvm/llvm-project/blob/8ef3e895ad8ab1724e2b87cabad1dacdc7a397a3/llvm/lib/Object/ArchiveWriter.cpp

#include "llvm/IR/LLVMContext.h"
#include "llvm/Object/ObjectFile.h"

using namespace llvm;
using namespace llvm::sys;
using namespace llvm::object;

static bool isArchiveSymbol(const object::BasicSymbolRef &S) {
  Expected<uint32_t> SymFlagsOrErr = S.getFlags();
  if (!SymFlagsOrErr)
    // FIXME: Actually report errors helpfully.
    report_fatal_error(SymFlagsOrErr.takeError());
  if (*SymFlagsOrErr & object::SymbolRef::SF_FormatSpecific)
    return false;
  if (!(*SymFlagsOrErr & object::SymbolRef::SF_Global))
    return false;
  if (*SymFlagsOrErr & object::SymbolRef::SF_Undefined)
    return false;
  return true;
}

typedef void *(*LLVMRustGetSymbolsCallback)(void *, const char *);
typedef void *(*LLVMRustGetSymbolsErrorCallback)(const char *);

// Note: This is implemented in C++ instead of using the C api from Rust as IRObjectFile doesn't
// implement getSymbolName, only printSymbolName, which is inaccessible from the C api.
extern "C" void *LLVMRustGetSymbols(
  char *BufPtr, size_t BufLen, void *State, LLVMRustGetSymbolsCallback Callback,
  LLVMRustGetSymbolsErrorCallback ErrorCallback) {
  std::unique_ptr<MemoryBuffer> Buf =
    MemoryBuffer::getMemBuffer(StringRef(BufPtr, BufLen), StringRef("LLVMRustGetSymbolsObject"),
                               false);
  SmallString<0> SymNameBuf;
  raw_svector_ostream SymName(SymNameBuf);

  // In the scenario when LLVMContext is populated SymbolicFile will contain a
  // reference to it, thus SymbolicFile should be destroyed first.
  LLVMContext Context;
  std::unique_ptr<object::SymbolicFile> Obj;

  const file_magic Type = identify_magic(Buf->getBuffer());
  if (!object::SymbolicFile::isSymbolicFile(Type, &Context)) {
    return 0;
  }

  if (Type == file_magic::bitcode) {
    auto ObjOrErr = object::SymbolicFile::createSymbolicFile(
      Buf->getMemBufferRef(), file_magic::bitcode, &Context);
    if (!ObjOrErr) {
      Error E = ObjOrErr.takeError();
      SmallString<0> ErrorBuf;
      raw_svector_ostream Error(ErrorBuf);
      Error << E << '\0';
      return ErrorCallback(Error.str().data());
    }
    Obj = std::move(*ObjOrErr);
  } else {
    auto ObjOrErr = object::SymbolicFile::createSymbolicFile(Buf->getMemBufferRef());
    if (!ObjOrErr) {
      Error E = ObjOrErr.takeError();
      SmallString<0> ErrorBuf;
      raw_svector_ostream Error(ErrorBuf);
      Error << E << '\0';
      return ErrorCallback(Error.str().data());
    }
    Obj = std::move(*ObjOrErr);
  }


  for (const object::BasicSymbolRef &S : Obj->symbols()) {
    if (!isArchiveSymbol(S))
      continue;
    if (Error E = S.printName(SymName)) {
      SmallString<0> ErrorBuf;
      raw_svector_ostream Error(ErrorBuf);
      Error << E << '\0';
      return ErrorCallback(Error.str().data());
    }
    SymName << '\0';
    if (void *E = Callback(State, SymNameBuf.str().data())) {
      return E;
    }
    SymNameBuf.clear();
  }
  return 0;
}
