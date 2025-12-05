// Derived from code in LLVM, which is:
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Derived from:
// * https://github.com/llvm/llvm-project/blob/ef6d1ec07c693352c4a60dd58db08d2d8558f6ea/llvm/include/llvm/Object/ArchiveWriter.h
// * https://github.com/llvm/llvm-project/blob/ef6d1ec07c693352c4a60dd58db08d2d8558f6ea/llvm/lib/Object/ArchiveWriter.cpp

#include "LLVMWrapper.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Object/COFF.h"
#include "llvm/Object/COFFImportFile.h"
#include "llvm/Object/IRObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/raw_ostream.h"

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

// This function is copied from ArchiveWriter.cpp.
static Expected<std::unique_ptr<SymbolicFile>>
getSymbolicFile(MemoryBufferRef Buf, LLVMContext &Context) {
  const file_magic Type = identify_magic(Buf.getBuffer());
  // Don't attempt to read non-symbolic file types.
  if (!object::SymbolicFile::isSymbolicFile(Type, &Context))
    return nullptr;
  if (Type == file_magic::bitcode) {
    auto ObjOrErr = object::SymbolicFile::createSymbolicFile(
        Buf, file_magic::bitcode, &Context);
    if (!ObjOrErr)
      return ObjOrErr.takeError();
    return std::move(*ObjOrErr);
  } else {
    auto ObjOrErr = object::SymbolicFile::createSymbolicFile(Buf);
    if (!ObjOrErr)
      return ObjOrErr.takeError();
    return std::move(*ObjOrErr);
  }
}

// Note: This is implemented in C++ instead of using the C api from Rust as
// IRObjectFile doesn't implement getSymbolName, only printSymbolName, which is
// inaccessible from the C api.
extern "C" void *
LLVMRustGetSymbols(char *BufPtr, size_t BufLen, void *State,
                   LLVMRustGetSymbolsCallback Callback,
                   LLVMRustGetSymbolsErrorCallback ErrorCallback) {
  std::unique_ptr<MemoryBuffer> Buf = MemoryBuffer::getMemBuffer(
      StringRef(BufPtr, BufLen), StringRef("LLVMRustGetSymbolsObject"), false);
  SmallString<0> SymNameBuf;
  auto SymName = raw_svector_ostream(SymNameBuf);

  // In the scenario when LLVMContext is populated SymbolicFile will contain a
  // reference to it, thus SymbolicFile should be destroyed first.
  LLVMContext Context;
  Expected<std::unique_ptr<object::SymbolicFile>> ObjOrErr =
      getSymbolicFile(Buf->getMemBufferRef(), Context);
  if (!ObjOrErr) {
    return ErrorCallback(toString(ObjOrErr.takeError()).c_str());
  }
  std::unique_ptr<object::SymbolicFile> Obj = std::move(*ObjOrErr);
  if (Obj == nullptr) {
    return 0;
  }

  for (const object::BasicSymbolRef &S : Obj->symbols()) {
    if (!isArchiveSymbol(S))
      continue;
    if (Error E = S.printName(SymName)) {
      return ErrorCallback(toString(std::move(E)).c_str());
    }
    SymName << '\0';
    if (void *E = Callback(State, SymNameBuf.str().data())) {
      return E;
    }
    SymNameBuf.clear();
  }
  return 0;
}

// Encoding true and false as invalid pointer values
#define TRUE_PTR (void *)1
#define FALSE_PTR (void *)0

bool withBufferAsSymbolicFile(
    char *BufPtr, size_t BufLen,
    std::function<bool(object::SymbolicFile &)> Callback) {
  std::unique_ptr<MemoryBuffer> Buf = MemoryBuffer::getMemBuffer(
      StringRef(BufPtr, BufLen), StringRef("LLVMRustGetSymbolsObject"), false);
  SmallString<0> SymNameBuf;
  auto SymName = raw_svector_ostream(SymNameBuf);

  // In the scenario when LLVMContext is populated SymbolicFile will contain a
  // reference to it, thus SymbolicFile should be destroyed first.
  LLVMContext Context;
  Expected<std::unique_ptr<object::SymbolicFile>> ObjOrErr =
      getSymbolicFile(Buf->getMemBufferRef(), Context);
  if (!ObjOrErr) {
    return false;
  }
  std::unique_ptr<object::SymbolicFile> Obj = std::move(*ObjOrErr);
  if (Obj == nullptr) {
    return false;
  }
  return Callback(*Obj);
}

extern "C" bool LLVMRustIs64BitSymbolicFile(char *BufPtr, size_t BufLen) {
  return withBufferAsSymbolicFile(
      BufPtr, BufLen, [](object::SymbolicFile &Obj) { return Obj.is64Bit(); });
}

extern "C" bool LLVMRustIsECObject(char *BufPtr, size_t BufLen) {
  return withBufferAsSymbolicFile(
      BufPtr, BufLen, [](object::SymbolicFile &Obj) {
        // Code starting from this line is copied from isECObject in
        // ArchiveWriter.cpp with an extra #if to work with LLVM 17.
        if (Obj.isCOFF())
          return cast<llvm::object::COFFObjectFile>(&Obj)->getMachine() !=
                 COFF::IMAGE_FILE_MACHINE_ARM64;

        if (Obj.isCOFFImportFile())
          return cast<llvm::object::COFFImportFile>(&Obj)->getMachine() !=
                 COFF::IMAGE_FILE_MACHINE_ARM64;

        if (Obj.isIR()) {
          Expected<std::string> TripleStr =
              getBitcodeTargetTriple(Obj.getMemoryBufferRef());
          if (!TripleStr)
            return false;
          Triple T(*TripleStr);
          return T.isWindowsArm64EC() || T.getArch() == Triple::x86_64;
        }

        return false;
      });
}

extern "C" bool LLVMRustIsAnyArm64Coff(char *BufPtr, size_t BufLen) {
  return withBufferAsSymbolicFile(
      BufPtr, BufLen, [](object::SymbolicFile &Obj) {
        // Code starting from this line is copied from isAnyArm64COFF in
        // ArchiveWriter.cpp.
        if (Obj.isCOFF())
          return COFF::isAnyArm64(cast<COFFObjectFile>(&Obj)->getMachine());

        if (Obj.isCOFFImportFile())
          return COFF::isAnyArm64(cast<COFFImportFile>(&Obj)->getMachine());

        if (Obj.isIR()) {
          Expected<std::string> TripleStr =
              getBitcodeTargetTriple(Obj.getMemoryBufferRef());
          if (!TripleStr)
            return false;
          Triple T(*TripleStr);
          return T.isOSWindows() && T.getArch() == Triple::aarch64;
        }

        return false;
      });
}
