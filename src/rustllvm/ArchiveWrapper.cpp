// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#include "rustllvm.h"

#include "llvm/Object/Archive.h"
#include "llvm/Object/ArchiveWriter.h"

using namespace llvm;
using namespace llvm::object;

struct RustArchiveMember {
  const char *filename;
  const char *name;
  Archive::Child child;

  RustArchiveMember(): filename(NULL), name(NULL),
#if LLVM_VERSION_GE(3, 8)
    child(NULL, NULL, NULL)
#else
    child(NULL, NULL)
#endif
  {}
  ~RustArchiveMember() {}
};


struct RustArchiveIterator {
    Archive::child_iterator cur;
    Archive::child_iterator end;
#if LLVM_VERSION_GE(3, 9)
    Error err;
#endif
};

enum class LLVMRustArchiveKind {
    Other,
    GNU,
    MIPS64,
    BSD,
    COFF,
};

static Archive::Kind
from_rust(LLVMRustArchiveKind kind)
{
    switch (kind) {
    case LLVMRustArchiveKind::GNU:
        return Archive::K_GNU;
    case LLVMRustArchiveKind::MIPS64:
        return Archive::K_MIPS64;
    case LLVMRustArchiveKind::BSD:
        return Archive::K_BSD;
    case LLVMRustArchiveKind::COFF:
        return Archive::K_COFF;
    default:
      llvm_unreachable("Bad ArchiveKind.");
  }
}

typedef OwningBinary<Archive> *LLVMRustArchiveRef;
typedef RustArchiveMember *LLVMRustArchiveMemberRef;
typedef Archive::Child *LLVMRustArchiveChildRef;
typedef Archive::Child const *LLVMRustArchiveChildConstRef;
typedef RustArchiveIterator *LLVMRustArchiveIteratorRef;

extern "C" LLVMRustArchiveRef
LLVMRustOpenArchive(char *path) {
    ErrorOr<std::unique_ptr<MemoryBuffer>> buf_or = MemoryBuffer::getFile(path,
                                                                          -1,
                                                                          false);
    if (!buf_or) {
        LLVMRustSetLastError(buf_or.getError().message().c_str());
        return nullptr;
    }

#if LLVM_VERSION_LE(3, 8)
    ErrorOr<std::unique_ptr<Archive>> archive_or =
#else
    Expected<std::unique_ptr<Archive>> archive_or =
#endif
        Archive::create(buf_or.get()->getMemBufferRef());

    if (!archive_or) {
#if LLVM_VERSION_LE(3, 8)
        LLVMRustSetLastError(archive_or.getError().message().c_str());
#else
        LLVMRustSetLastError(toString(archive_or.takeError()).c_str());
#endif
        return nullptr;
    }

    OwningBinary<Archive> *ret = new OwningBinary<Archive>(
            std::move(archive_or.get()), std::move(buf_or.get()));

    return ret;
}

extern "C" void
LLVMRustDestroyArchive(LLVMRustArchiveRef ar) {
    delete ar;
}

extern "C" LLVMRustArchiveIteratorRef
LLVMRustArchiveIteratorNew(LLVMRustArchiveRef ra) {
    Archive *ar = ra->getBinary();
    RustArchiveIterator *rai = new RustArchiveIterator();
#if LLVM_VERSION_LE(3, 8)
    rai->cur = ar->child_begin();
#else
    rai->cur = ar->child_begin(rai->err);
    if (rai->err) {
        LLVMRustSetLastError(toString(std::move(rai->err)).c_str());
        return NULL;
    }
#endif
    rai->end = ar->child_end();
    return rai;
}

extern "C" LLVMRustArchiveChildConstRef
LLVMRustArchiveIteratorNext(LLVMRustArchiveIteratorRef rai) {
#if LLVM_VERSION_GE(3, 9)
    if (rai->err) {
        LLVMRustSetLastError(toString(std::move(rai->err)).c_str());
        return NULL;
    }
#endif
    if (rai->cur == rai->end)
        return NULL;
#if LLVM_VERSION_EQ(3, 8)
    const ErrorOr<Archive::Child>* cur = rai->cur.operator->();
    if (!*cur) {
        LLVMRustSetLastError(cur->getError().message().c_str());
        return NULL;
    }
    const Archive::Child &child = cur->get();
#else
    const Archive::Child &child = *rai->cur.operator->();
#endif
    Archive::Child *ret = new Archive::Child(child);

    ++rai->cur;
    return ret;
}

extern "C" void
LLVMRustArchiveChildFree(LLVMRustArchiveChildRef child) {
    delete child;
}

extern "C" void
LLVMRustArchiveIteratorFree(LLVMRustArchiveIteratorRef rai) {
    delete rai;
}

extern "C" const char*
LLVMRustArchiveChildName(LLVMRustArchiveChildConstRef child, size_t *size) {
    ErrorOr<StringRef> name_or_err = child->getName();
    if (name_or_err.getError())
        return NULL;
    StringRef name = name_or_err.get();
    *size = name.size();
    return name.data();
}

extern "C" const char*
LLVMRustArchiveChildData(LLVMRustArchiveChildRef child, size_t *size) {
    StringRef buf;
    ErrorOr<StringRef> buf_or_err = child->getBuffer();
    if (buf_or_err.getError()) {
      LLVMRustSetLastError(buf_or_err.getError().message().c_str());
      return NULL;
    }
    buf = buf_or_err.get();
    *size = buf.size();
    return buf.data();
}

extern "C" LLVMRustArchiveMemberRef
LLVMRustArchiveMemberNew(char *Filename, char *Name,
			 LLVMRustArchiveChildRef child) {
    RustArchiveMember *Member = new RustArchiveMember;
    Member->filename = Filename;
    Member->name = Name;
    if (child)
        Member->child = *child;
    return Member;
}

extern "C" void
LLVMRustArchiveMemberFree(LLVMRustArchiveMemberRef Member) {
    delete Member;
}

extern "C" LLVMRustResult
LLVMRustWriteArchive(char *Dst,
                     size_t NumMembers,
                     const LLVMRustArchiveMemberRef *NewMembers,
                     bool WriteSymbtab,
                     LLVMRustArchiveKind rust_kind) {

#if LLVM_VERSION_LE(3, 8)
  std::vector<NewArchiveIterator> Members;
#else
  std::vector<NewArchiveMember> Members;
#endif
  auto Kind = from_rust(rust_kind);

  for (size_t i = 0; i < NumMembers; i++) {
    auto Member = NewMembers[i];
    assert(Member->name);
    if (Member->filename) {
#if LLVM_VERSION_GE(3, 9)
      Expected<NewArchiveMember> MOrErr = NewArchiveMember::getFile(Member->filename, true);
      if (!MOrErr) {
        LLVMRustSetLastError(toString(MOrErr.takeError()).c_str());
        return LLVMRustResult::Failure;
      }
      Members.push_back(std::move(*MOrErr));
#elif LLVM_VERSION_EQ(3, 8)
      Members.push_back(NewArchiveIterator(Member->filename));
#else
      Members.push_back(NewArchiveIterator(Member->filename, Member->name));
#endif
    } else {
#if LLVM_VERSION_LE(3, 8)
      Members.push_back(NewArchiveIterator(Member->child, Member->name));
#else
      Expected<NewArchiveMember> MOrErr = NewArchiveMember::getOldMember(Member->child, true);
      if (!MOrErr) {
        LLVMRustSetLastError(toString(MOrErr.takeError()).c_str());
        return LLVMRustResult::Failure;
      }
      Members.push_back(std::move(*MOrErr));
#endif
    }
  }
#if LLVM_VERSION_GE(3, 8)
  auto pair = writeArchive(Dst, Members, WriteSymbtab, Kind, true, false);
#else
  auto pair = writeArchive(Dst, Members, WriteSymbtab, Kind, true);
#endif
  if (!pair.second)
    return LLVMRustResult::Success;
  LLVMRustSetLastError(pair.second.message().c_str());
  return LLVMRustResult::Failure;
}
