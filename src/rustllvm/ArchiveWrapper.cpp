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

struct LLVMRustArchiveMember {
  const char *filename;
  const char *name;
  Archive::Child child;

  LLVMRustArchiveMember(): filename(NULL), name(NULL),
#if LLVM_VERSION_MINOR >= 8
    child(NULL, NULL, NULL)
#else
    child(NULL, NULL)
#endif
  {}
  ~LLVMRustArchiveMember() {}
};

typedef OwningBinary<Archive> RustArchive;

extern "C" void*
LLVMRustOpenArchive(char *path) {
    ErrorOr<std::unique_ptr<MemoryBuffer>> buf_or = MemoryBuffer::getFile(path,
                                                                          -1,
                                                                          false);
    if (!buf_or) {
        LLVMRustSetLastError(buf_or.getError().message().c_str());
        return nullptr;
    }

#if LLVM_VERSION_MINOR <= 8
    ErrorOr<std::unique_ptr<Archive>> archive_or =
#else
    Expected<std::unique_ptr<Archive>> archive_or =
#endif
        Archive::create(buf_or.get()->getMemBufferRef());

    if (!archive_or) {
#if LLVM_VERSION_MINOR <= 8
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
LLVMRustDestroyArchive(RustArchive *ar) {
    delete ar;
}

struct RustArchiveIterator {
    Archive::child_iterator cur;
    Archive::child_iterator end;
#if LLVM_VERSION_MINOR >= 9
    Error err;
#endif
};

extern "C" RustArchiveIterator*
LLVMRustArchiveIteratorNew(RustArchive *ra) {
    Archive *ar = ra->getBinary();
    RustArchiveIterator *rai = new RustArchiveIterator();
#if LLVM_VERSION_MINOR <= 8
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

extern "C" const Archive::Child*
LLVMRustArchiveIteratorNext(RustArchiveIterator *rai) {
#if LLVM_VERSION_MINOR >= 9
    if (rai->err) {
        LLVMRustSetLastError(toString(std::move(rai->err)).c_str());
        return NULL;
    }
#endif
    if (rai->cur == rai->end)
        return NULL;
#if LLVM_VERSION_MINOR == 8
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
LLVMRustArchiveChildFree(Archive::Child *child) {
    delete child;
}

extern "C" void
LLVMRustArchiveIteratorFree(RustArchiveIterator *rai) {
    delete rai;
}

extern "C" const char*
LLVMRustArchiveChildName(const Archive::Child *child, size_t *size) {
    ErrorOr<StringRef> name_or_err = child->getName();
    if (name_or_err.getError())
        return NULL;
    StringRef name = name_or_err.get();
    *size = name.size();
    return name.data();
}

extern "C" const char*
LLVMRustArchiveChildData(Archive::Child *child, size_t *size) {
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

extern "C" LLVMRustArchiveMember*
LLVMRustArchiveMemberNew(char *Filename, char *Name, Archive::Child *child) {
    LLVMRustArchiveMember *Member = new LLVMRustArchiveMember;
    Member->filename = Filename;
    Member->name = Name;
    if (child)
        Member->child = *child;
    return Member;
}

extern "C" void
LLVMRustArchiveMemberFree(LLVMRustArchiveMember *Member) {
    delete Member;
}

extern "C" int
LLVMRustWriteArchive(char *Dst,
                     size_t NumMembers,
                     const LLVMRustArchiveMember **NewMembers,
                     bool WriteSymbtab,
                     Archive::Kind Kind) {

#if LLVM_VERSION_MINOR <= 8
  std::vector<NewArchiveIterator> Members;
#else
  std::vector<NewArchiveMember> Members;
#endif

  for (size_t i = 0; i < NumMembers; i++) {
    auto Member = NewMembers[i];
    assert(Member->name);
    if (Member->filename) {
#if LLVM_VERSION_MINOR >= 9
      Expected<NewArchiveMember> MOrErr = NewArchiveMember::getFile(Member->filename, true);
      if (!MOrErr) {
        LLVMRustSetLastError(toString(MOrErr.takeError()).c_str());
        return -1;
      }
      Members.push_back(std::move(*MOrErr));
#elif LLVM_VERSION_MINOR == 8
      Members.push_back(NewArchiveIterator(Member->filename));
#else
      Members.push_back(NewArchiveIterator(Member->filename, Member->name));
#endif
    } else {
#if LLVM_VERSION_MINOR <= 8
      Members.push_back(NewArchiveIterator(Member->child, Member->name));
#else
      Expected<NewArchiveMember> MOrErr = NewArchiveMember::getOldMember(Member->child, true);
      if (!MOrErr) {
        LLVMRustSetLastError(toString(MOrErr.takeError()).c_str());
        return -1;
      }
      Members.push_back(std::move(*MOrErr));
#endif
    }
  }
#if LLVM_VERSION_MINOR >= 8
  auto pair = writeArchive(Dst, Members, WriteSymbtab, Kind, true, false);
#else
  auto pair = writeArchive(Dst, Members, WriteSymbtab, Kind, true);
#endif
  if (!pair.second)
    return 0;
  LLVMRustSetLastError(pair.second.message().c_str());
  return -1;
}
