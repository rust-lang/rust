#include "LLVMWrapper.h"

#include "llvm/Object/Archive.h"
#include "llvm/Object/ArchiveWriter.h"
#include "llvm/Support/Path.h"

using namespace llvm;
using namespace llvm::object;

struct RustArchiveMember {
  const char *Filename;
  const char *Name;
  Archive::Child Child;

  RustArchiveMember()
      : Filename(nullptr), Name(nullptr),
        Child(nullptr, nullptr, nullptr)
  {
  }
  ~RustArchiveMember() {}
};

struct RustArchiveIterator {
  bool First;
  Archive::child_iterator Cur;
  Archive::child_iterator End;
  std::unique_ptr<Error> Err;

  RustArchiveIterator(Archive::child_iterator Cur, Archive::child_iterator End,
      std::unique_ptr<Error> Err)
    : First(true),
      Cur(Cur),
      End(End),
      Err(std::move(Err)) {}
};

enum class LLVMRustArchiveKind {
  GNU,
  BSD,
  DARWIN,
  COFF,
};

static Archive::Kind fromRust(LLVMRustArchiveKind Kind) {
  switch (Kind) {
  case LLVMRustArchiveKind::GNU:
    return Archive::K_GNU;
  case LLVMRustArchiveKind::BSD:
    return Archive::K_BSD;
  case LLVMRustArchiveKind::DARWIN:
    return Archive::K_DARWIN;
  case LLVMRustArchiveKind::COFF:
    return Archive::K_COFF;
  default:
    report_fatal_error("Bad ArchiveKind.");
  }
}

typedef OwningBinary<Archive> *LLVMRustArchiveRef;
typedef RustArchiveMember *LLVMRustArchiveMemberRef;
typedef Archive::Child *LLVMRustArchiveChildRef;
typedef Archive::Child const *LLVMRustArchiveChildConstRef;
typedef RustArchiveIterator *LLVMRustArchiveIteratorRef;

extern "C" LLVMRustArchiveRef LLVMRustOpenArchive(char *Path) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> BufOr =
      MemoryBuffer::getFile(Path, -1, false);
  if (!BufOr) {
    LLVMRustSetLastError(BufOr.getError().message().c_str());
    return nullptr;
  }

  Expected<std::unique_ptr<Archive>> ArchiveOr =
      Archive::create(BufOr.get()->getMemBufferRef());

  if (!ArchiveOr) {
    LLVMRustSetLastError(toString(ArchiveOr.takeError()).c_str());
    return nullptr;
  }

  OwningBinary<Archive> *Ret = new OwningBinary<Archive>(
      std::move(ArchiveOr.get()), std::move(BufOr.get()));

  return Ret;
}

extern "C" void LLVMRustDestroyArchive(LLVMRustArchiveRef RustArchive) {
  delete RustArchive;
}

extern "C" LLVMRustArchiveIteratorRef
LLVMRustArchiveIteratorNew(LLVMRustArchiveRef RustArchive) {
  Archive *Archive = RustArchive->getBinary();
  std::unique_ptr<Error> Err = std::make_unique<Error>(Error::success());
  auto Cur = Archive->child_begin(*Err);
  if (*Err) {
    LLVMRustSetLastError(toString(std::move(*Err)).c_str());
    return nullptr;
  }
  auto End = Archive->child_end();
  return new RustArchiveIterator(Cur, End, std::move(Err));
}

extern "C" LLVMRustArchiveChildConstRef
LLVMRustArchiveIteratorNext(LLVMRustArchiveIteratorRef RAI) {
  if (RAI->Cur == RAI->End)
    return nullptr;

  // Advancing the iterator validates the next child, and this can
  // uncover an error. LLVM requires that we check all Errors,
  // so we only advance the iterator if we actually need to fetch
  // the next child.
  // This means we must not advance the iterator in the *first* call,
  // but instead advance it *before* fetching the child in all later calls.
  if (!RAI->First) {
    ++RAI->Cur;
    if (*RAI->Err) {
      LLVMRustSetLastError(toString(std::move(*RAI->Err)).c_str());
      return nullptr;
    }
  } else {
    RAI->First = false;
  }

  if (RAI->Cur == RAI->End)
    return nullptr;

  const Archive::Child &Child = *RAI->Cur.operator->();
  Archive::Child *Ret = new Archive::Child(Child);

  return Ret;
}

extern "C" void LLVMRustArchiveChildFree(LLVMRustArchiveChildRef Child) {
  delete Child;
}

extern "C" void LLVMRustArchiveIteratorFree(LLVMRustArchiveIteratorRef RAI) {
  delete RAI;
}

extern "C" const char *
LLVMRustArchiveChildName(LLVMRustArchiveChildConstRef Child, size_t *Size) {
  Expected<StringRef> NameOrErr = Child->getName();
  if (!NameOrErr) {
    // rustc_codegen_llvm currently doesn't use this error string, but it might be
    // useful in the future, and in the mean time this tells LLVM that the
    // error was not ignored and that it shouldn't abort the process.
    LLVMRustSetLastError(toString(NameOrErr.takeError()).c_str());
    return nullptr;
  }
  StringRef Name = NameOrErr.get();
  *Size = Name.size();
  return Name.data();
}

extern "C" LLVMRustArchiveMemberRef
LLVMRustArchiveMemberNew(char *Filename, char *Name,
                         LLVMRustArchiveChildRef Child) {
  RustArchiveMember *Member = new RustArchiveMember;
  Member->Filename = Filename;
  Member->Name = Name;
  if (Child)
    Member->Child = *Child;
  return Member;
}

extern "C" void LLVMRustArchiveMemberFree(LLVMRustArchiveMemberRef Member) {
  delete Member;
}

extern "C" LLVMRustResult
LLVMRustWriteArchive(char *Dst, size_t NumMembers,
                     const LLVMRustArchiveMemberRef *NewMembers,
                     bool WriteSymbtab, LLVMRustArchiveKind RustKind) {

  std::vector<NewArchiveMember> Members;
  auto Kind = fromRust(RustKind);

  for (size_t I = 0; I < NumMembers; I++) {
    auto Member = NewMembers[I];
    assert(Member->Name);
    if (Member->Filename) {
      Expected<NewArchiveMember> MOrErr =
          NewArchiveMember::getFile(Member->Filename, true);
      if (!MOrErr) {
        LLVMRustSetLastError(toString(MOrErr.takeError()).c_str());
        return LLVMRustResult::Failure;
      }
      MOrErr->MemberName = sys::path::filename(MOrErr->MemberName);
      Members.push_back(std::move(*MOrErr));
    } else {
      Expected<NewArchiveMember> MOrErr =
          NewArchiveMember::getOldMember(Member->Child, true);
      if (!MOrErr) {
        LLVMRustSetLastError(toString(MOrErr.takeError()).c_str());
        return LLVMRustResult::Failure;
      }
      Members.push_back(std::move(*MOrErr));
    }
  }

  auto Result = writeArchive(Dst, Members, WriteSymbtab, Kind, true, false);
  if (!Result)
    return LLVMRustResult::Success;
  LLVMRustSetLastError(toString(std::move(Result)).c_str());

  return LLVMRustResult::Failure;
}
