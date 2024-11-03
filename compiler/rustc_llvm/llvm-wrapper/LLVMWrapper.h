#ifndef INCLUDED_RUSTC_LLVM_LLVMWRAPPER_H
#define INCLUDED_RUSTC_LLVM_LLVMWRAPPER_H

#include "SuppressLLVMWarnings.h"

#include "llvm/Config/llvm-config.h"  // LLVM_VERSION_MAJOR, LLVM_VERSION_MINOR
#include "llvm/Support/raw_ostream.h" // llvm::raw_ostream
#include <cstddef>                    // size_t etc
#include <cstdint>                    // uint64_t etc

#define LLVM_VERSION_GE(major, minor)                                          \
  (LLVM_VERSION_MAJOR > (major) ||                                             \
   LLVM_VERSION_MAJOR == (major) && LLVM_VERSION_MINOR >= (minor))

#define LLVM_VERSION_LT(major, minor) (!LLVM_VERSION_GE((major), (minor)))

extern "C" void LLVMRustSetLastError(const char *);

enum class LLVMRustResult { Success, Failure };

typedef struct OpaqueRustString *RustStringRef;
typedef struct LLVMOpaqueTwine *LLVMTwineRef;
typedef struct LLVMOpaqueSMDiagnostic *LLVMSMDiagnosticRef;

extern "C" void LLVMRustStringWriteImpl(RustStringRef Str, const char *Ptr,
                                        size_t Size);

class RawRustStringOstream : public llvm::raw_ostream {
  RustStringRef Str;
  uint64_t Pos;

  void write_impl(const char *Ptr, size_t Size) override {
    LLVMRustStringWriteImpl(Str, Ptr, Size);
    Pos += Size;
  }

  uint64_t current_pos() const override { return Pos; }

public:
  explicit RawRustStringOstream(RustStringRef Str) : Str(Str), Pos(0) {}

  ~RawRustStringOstream() {
    // LLVM requires this.
    flush();
  }
};

#endif // INCLUDED_RUSTC_LLVM_LLVMWRAPPER_H
