#include "LLVMWrapper.h"
#include "llvm/ProfileData/Coverage/CoverageMapping.h"
#include "llvm/ProfileData/Coverage/CoverageMappingWriter.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/ADT/ArrayRef.h"

#include <iostream>

using namespace llvm;

extern "C" void LLVMRustCoverageWriteFilenamesSectionToBuffer(
    const char* const Filenames[],
    size_t FilenamesLen,
    RustStringRef BufferOut) {
  SmallVector<StringRef,32> FilenameRefs;
  for (size_t i = 0; i < FilenamesLen; i++) {
    FilenameRefs.push_back(StringRef(Filenames[i]));
  }
  auto FilenamesWriter = coverage::CoverageFilenamesSectionWriter(
    makeArrayRef(FilenameRefs));
  RawRustStringOstream OS(BufferOut);
  FilenamesWriter.write(OS);
}

extern "C" void LLVMRustCoverageWriteMappingToBuffer(
    const unsigned *VirtualFileMappingIDs,
    unsigned NumVirtualFileMappingIDs,
    const coverage::CounterExpression *Expressions,
    unsigned NumExpressions,
    coverage::CounterMappingRegion *MappingRegions,
    unsigned NumMappingRegions,
    RustStringRef BufferOut) {
  auto CoverageMappingWriter = coverage::CoverageMappingWriter(
      makeArrayRef(VirtualFileMappingIDs, NumVirtualFileMappingIDs),
      makeArrayRef(Expressions, NumExpressions),
      makeMutableArrayRef(MappingRegions, NumMappingRegions));
  RawRustStringOstream OS(BufferOut);
  CoverageMappingWriter.write(OS);
}

extern "C" LLVMValueRef LLVMRustCoverageCreatePGOFuncNameVar(LLVMValueRef F, const char *FuncName) {
  StringRef FuncNameRef(FuncName);
  return wrap(createPGOFuncNameVar(*cast<Function>(unwrap(F)), FuncNameRef));
}

extern "C" uint64_t LLVMRustCoverageHashCString(const char *StrVal) {
  StringRef StrRef(StrVal);
  return IndexedInstrProf::ComputeHash(StrRef);
}

extern "C" uint64_t LLVMRustCoverageHashByteArray(
    const char *Bytes,
    unsigned NumBytes) {
  StringRef StrRef(Bytes, NumBytes);
  return IndexedInstrProf::ComputeHash(StrRef);
}

static void WriteSectionNameToString(LLVMModuleRef M,
                                     InstrProfSectKind SK,
                                     RustStringRef Str) {
  Triple TargetTriple(unwrap(M)->getTargetTriple());
  auto name = getInstrProfSectionName(SK, TargetTriple.getObjectFormat());
  RawRustStringOstream OS(Str);
  OS << name;
}

extern "C" void LLVMRustCoverageWriteMapSectionNameToString(LLVMModuleRef M,
                                                            RustStringRef Str) {
  WriteSectionNameToString(M, IPSK_covmap, Str);
}

extern "C" void LLVMRustCoverageWriteFuncSectionNameToString(LLVMModuleRef M,
                                                             RustStringRef Str) {
#if LLVM_VERSION_GE(11, 0)
  WriteSectionNameToString(M, IPSK_covfun, Str);
// else do nothing; the `Version` check will abort codegen on the Rust side
#endif
}

extern "C" void LLVMRustCoverageWriteMappingVarNameToString(RustStringRef Str) {
  auto name = getCoverageMappingVarName();
  RawRustStringOstream OS(Str);
  OS << name;
}

extern "C" uint32_t LLVMRustCoverageMappingVersion() {
#if LLVM_VERSION_GE(11, 0)
  return coverage::CovMapVersion::Version4;
#else
  return coverage::CovMapVersion::Version3;
#endif
}
