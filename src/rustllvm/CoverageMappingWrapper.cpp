#include "rustllvm.h"
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

extern "C" uint64_t LLVMRustCoverageComputeHash(const char *Name) {
  StringRef NameRef(Name);
  return IndexedInstrProf::ComputeHash(NameRef);
}

extern "C" void LLVMRustCoverageWriteSectionNameToString(LLVMModuleRef M,
                                                         RustStringRef Str) {
  Triple TargetTriple(unwrap(M)->getTargetTriple());
  auto name = getInstrProfSectionName(IPSK_covmap,
                                      TargetTriple.getObjectFormat());
  RawRustStringOstream OS(Str);
  OS << name;
}

extern "C" void LLVMRustCoverageWriteMappingVarNameToString(RustStringRef Str) {
  auto name = getCoverageMappingVarName();
  RawRustStringOstream OS(Str);
  OS << name;
}

extern "C" uint32_t LLVMRustCoverageMappingVersion() {
  return coverage::CovMapVersion::CurrentVersion;
}
