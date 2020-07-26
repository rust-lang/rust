#include "LLVMWrapper.h"
#include "llvm/ProfileData/Coverage/CoverageMapping.h"
#include "llvm/ProfileData/Coverage/CoverageMappingWriter.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/LEB128.h"

#include <iostream>

using namespace llvm;

extern "C" void LLVMRustCoverageWriteFilenamesSectionToBuffer(
    const char* const Filenames[],
    size_t FilenamesLen,
    RustStringRef BufferOut) {
  // LLVM 11's CoverageFilenamesSectionWriter uses its new `Version4` format,
  // so we're manually writing the `Version3` format ourselves.
  RawRustStringOstream OS(BufferOut);
  encodeULEB128(FilenamesLen, OS);
  for (size_t i = 0; i < FilenamesLen; i++) {
    StringRef Filename(Filenames[i]);
    encodeULEB128(Filename.size(), OS);
    OS << Filename;
  }
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
  return coverage::CovMapVersion::Version3;
}
