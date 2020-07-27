#include "rustllvm.h"
#include "llvm/ProfileData/Coverage/CoverageMapping.h"
#include "llvm/ProfileData/Coverage/CoverageMappingWriter.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/ADT/ArrayRef.h"

#include <iostream>

using namespace llvm;

extern "C" SmallVectorTemplateBase<coverage::CounterExpression>
    *LLVMRustCoverageSmallVectorCounterExpressionCreate() {
  return new SmallVector<coverage::CounterExpression, 32>();
}

extern "C" void LLVMRustCoverageSmallVectorCounterExpressionDispose(
    SmallVectorTemplateBase<coverage::CounterExpression> *Vector) {
  delete Vector;
}

extern "C" void LLVMRustCoverageSmallVectorCounterExpressionAdd(
    SmallVectorTemplateBase<coverage::CounterExpression> *Expressions,
    coverage::CounterExpression::ExprKind Kind,
    unsigned LeftIndex,
    unsigned RightIndex) {
  auto LHS = coverage::Counter::getCounter(LeftIndex);
  auto RHS = coverage::Counter::getCounter(RightIndex);
  Expressions->push_back(coverage::CounterExpression { Kind, LHS, RHS });
}

extern "C" SmallVectorTemplateBase<coverage::CounterMappingRegion>
    *LLVMRustCoverageSmallVectorCounterMappingRegionCreate() {
  return new SmallVector<coverage::CounterMappingRegion, 32>();
}

extern "C" void LLVMRustCoverageSmallVectorCounterMappingRegionDispose(
    SmallVectorTemplateBase<coverage::CounterMappingRegion> *Vector) {
  delete Vector;
}

extern "C" void LLVMRustCoverageSmallVectorCounterMappingRegionAdd(
    SmallVectorTemplateBase<coverage::CounterMappingRegion> *MappingRegions,
    unsigned Index,
    unsigned FileID,
    unsigned LineStart,
    unsigned ColumnStart,
    unsigned LineEnd,
    unsigned ColumnEnd) {
  auto Counter = coverage::Counter::getCounter(Index);
  MappingRegions->push_back(coverage::CounterMappingRegion::makeRegion(
           Counter, FileID, LineStart,
           ColumnStart, LineEnd, ColumnEnd));

  // FIXME(richkadel): As applicable, implement additional CounterMappingRegion types using the
  // static method alternatives to `coverage::CounterMappingRegion::makeRegion`:
  //
  //   makeExpansion(unsigned FileID, unsigned ExpandedFileID, unsigned LineStart,
  //                 unsigned ColumnStart, unsigned LineEnd, unsigned ColumnEnd) {
  //   makeSkipped(unsigned FileID, unsigned LineStart, unsigned ColumnStart,
  //               unsigned LineEnd, unsigned ColumnEnd) {
  //   makeGapRegion(Counter Count, unsigned FileID, unsigned LineStart,
  //                 unsigned ColumnStart, unsigned LineEnd, unsigned ColumnEnd) {
}

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
    const SmallVectorImpl<coverage::CounterExpression> *Expressions,
    SmallVectorImpl<coverage::CounterMappingRegion> *MappingRegions,
    RustStringRef BufferOut) {
  auto CoverageMappingWriter = coverage::CoverageMappingWriter(
    makeArrayRef(VirtualFileMappingIDs, NumVirtualFileMappingIDs),
    makeArrayRef(*Expressions),
    MutableArrayRef<coverage::CounterMappingRegion> { *MappingRegions });
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
