#include "LLVMWrapper.h"
#include "llvm/ProfileData/Coverage/CoverageMapping.h"
#include "llvm/ProfileData/Coverage/CoverageMappingWriter.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/ADT/ArrayRef.h"

#include <iostream>

using namespace llvm;

// FFI equivalent of enum `llvm::coverage::Counter::CounterKind`
// https://github.com/rust-lang/llvm-project/blob/ea6fa9c2/llvm/include/llvm/ProfileData/Coverage/CoverageMapping.h#L97-L99
enum class LLVMRustCounterKind {
  Zero = 0,
  CounterValueReference = 1,
  Expression = 2,
};

// FFI equivalent of struct `llvm::coverage::Counter`
// https://github.com/rust-lang/llvm-project/blob/ea6fa9c2/llvm/include/llvm/ProfileData/Coverage/CoverageMapping.h#L94-L149
struct LLVMRustCounter {
  LLVMRustCounterKind CounterKind;
  uint32_t ID;
};

static coverage::Counter fromRust(LLVMRustCounter Counter) {
  switch (Counter.CounterKind) {
  case LLVMRustCounterKind::Zero:
    return coverage::Counter::getZero();
  case LLVMRustCounterKind::CounterValueReference:
    return coverage::Counter::getCounter(Counter.ID);
  case LLVMRustCounterKind::Expression:
    return coverage::Counter::getExpression(Counter.ID);
  }
  report_fatal_error("Bad LLVMRustCounterKind!");
}

// FFI equivalent of enum `llvm::coverage::CounterMappingRegion::RegionKind`
// https://github.com/rust-lang/llvm-project/blob/ea6fa9c2/llvm/include/llvm/ProfileData/Coverage/CoverageMapping.h#L213-L234
enum class LLVMRustCounterMappingRegionKind {
  CodeRegion = 0,
  ExpansionRegion = 1,
  SkippedRegion = 2,
  GapRegion = 3,
  BranchRegion = 4,
};

static coverage::CounterMappingRegion::RegionKind
fromRust(LLVMRustCounterMappingRegionKind Kind) {
  switch (Kind) {
  case LLVMRustCounterMappingRegionKind::CodeRegion:
    return coverage::CounterMappingRegion::CodeRegion;
  case LLVMRustCounterMappingRegionKind::ExpansionRegion:
    return coverage::CounterMappingRegion::ExpansionRegion;
  case LLVMRustCounterMappingRegionKind::SkippedRegion:
    return coverage::CounterMappingRegion::SkippedRegion;
  case LLVMRustCounterMappingRegionKind::GapRegion:
    return coverage::CounterMappingRegion::GapRegion;
  case LLVMRustCounterMappingRegionKind::BranchRegion:
    return coverage::CounterMappingRegion::BranchRegion;
  }
  report_fatal_error("Bad LLVMRustCounterMappingRegionKind!");
}

// FFI equivalent of struct `llvm::coverage::CounterMappingRegion`
// https://github.com/rust-lang/llvm-project/blob/ea6fa9c2/llvm/include/llvm/ProfileData/Coverage/CoverageMapping.h#L211-L304
struct LLVMRustCounterMappingRegion {
  LLVMRustCounter Count;
  LLVMRustCounter FalseCount;
  uint32_t FileID;
  uint32_t ExpandedFileID;
  uint32_t LineStart;
  uint32_t ColumnStart;
  uint32_t LineEnd;
  uint32_t ColumnEnd;
  LLVMRustCounterMappingRegionKind Kind;
};

// FFI equivalent of enum `llvm::coverage::CounterExpression::ExprKind`
// https://github.com/rust-lang/llvm-project/blob/ea6fa9c2/llvm/include/llvm/ProfileData/Coverage/CoverageMapping.h#L154
enum class LLVMRustCounterExprKind {
  Subtract = 0,
  Add = 1,
};

// FFI equivalent of struct `llvm::coverage::CounterExpression`
// https://github.com/rust-lang/llvm-project/blob/ea6fa9c2/llvm/include/llvm/ProfileData/Coverage/CoverageMapping.h#L151-L160
struct LLVMRustCounterExpression {
  LLVMRustCounterExprKind Kind;
  LLVMRustCounter LHS;
  LLVMRustCounter RHS;
};

static coverage::CounterExpression::ExprKind
fromRust(LLVMRustCounterExprKind Kind) {
  switch (Kind) {
  case LLVMRustCounterExprKind::Subtract:
    return coverage::CounterExpression::Subtract;
  case LLVMRustCounterExprKind::Add:
    return coverage::CounterExpression::Add;
  }
  report_fatal_error("Bad LLVMRustCounterExprKind!");
}

extern "C" void LLVMRustCoverageWriteFilenamesSectionToBuffer(
    const char* const Filenames[],
    size_t FilenamesLen,
    RustStringRef BufferOut) {
  SmallVector<std::string,32> FilenameRefs;
  for (size_t i = 0; i < FilenamesLen; i++) {
    FilenameRefs.push_back(std::string(Filenames[i]));
  }
  auto FilenamesWriter =
      coverage::CoverageFilenamesSectionWriter(ArrayRef<std::string>(FilenameRefs));
  RawRustStringOstream OS(BufferOut);
  FilenamesWriter.write(OS);
}

extern "C" void LLVMRustCoverageWriteMappingToBuffer(
    const unsigned *VirtualFileMappingIDs,
    unsigned NumVirtualFileMappingIDs,
    const LLVMRustCounterExpression *RustExpressions,
    unsigned NumExpressions,
    const LLVMRustCounterMappingRegion *RustMappingRegions,
    unsigned NumMappingRegions,
    RustStringRef BufferOut) {
  // Convert from FFI representation to LLVM representation.
  SmallVector<coverage::CounterMappingRegion, 0> MappingRegions;
  MappingRegions.reserve(NumMappingRegions);
  for (const auto &Region : ArrayRef<LLVMRustCounterMappingRegion>(
           RustMappingRegions, NumMappingRegions)) {
    MappingRegions.emplace_back(
        fromRust(Region.Count), fromRust(Region.FalseCount),
        Region.FileID, Region.ExpandedFileID,
        Region.LineStart, Region.ColumnStart, Region.LineEnd, Region.ColumnEnd,
        fromRust(Region.Kind));
  }

  std::vector<coverage::CounterExpression> Expressions;
  Expressions.reserve(NumExpressions);
  for (const auto &Expression :
       ArrayRef<LLVMRustCounterExpression>(RustExpressions, NumExpressions)) {
    Expressions.emplace_back(fromRust(Expression.Kind),
                             fromRust(Expression.LHS),
                             fromRust(Expression.RHS));
  }

  auto CoverageMappingWriter = coverage::CoverageMappingWriter(
      ArrayRef<unsigned>(VirtualFileMappingIDs, NumVirtualFileMappingIDs),
      Expressions,
      MappingRegions);
  RawRustStringOstream OS(BufferOut);
  CoverageMappingWriter.write(OS);
}

extern "C" LLVMValueRef LLVMRustCoverageCreatePGOFuncNameVar(LLVMValueRef F, const char *FuncName) {
  StringRef FuncNameRef(FuncName);
  return wrap(createPGOFuncNameVar(*cast<Function>(unwrap(F)), FuncNameRef));
}

extern "C" uint64_t LLVMRustCoverageHashByteArray(
    const char *Bytes,
    size_t NumBytes) {
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
  WriteSectionNameToString(M, IPSK_covfun, Str);
}

extern "C" void LLVMRustCoverageWriteMappingVarNameToString(RustStringRef Str) {
  auto name = getCoverageMappingVarName();
  RawRustStringOstream OS(Str);
  OS << name;
}

extern "C" uint32_t LLVMRustCoverageMappingVersion() {
  return coverage::CovMapVersion::Version6;
}
