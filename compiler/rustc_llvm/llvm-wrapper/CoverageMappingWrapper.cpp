#include "LLVMWrapper.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/ProfileData/Coverage/CoverageMapping.h"
#include "llvm/ProfileData/Coverage/CoverageMappingWriter.h"
#include "llvm/ProfileData/InstrProf.h"

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

struct LLVMRustMCDCDecisionParameters {
  uint32_t BitmapIdx;
  uint16_t NumConditions;
};

struct LLVMRustMCDCBranchParameters {
  int16_t ConditionID;
  int16_t ConditionIDs[2];
};

static coverage::mcdc::BranchParameters
fromRust(LLVMRustMCDCBranchParameters Params) {
  return coverage::mcdc::BranchParameters(
      Params.ConditionID, {Params.ConditionIDs[0], Params.ConditionIDs[1]});
}

static coverage::mcdc::DecisionParameters
fromRust(LLVMRustMCDCDecisionParameters Params) {
  return coverage::mcdc::DecisionParameters(Params.BitmapIdx,
                                            Params.NumConditions);
}

// Must match the layout of
// `rustc_codegen_llvm::coverageinfo::ffi::CoverageSpan`.
struct LLVMRustCoverageSpan {
  uint32_t FileID;
  uint32_t LineStart;
  uint32_t ColumnStart;
  uint32_t LineEnd;
  uint32_t ColumnEnd;
};

// Must match the layout of `rustc_codegen_llvm::coverageinfo::ffi::CodeRegion`.
struct LLVMRustCoverageCodeRegion {
  LLVMRustCoverageSpan Span;
  LLVMRustCounter Count;
};

// Must match the layout of
// `rustc_codegen_llvm::coverageinfo::ffi::ExpansionRegion`.
struct LLVMRustCoverageExpansionRegion {
  LLVMRustCoverageSpan Span;
  uint32_t ExpandedFileID;
};

// Must match the layout of
// `rustc_codegen_llvm::coverageinfo::ffi::BranchRegion`.
struct LLVMRustCoverageBranchRegion {
  LLVMRustCoverageSpan Span;
  LLVMRustCounter TrueCount;
  LLVMRustCounter FalseCount;
};

// Must match the layout of
// `rustc_codegen_llvm::coverageinfo::ffi::MCDCBranchRegion`.
struct LLVMRustCoverageMCDCBranchRegion {
  LLVMRustCoverageSpan Span;
  LLVMRustCounter TrueCount;
  LLVMRustCounter FalseCount;
  LLVMRustMCDCBranchParameters MCDCBranchParams;
};

// Must match the layout of
// `rustc_codegen_llvm::coverageinfo::ffi::MCDCDecisionRegion`.
struct LLVMRustCoverageMCDCDecisionRegion {
  LLVMRustCoverageSpan Span;
  LLVMRustMCDCDecisionParameters MCDCDecisionParams;
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

extern "C" void LLVMRustCoverageWriteFilenamesToBuffer(
    const char *const Filenames[], size_t FilenamesLen, // String start pointers
    const size_t *const Lengths, size_t LengthsLen,     // Corresponding lengths
    RustStringRef BufferOut) {
  if (FilenamesLen != LengthsLen) {
    report_fatal_error(
        "Mismatched lengths in LLVMRustCoverageWriteFilenamesToBuffer");
  }

  SmallVector<std::string, 32> FilenameRefs;
  FilenameRefs.reserve(FilenamesLen);
  for (size_t i = 0; i < FilenamesLen; i++) {
    FilenameRefs.emplace_back(Filenames[i], Lengths[i]);
  }
  auto FilenamesWriter = coverage::CoverageFilenamesSectionWriter(
      ArrayRef<std::string>(FilenameRefs));
  auto OS = RawRustStringOstream(BufferOut);
  FilenamesWriter.write(OS);
}

extern "C" void LLVMRustCoverageWriteFunctionMappingsToBuffer(
    const unsigned *VirtualFileMappingIDs, size_t NumVirtualFileMappingIDs,
    const LLVMRustCounterExpression *RustExpressions, size_t NumExpressions,
    const LLVMRustCoverageCodeRegion *CodeRegions, size_t NumCodeRegions,
    const LLVMRustCoverageExpansionRegion *ExpansionRegions,
    size_t NumExpansionRegions,
    const LLVMRustCoverageBranchRegion *BranchRegions, size_t NumBranchRegions,
    const LLVMRustCoverageMCDCBranchRegion *MCDCBranchRegions,
    size_t NumMCDCBranchRegions,
    const LLVMRustCoverageMCDCDecisionRegion *MCDCDecisionRegions,
    size_t NumMCDCDecisionRegions, RustStringRef BufferOut) {
  // Convert from FFI representation to LLVM representation.

  // Expressions:
  std::vector<coverage::CounterExpression> Expressions;
  Expressions.reserve(NumExpressions);
  for (const auto &Expression :
       ArrayRef<LLVMRustCounterExpression>(RustExpressions, NumExpressions)) {
    Expressions.emplace_back(fromRust(Expression.Kind),
                             fromRust(Expression.LHS),
                             fromRust(Expression.RHS));
  }

  std::vector<coverage::CounterMappingRegion> MappingRegions;
  MappingRegions.reserve(NumCodeRegions + NumBranchRegions +
                         NumMCDCBranchRegions + NumMCDCDecisionRegions);

  // Code regions:
  for (const auto &Region : ArrayRef(CodeRegions, NumCodeRegions)) {
    MappingRegions.push_back(coverage::CounterMappingRegion::makeRegion(
        fromRust(Region.Count), Region.Span.FileID, Region.Span.LineStart,
        Region.Span.ColumnStart, Region.Span.LineEnd, Region.Span.ColumnEnd));
  }

  // Expansion regions:
  for (const auto &Region : ArrayRef(ExpansionRegions, NumExpansionRegions)) {
    MappingRegions.push_back(coverage::CounterMappingRegion::makeExpansion(
        Region.Span.FileID, Region.ExpandedFileID, Region.Span.LineStart,
        Region.Span.ColumnStart, Region.Span.LineEnd, Region.Span.ColumnEnd));
  }

  // Branch regions:
  for (const auto &Region : ArrayRef(BranchRegions, NumBranchRegions)) {
    MappingRegions.push_back(coverage::CounterMappingRegion::makeBranchRegion(
        fromRust(Region.TrueCount), fromRust(Region.FalseCount),
        Region.Span.FileID, Region.Span.LineStart, Region.Span.ColumnStart,
        Region.Span.LineEnd, Region.Span.ColumnEnd));
  }

  // MC/DC branch regions:
  for (const auto &Region : ArrayRef(MCDCBranchRegions, NumMCDCBranchRegions)) {
    MappingRegions.push_back(coverage::CounterMappingRegion::makeBranchRegion(
        fromRust(Region.TrueCount), fromRust(Region.FalseCount),
        Region.Span.FileID, Region.Span.LineStart, Region.Span.ColumnStart,
        Region.Span.LineEnd, Region.Span.ColumnEnd,
        fromRust(Region.MCDCBranchParams)));
  }

  // MC/DC decision regions:
  for (const auto &Region :
       ArrayRef(MCDCDecisionRegions, NumMCDCDecisionRegions)) {
    MappingRegions.push_back(coverage::CounterMappingRegion::makeDecisionRegion(
        fromRust(Region.MCDCDecisionParams), Region.Span.FileID,
        Region.Span.LineStart, Region.Span.ColumnStart, Region.Span.LineEnd,
        Region.Span.ColumnEnd));
  }

  // Write the converted expressions and mappings to a byte buffer.
  auto CoverageMappingWriter = coverage::CoverageMappingWriter(
      ArrayRef<unsigned>(VirtualFileMappingIDs, NumVirtualFileMappingIDs),
      Expressions, MappingRegions);
  auto OS = RawRustStringOstream(BufferOut);
  CoverageMappingWriter.write(OS);
}

extern "C" LLVMValueRef
LLVMRustCoverageCreatePGOFuncNameVar(LLVMValueRef F, const char *FuncName,
                                     size_t FuncNameLen) {
  auto FuncNameRef = StringRef(FuncName, FuncNameLen);
  return wrap(createPGOFuncNameVar(*cast<Function>(unwrap(F)), FuncNameRef));
}

extern "C" uint64_t LLVMRustCoverageHashBytes(const char *Bytes,
                                              size_t NumBytes) {
  return IndexedInstrProf::ComputeHash(StringRef(Bytes, NumBytes));
}

// Private helper function for getting the covmap and covfun section names.
static void writeInstrProfSectionNameToString(LLVMModuleRef M,
                                              InstrProfSectKind SectKind,
                                              RustStringRef OutStr) {
  auto TargetTriple = Triple(unwrap(M)->getTargetTriple());
  auto name = getInstrProfSectionName(SectKind, TargetTriple.getObjectFormat());
  auto OS = RawRustStringOstream(OutStr);
  OS << name;
}

extern "C" void
LLVMRustCoverageWriteCovmapSectionNameToString(LLVMModuleRef M,
                                               RustStringRef OutStr) {
  writeInstrProfSectionNameToString(M, IPSK_covmap, OutStr);
}

extern "C" void
LLVMRustCoverageWriteCovfunSectionNameToString(LLVMModuleRef M,
                                               RustStringRef OutStr) {
  writeInstrProfSectionNameToString(M, IPSK_covfun, OutStr);
}

extern "C" void
LLVMRustCoverageWriteCovmapVarNameToString(RustStringRef OutStr) {
  auto name = getCoverageMappingVarName();
  auto OS = RawRustStringOstream(OutStr);
  OS << name;
}

extern "C" uint32_t LLVMRustCoverageMappingVersion() {
  // This should always be `CurrentVersion`, because that's the version LLVM
  // will use when encoding the data we give it. If for some reason we ever
  // want to override the version number we _emit_, do it on the Rust side.
  return coverage::CovMapVersion::CurrentVersion;
}
