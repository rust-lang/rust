#include "LLVMWrapper.h"
#include "llvm/ADT/ArrayRef.h"
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

// FFI equivalent of enum `llvm::coverage::CounterMappingRegion::RegionKind`
// https://github.com/rust-lang/llvm-project/blob/ea6fa9c2/llvm/include/llvm/ProfileData/Coverage/CoverageMapping.h#L213-L234
enum class LLVMRustCounterMappingRegionKind {
  CodeRegion = 0,
  ExpansionRegion = 1,
  SkippedRegion = 2,
  GapRegion = 3,
  BranchRegion = 4,
  MCDCDecisionRegion = 5,
  MCDCBranchRegion = 6
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
  case LLVMRustCounterMappingRegionKind::MCDCDecisionRegion:
    return coverage::CounterMappingRegion::MCDCDecisionRegion;
  case LLVMRustCounterMappingRegionKind::MCDCBranchRegion:
    return coverage::CounterMappingRegion::MCDCBranchRegion;
  }
  report_fatal_error("Bad LLVMRustCounterMappingRegionKind!");
}

enum LLVMRustMCDCParametersTag {
  None = 0,
  Decision = 1,
  Branch = 2,
};

struct LLVMRustMCDCDecisionParameters {
  uint32_t BitmapIdx;
  uint16_t NumConditions;
};

struct LLVMRustMCDCBranchParameters {
  int16_t ConditionID;
  int16_t ConditionIDs[2];
};

struct LLVMRustMCDCParameters {
  LLVMRustMCDCParametersTag Tag;
  LLVMRustMCDCDecisionParameters DecisionParameters;
  LLVMRustMCDCBranchParameters BranchParameters;
};

#if LLVM_VERSION_GE(19, 0)
static coverage::mcdc::Parameters fromRust(LLVMRustMCDCParameters Params) {
  switch (Params.Tag) {
  case LLVMRustMCDCParametersTag::None:
    return std::monostate();
  case LLVMRustMCDCParametersTag::Decision:
    return coverage::mcdc::DecisionParameters(
        Params.DecisionParameters.BitmapIdx,
        Params.DecisionParameters.NumConditions);
  case LLVMRustMCDCParametersTag::Branch:
    return coverage::mcdc::BranchParameters(
        static_cast<coverage::mcdc::ConditionID>(
            Params.BranchParameters.ConditionID),
        {static_cast<coverage::mcdc::ConditionID>(
             Params.BranchParameters.ConditionIDs[0]),
         static_cast<coverage::mcdc::ConditionID>(
             Params.BranchParameters.ConditionIDs[1])});
  }
  report_fatal_error("Bad LLVMRustMCDCParametersTag!");
}
#endif

// FFI equivalent of struct `llvm::coverage::CounterMappingRegion`
// https://github.com/rust-lang/llvm-project/blob/ea6fa9c2/llvm/include/llvm/ProfileData/Coverage/CoverageMapping.h#L211-L304
struct LLVMRustCounterMappingRegion {
  LLVMRustCounter Count;
  LLVMRustCounter FalseCount;
  LLVMRustMCDCParameters MCDCParameters;
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
    const char *const Filenames[], size_t FilenamesLen, // String start pointers
    const size_t *const Lengths, size_t LengthsLen,     // Corresponding lengths
    RustStringRef BufferOut) {
  if (FilenamesLen != LengthsLen) {
    report_fatal_error(
        "Mismatched lengths in LLVMRustCoverageWriteFilenamesSectionToBuffer");
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

extern "C" void LLVMRustCoverageWriteMappingToBuffer(
    const unsigned *VirtualFileMappingIDs, unsigned NumVirtualFileMappingIDs,
    const LLVMRustCounterExpression *RustExpressions, unsigned NumExpressions,
    const LLVMRustCounterMappingRegion *RustMappingRegions,
    unsigned NumMappingRegions, RustStringRef BufferOut) {
  // Convert from FFI representation to LLVM representation.
  SmallVector<coverage::CounterMappingRegion, 0> MappingRegions;
  MappingRegions.reserve(NumMappingRegions);
  for (const auto &Region : ArrayRef<LLVMRustCounterMappingRegion>(
           RustMappingRegions, NumMappingRegions)) {
    MappingRegions.emplace_back(
        fromRust(Region.Count), fromRust(Region.FalseCount),
#if LLVM_VERSION_LT(19, 0)
        coverage::CounterMappingRegion::MCDCParameters{},
#endif
        Region.FileID, Region.ExpandedFileID, // File IDs, then region info.
        Region.LineStart, Region.ColumnStart, Region.LineEnd, Region.ColumnEnd,
        fromRust(Region.Kind)
#if LLVM_VERSION_GE(19, 0)
            ,
        fromRust(Region.MCDCParameters)
#endif
    );
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

extern "C" uint64_t LLVMRustCoverageHashByteArray(const char *Bytes,
                                                  size_t NumBytes) {
  auto StrRef = StringRef(Bytes, NumBytes);
  return IndexedInstrProf::ComputeHash(StrRef);
}

static void WriteSectionNameToString(LLVMModuleRef M, InstrProfSectKind SK,
                                     RustStringRef Str) {
  auto TargetTriple = Triple(unwrap(M)->getTargetTriple());
  auto name = getInstrProfSectionName(SK, TargetTriple.getObjectFormat());
  auto OS = RawRustStringOstream(Str);
  OS << name;
}

extern "C" void LLVMRustCoverageWriteMapSectionNameToString(LLVMModuleRef M,
                                                            RustStringRef Str) {
  WriteSectionNameToString(M, IPSK_covmap, Str);
}

extern "C" void
LLVMRustCoverageWriteFuncSectionNameToString(LLVMModuleRef M,
                                             RustStringRef Str) {
  WriteSectionNameToString(M, IPSK_covfun, Str);
}

extern "C" void LLVMRustCoverageWriteMappingVarNameToString(RustStringRef Str) {
  auto name = getCoverageMappingVarName();
  auto OS = RawRustStringOstream(Str);
  OS << name;
}

extern "C" uint32_t LLVMRustCoverageMappingVersion() {
  // This should always be `CurrentVersion`, because that's the version LLVM
  // will use when encoding the data we give it. If for some reason we ever
  // want to override the version number we _emit_, do it on the Rust side.
  return coverage::CovMapVersion::CurrentVersion;
}
