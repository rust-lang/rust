//! Metadata from source code coverage analysis and instrumentation.

use rustc_index::IndexVec;
use rustc_macros::HashStable;
use rustc_span::{Span, Symbol};

use std::fmt::{self, Debug, Formatter};

rustc_index::newtype_index! {
    /// Used by [`CoverageKind::BlockMarker`] to mark blocks during THIR-to-MIR
    /// lowering, so that those blocks can be identified later.
    #[derive(HashStable)]
    #[encodable]
    #[debug_format = "BlockMarkerId({})"]
    pub struct BlockMarkerId {}
}

// DecisionMarkerId and ConditionMarkerId are used between THIR and MIR representations.
// DecisionId and ConditionId are used between MIR representation and LLVM-IR.

rustc_index::newtype_index! {
    #[derive(HashStable)]
    #[encodable]
    #[debug_format = "DecisionMarkerId({})"]
    pub struct DecisionMarkerId {}
}

rustc_index::newtype_index! {
    #[derive(HashStable)]
    #[encodable]
    #[debug_format = "ConditionMarkerId({})"]
    pub struct ConditionMarkerId {}
}

rustc_index::newtype_index! {
    #[derive(HashStable)]
    #[encodable]
    #[debug_format = "DecisionId({})"]
    pub struct DecisionId {}
}

rustc_index::newtype_index! {
    #[derive(HashStable)]
    #[encodable]
    #[debug_format = "ConditionId({})"]
    pub struct ConditionId {}
}

rustc_index::newtype_index! {
    /// ID of a coverage counter. Values ascend from 0.
    ///
    /// Before MIR inlining, counter IDs are local to their enclosing function.
    /// After MIR inlining, coverage statements may have been inlined into
    /// another function, so use the statement's source-scope to find which
    /// function/instance its IDs are meaningful for.
    ///
    /// Note that LLVM handles counter IDs as `uint32_t`, so there is no need
    /// to use a larger representation on the Rust side.
    #[derive(HashStable)]
    #[encodable]
    #[orderable]
    #[max = 0xFFFF_FFFF]
    #[debug_format = "CounterId({})"]
    pub struct CounterId {}
}

impl CounterId {
    pub const START: Self = Self::from_u32(0);
}

rustc_index::newtype_index! {
    /// ID of a coverage-counter expression. Values ascend from 0.
    ///
    /// Before MIR inlining, expression IDs are local to their enclosing function.
    /// After MIR inlining, coverage statements may have been inlined into
    /// another function, so use the statement's source-scope to find which
    /// function/instance its IDs are meaningful for.
    ///
    /// Note that LLVM handles expression IDs as `uint32_t`, so there is no need
    /// to use a larger representation on the Rust side.
    #[derive(HashStable)]
    #[encodable]
    #[orderable]
    #[max = 0xFFFF_FFFF]
    #[debug_format = "ExpressionId({})"]
    pub struct ExpressionId {}
}

impl ExpressionId {
    pub const START: Self = Self::from_u32(0);
}

/// Enum that can hold a constant zero value, the ID of an physical coverage
/// counter, or the ID of a coverage-counter expression.
///
/// This was originally only used for expression operands (and named `Operand`),
/// but the zero/counter/expression distinction is also useful for representing
/// the value of code/gap mappings, and the true/false arms of branch mappings.
#[derive(Copy, Clone, PartialEq, Eq)]
#[derive(TyEncodable, TyDecodable, Hash, HashStable, TypeFoldable, TypeVisitable)]
pub enum CovTerm {
    Zero,
    Counter(CounterId),
    Expression(ExpressionId),
}

impl Debug for CovTerm {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Zero => write!(f, "Zero"),
            Self::Counter(id) => f.debug_tuple("Counter").field(&id.as_u32()).finish(),
            Self::Expression(id) => f.debug_tuple("Expression").field(&id.as_u32()).finish(),
        }
    }
}

#[derive(Clone, PartialEq, TyEncodable, TyDecodable, Hash, HashStable, TypeFoldable, TypeVisitable)]
pub enum CoverageKind {
    /// Marks a span that might otherwise not be represented in MIR, so that
    /// coverage instrumentation can associate it with its enclosing block/BCB.
    ///
    /// Should be erased before codegen (at some point after `InstrumentCoverage`).
    SpanMarker,

    /// Marks its enclosing basic block with an ID that can be referred to by
    /// side data in [`BranchInfo`].
    ///
    /// Should be erased before codegen (at some point after `InstrumentCoverage`).
    BlockMarker { id: BlockMarkerId },

    /// Marks the first condition of a decision (boolean expression). All
    /// conditions in the same decision will reference this id.
    ///
    /// Has no effect during codegen.
    MCDCDecisionEntryMarker { decm_id: DecisionMarkerId },

    /// Marks one of the basic blocks following the decision referenced with `id`.
    /// the outcome bool designates which branch of the decision it is:
    /// `true` for the then block, `false` for the else block.
    ///
    /// Has no effect during codegen.
    MCDCDecisionOutputMarker { decm_id: DecisionMarkerId, outcome: bool },

    /// Marks a basic block where a condition evaluation occurs
    /// The block may end with a SwitchInt where the 2 successors BBs have a
    /// MCDCConditionOutcomeMarker statement with a matching ID.
    ///
    /// Has no effect during codegen.
    MCDCConditionEntryMarker { decm_id: DecisionMarkerId, condm_id: ConditionMarkerId },

    /// Marks a basic block that is branched from a condition evaluation.
    /// The block may have a predecessor with a matching ID.
    ///
    /// Has no effect during codegen.
    MCDCConditionOutputMarker {
        decm_id: DecisionMarkerId,
        condm_id: ConditionMarkerId,
        outcome: bool,
    },

    /// Marks the point in MIR control flow represented by a coverage counter.
    ///
    /// This is eventually lowered to `llvm.instrprof.increment` in LLVM IR.
    ///
    /// If this statement does not survive MIR optimizations, any mappings that
    /// refer to this counter can have those references simplified to zero.
    CounterIncrement { id: CounterId },

    /// Marks the point in MIR control-flow represented by a coverage expression.
    ///
    /// If this statement does not survive MIR optimizations, any mappings that
    /// refer to this expression can have those references simplified to zero.
    ///
    /// (This is only inserted for expression IDs that are directly used by
    /// mappings. Intermediate expressions with no direct mappings are
    /// retained/zeroed based on whether they are transitively used.)
    ExpressionUsed { id: ExpressionId },

    /// Declares the number of bytes needed to store the test-vector bitmaps of
    /// all the decisions in the function body.
    ///
    /// In LLVM backend, this is done by inserting a call to the
    /// `instrprof.mcdc.parameters` intrinsic.
    MCDCBitmapRequire { needed_bytes: u32 },

    /// Marks a point where the condition bitmap should be set to 0.
    MCDCCondBitmapReset,

    /// Marks a point where a bit of the condition bitmap should be set.
    MCDCCondBitmapUpdate { condition_id: u32, bool_value: bool },

    /// Marks a point where a bit of the global Test Vector bitmap should be set to one.
    MCDCTestVectorBitmapUpdate { needed_bytes: u32, decision_index: u32 },
}

impl Debug for CoverageKind {
    fn fmt(&self, fmt: &mut Formatter<'_>) -> fmt::Result {
        use CoverageKind::*;
        match self {
            SpanMarker => write!(fmt, "SpanMarker"),
            BlockMarker { id } => write!(fmt, "BlockMarker({:?})", id.index()),
            MCDCDecisionEntryMarker { decm_id: id } => {
                write!(fmt, "MCDCDecisionEntryMarker({:?})", id.index())
            }
            MCDCDecisionOutputMarker { decm_id: id, outcome } => {
                write!(fmt, "MCDCDecisionOutputMarker({:?}, {})", id.index(), outcome)
            }
            MCDCConditionEntryMarker { decm_id: decision_id, condm_id: id } => {
                write!(fmt, "MCDCConditionEntryMarker({:?}, {:?})", decision_id.index(), id.index())
            }
            MCDCConditionOutputMarker { decm_id: decision_marker_id, condm_id: id, outcome } => {
                write!(
                    fmt,
                    "MCDCConditionOutcomeMarker({:?}, {:?}, {})",
                    decision_marker_id.index(),
                    id.index(),
                    outcome
                )
            }
            CounterIncrement { id } => write!(fmt, "CounterIncrement({:?})", id.index()),
            ExpressionUsed { id } => write!(fmt, "ExpressionUsed({:?})", id.index()),
            MCDCBitmapRequire { needed_bytes } => {
                write!(fmt, "MCDCBitmapRequire({needed_bytes} bytes)")
            }
            MCDCCondBitmapReset => {
                write!(fmt, "MCDCCondBitmapReset()")
            }
            MCDCCondBitmapUpdate { condition_id, bool_value } => {
                write!(fmt, "MCDCCondBitmapUpdate({condition_id}, {bool_value})")
            }
            MCDCTestVectorBitmapUpdate { needed_bytes, decision_index } => {
                write!(fmt, "MCDCTVBitmapUpdate({needed_bytes} bytes, {decision_index})")
            }
        }
    }
}

#[derive(Clone, TyEncodable, TyDecodable, Hash, HashStable, PartialEq, Eq, PartialOrd, Ord)]
#[derive(TypeFoldable, TypeVisitable)]
pub struct CodeRegion {
    pub file_name: Symbol,
    pub start_line: u32,
    pub start_col: u32,
    pub end_line: u32,
    pub end_col: u32,
}

impl Debug for CodeRegion {
    fn fmt(&self, fmt: &mut Formatter<'_>) -> fmt::Result {
        write!(
            fmt,
            "{}:{}:{} - {}:{}",
            self.file_name, self.start_line, self.start_col, self.end_line, self.end_col
        )
    }
}

#[derive(Copy, Clone, Debug, PartialEq, TyEncodable, TyDecodable, Hash, HashStable)]
#[derive(TypeFoldable, TypeVisitable)]
pub enum Op {
    Subtract,
    Add,
}

impl Op {
    pub fn is_add(&self) -> bool {
        matches!(self, Self::Add)
    }

    pub fn is_subtract(&self) -> bool {
        matches!(self, Self::Subtract)
    }
}

#[derive(Clone, Debug)]
#[derive(TyEncodable, TyDecodable, Hash, HashStable, TypeFoldable, TypeVisitable)]
pub struct Expression {
    pub lhs: CovTerm,
    pub op: Op,
    pub rhs: CovTerm,
}

#[derive(Clone, Debug)]
#[derive(TyEncodable, TyDecodable, Hash, HashStable, TypeFoldable, TypeVisitable)]
pub enum MappingKind {
    /// Associates a normal region of code with a counter/expression/zero.
    Code(CovTerm),
    /// Associates a branch region with separate counters for true and false.
    Branch { true_term: CovTerm, false_term: CovTerm },
}

impl MappingKind {
    /// Iterator over all coverage terms in this mapping kind.
    pub fn terms(&self) -> impl Iterator<Item = CovTerm> {
        let one = |a| std::iter::once(a).chain(None);
        let two = |a, b| std::iter::once(a).chain(Some(b));
        match *self {
            Self::Code(term) => one(term),
            Self::Branch { true_term, false_term } => two(true_term, false_term),
        }
    }

    /// Returns a copy of this mapping kind, in which all coverage terms have
    /// been replaced with ones returned by the given function.
    pub fn map_terms(&self, map_fn: impl Fn(CovTerm) -> CovTerm) -> Self {
        match *self {
            Self::Code(term) => Self::Code(map_fn(term)),
            Self::Branch { true_term, false_term } => {
                Self::Branch { true_term: map_fn(true_term), false_term: map_fn(false_term) }
            }
        }
    }
}

#[derive(Clone, Debug)]
#[derive(TyEncodable, TyDecodable, Hash, HashStable, TypeFoldable, TypeVisitable)]
pub struct Mapping {
    pub kind: MappingKind,
    pub code_region: CodeRegion,
}

/// Stores per-function coverage information attached to a `mir::Body`,
/// to be used in conjunction with the individual coverage statements injected
/// into the function's basic blocks.
#[derive(Clone, Debug)]
#[derive(TyEncodable, TyDecodable, Hash, HashStable, TypeFoldable, TypeVisitable)]
pub struct FunctionCoverageInfo {
    pub function_source_hash: u64,
    pub num_counters: usize,

    pub expressions: IndexVec<ExpressionId, Expression>,
    pub mappings: Vec<Mapping>,
}

/// Branch information recorded during THIR-to-MIR lowering, and stored in MIR.
#[derive(Clone, Debug)]
#[derive(TyEncodable, TyDecodable, Hash, HashStable, TypeFoldable, TypeVisitable)]
pub struct BranchInfo {
    /// 1 more than the highest-numbered [`CoverageKind::BlockMarker`] that was
    /// injected into the MIR body. This makes it possible to allocate per-ID
    /// data structures without having to scan the entire body first.
    pub num_block_markers: usize,
    pub branch_spans: Vec<BranchSpan>,

    /// Associate a span for every decision in the function body.
    /// Empty if MCDC coverage is disabled.
    pub decision_spans: IndexVec<DecisionMarkerId, DecisionSpan>,
}

#[derive(Clone, Debug)]
#[derive(TyEncodable, TyDecodable, Hash, HashStable, TypeFoldable, TypeVisitable)]
pub struct DecisionSpan {
    /// Source code region associated to the decision.
    pub span: Span,
    /// Number of conditions in the decision.
    pub num_conditions: u32,
}

#[derive(Clone, Debug)]
#[derive(TyEncodable, TyDecodable, Hash, HashStable, TypeFoldable, TypeVisitable)]
pub struct BranchSpan {
    /// Source code region associated to the branch.
    pub span: Span,
    /// ID of Decision structure the branch is part of. Only used in
    /// the MCDC coverage.
    pub decision_id: DecisionMarkerId,
    pub true_marker: BlockMarkerId,
    pub false_marker: BlockMarkerId,
}
