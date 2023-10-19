//! Metadata from source code coverage analysis and instrumentation.

use rustc_index::IndexVec;
use rustc_macros::HashStable;
use rustc_span::Symbol;

use std::fmt::{self, Debug, Formatter};

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
}

impl Debug for CoverageKind {
    fn fmt(&self, fmt: &mut Formatter<'_>) -> fmt::Result {
        use CoverageKind::*;
        match self {
            CounterIncrement { id } => write!(fmt, "CounterIncrement({:?})", id.index()),
            ExpressionUsed { id } => write!(fmt, "ExpressionUsed({:?})", id.index()),
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
pub struct Mapping {
    pub code_region: CodeRegion,

    /// Indicates whether this mapping uses a counter value, expression value,
    /// or zero value.
    ///
    /// FIXME: When we add support for mapping kinds other than `Code`
    /// (e.g. branch regions, expansion regions), replace this with a dedicated
    /// mapping-kind enum.
    pub term: CovTerm,
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
