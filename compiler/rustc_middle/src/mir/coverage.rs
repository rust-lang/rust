//! Metadata from source code coverage analysis and instrumentation.

use rustc_index::IndexVec;
use rustc_macros::HashStable;
use rustc_span::Symbol;

use std::fmt::{self, Debug, Formatter};

rustc_index::newtype_index! {
    /// ID of a coverage counter. Values ascend from 0.
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

    #[inline(always)]
    pub fn next_id(self) -> Self {
        Self::from_u32(self.as_u32() + 1)
    }
}

rustc_index::newtype_index! {
    /// ID of a coverage-counter expression. Values ascend from 0.
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

    #[inline(always)]
    pub fn next_id(self) -> Self {
        Self::from_u32(self.as_u32() + 1)
    }
}

/// Operand of a coverage-counter expression.
///
/// Operands can be a constant zero value, an actual coverage counter, or another
/// expression. Counter/expression operands are referred to by ID.
#[derive(Copy, Clone, PartialEq, Eq)]
#[derive(TyEncodable, TyDecodable, Hash, HashStable, TypeFoldable, TypeVisitable)]
pub enum Operand {
    Zero,
    Counter(CounterId),
    Expression(ExpressionId),
}

impl Debug for Operand {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Zero => write!(f, "Zero"),
            Self::Counter(id) => f.debug_tuple("Counter").field(&id.as_u32()).finish(),
            Self::Expression(id) => f.debug_tuple("Expression").field(&id.as_u32()).finish(),
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

#[derive(Copy, Clone, Debug, PartialEq, Eq, TyEncodable, TyDecodable, Hash, HashStable)]
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

#[derive(Clone, TyEncodable, TyDecodable, Hash, HashStable, PartialEq, Eq)]
#[derive(TypeFoldable, TypeVisitable)]
pub struct CoverageExpression {
    /// ID of this coverage-counter expression within its enclosing function.
    /// Other expressions in the same function can refer to it as an operand.
    pub id: ExpressionId,
    pub lhs: Operand,
    pub op: Op,
    pub rhs: Operand,
}

impl Debug for CoverageExpression {
    fn fmt(&self, fmt: &mut Formatter<'_>) -> fmt::Result {
        let CoverageExpression { id, lhs, op, rhs } = self;
        write!(
            fmt,
            "Expression({:?}) = {:?} {} {:?}",
            id.index(),
            lhs,
            match op {
                Op::Add => "+",
                Op::Subtract => "-",
            },
            rhs,
        )
    }
}

#[derive(Clone, Debug, TyEncodable, TyDecodable, Hash, HashStable, PartialEq, Eq)]
#[derive(TypeFoldable, TypeVisitable)]
pub enum CoverageRegionKind {
    Code(Operand),
}

#[derive(Clone, Debug, TyEncodable, TyDecodable, Hash, HashStable, PartialEq, Eq)]
#[derive(TypeFoldable, TypeVisitable)]
pub struct CoverageRegion {
    pub kind: CoverageRegionKind,
    pub code_region: CodeRegion,
}

#[derive(Clone, Debug, TyEncodable, TyDecodable, Hash, HashStable, PartialEq, Eq)]
#[derive(TypeFoldable, TypeVisitable)]
pub struct CoverageInfo {
    pub num_counters: u32,
    pub expressions: IndexVec<ExpressionId, CoverageExpression>,
    pub regions: Vec<CoverageRegion>,
}
