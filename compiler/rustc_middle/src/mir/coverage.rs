//! Metadata from source code coverage analysis and instrumentation.

use rustc_macros::HashStable;
use rustc_span::Symbol;

use std::cmp::Ord;
use std::fmt::{self, Debug, Formatter};

rustc_index::newtype_index! {
    pub struct ExpressionOperandId {
        derive [HashStable]
        DEBUG_FORMAT = "ExpressionOperandId({})",
        MAX = 0xFFFF_FFFF,
    }
}

impl ExpressionOperandId {
    /// An expression operand for a "zero counter", as described in the following references:
    ///
    /// * https://github.com/rust-lang/llvm-project/blob/llvmorg-8.0.0/llvm/docs/CoverageMappingFormat.rst#counter
    /// * https://github.com/rust-lang/llvm-project/blob/llvmorg-8.0.0/llvm/docs/CoverageMappingFormat.rst#tag
    /// * https://github.com/rust-lang/llvm-project/blob/llvmorg-8.0.0/llvm/docs/CoverageMappingFormat.rst#counter-expressions
    ///
    /// This operand can be used to count two or more separate code regions with a single counter,
    /// if they run sequentially with no branches, by injecting the `Counter` in a `BasicBlock` for
    /// one of the code regions, and inserting `CounterExpression`s ("add ZERO to the counter") in
    /// the coverage map for the other code regions.
    pub const ZERO: Self = Self::from_u32(0);
}

rustc_index::newtype_index! {
    pub struct CounterValueReference {
        derive [HashStable]
        DEBUG_FORMAT = "CounterValueReference({})",
        MAX = 0xFFFF_FFFF,
    }
}

impl CounterValueReference {
    // Counters start at 1 to reserve 0 for ExpressionOperandId::ZERO.
    pub const START: Self = Self::from_u32(1);
}

rustc_index::newtype_index! {
    pub struct InjectedExpressionIndex {
        derive [HashStable]
        DEBUG_FORMAT = "InjectedExpressionIndex({})",
        MAX = 0xFFFF_FFFF,
    }
}

rustc_index::newtype_index! {
    pub struct MappedExpressionIndex {
        derive [HashStable]
        DEBUG_FORMAT = "MappedExpressionIndex({})",
        MAX = 0xFFFF_FFFF,
    }
}

impl From<CounterValueReference> for ExpressionOperandId {
    #[inline]
    fn from(v: CounterValueReference) -> ExpressionOperandId {
        ExpressionOperandId::from(v.as_u32())
    }
}

impl From<&mut CounterValueReference> for ExpressionOperandId {
    #[inline]
    fn from(v: &mut CounterValueReference) -> ExpressionOperandId {
        ExpressionOperandId::from(v.as_u32())
    }
}

impl From<InjectedExpressionIndex> for ExpressionOperandId {
    #[inline]
    fn from(v: InjectedExpressionIndex) -> ExpressionOperandId {
        ExpressionOperandId::from(v.as_u32())
    }
}

impl From<&mut InjectedExpressionIndex> for ExpressionOperandId {
    #[inline]
    fn from(v: &mut InjectedExpressionIndex) -> ExpressionOperandId {
        ExpressionOperandId::from(v.as_u32())
    }
}

#[derive(Clone, PartialEq, TyEncodable, TyDecodable, HashStable, TypeFoldable)]
pub enum CoverageKind {
    Counter {
        function_source_hash: u64,
        id: CounterValueReference,
    },
    Expression {
        id: InjectedExpressionIndex,
        lhs: ExpressionOperandId,
        op: Op,
        rhs: ExpressionOperandId,
    },
    Unreachable,
}

impl CoverageKind {
    pub fn as_operand_id(&self) -> ExpressionOperandId {
        use CoverageKind::*;
        match *self {
            Counter { id, .. } => ExpressionOperandId::from(id),
            Expression { id, .. } => ExpressionOperandId::from(id),
            Unreachable => {
                bug!("Unreachable coverage cannot be part of an expression")
            }
        }
    }

    pub fn is_counter(&self) -> bool {
        match self {
            Self::Counter { .. } => true,
            _ => false,
        }
    }

    pub fn is_expression(&self) -> bool {
        match self {
            Self::Expression { .. } => true,
            _ => false,
        }
    }

    pub fn is_unreachable(&self) -> bool {
        *self == Self::Unreachable
    }
}

impl Debug for CoverageKind {
    fn fmt(&self, fmt: &mut Formatter<'_>) -> fmt::Result {
        use CoverageKind::*;
        match self {
            Counter { id, .. } => write!(fmt, "Counter({:?})", id.index()),
            Expression { id, lhs, op, rhs } => write!(
                fmt,
                "Expression({:?}) = {} {} {}",
                id.index(),
                lhs.index(),
                if *op == Op::Add { "+" } else { "-" },
                rhs.index(),
            ),
            Unreachable => write!(fmt, "Unreachable"),
        }
    }
}

#[derive(Clone, TyEncodable, TyDecodable, HashStable, TypeFoldable, PartialEq, Eq, PartialOrd, Ord)]
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

#[derive(Copy, Clone, Debug, PartialEq, TyEncodable, TyDecodable, HashStable, TypeFoldable)]
pub enum Op {
    Subtract,
    Add,
}
