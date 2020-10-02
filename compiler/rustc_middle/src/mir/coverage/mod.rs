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

rustc_index::newtype_index! {
    pub struct CounterValueReference {
        derive [HashStable]
        DEBUG_FORMAT = "CounterValueReference({})",
        MAX = 0xFFFF_FFFF,
    }
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

impl From<InjectedExpressionIndex> for ExpressionOperandId {
    #[inline]
    fn from(v: InjectedExpressionIndex) -> ExpressionOperandId {
        ExpressionOperandId::from(v.as_u32())
    }
}

#[derive(Clone, Debug, PartialEq, TyEncodable, TyDecodable, HashStable, TypeFoldable)]
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
        match *self {
            CoverageKind::Counter { id, .. } => ExpressionOperandId::from(id),
            CoverageKind::Expression { id, .. } => ExpressionOperandId::from(id),
            CoverageKind::Unreachable => {
                bug!("Unreachable coverage cannot be part of an expression")
            }
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
