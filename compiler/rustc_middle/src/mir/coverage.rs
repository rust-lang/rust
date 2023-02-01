//! Metadata from source code coverage analysis and instrumentation.

use rustc_macros::HashStable;
use rustc_span::Symbol;

use std::fmt::{self, Debug, Formatter};

rustc_index::newtype_index! {
    /// An ExpressionOperandId value is assigned directly from either a
    /// CounterValueReference.as_u32() (which ascend from 1) or an ExpressionOperandId.as_u32()
    /// (which _*descend*_ from u32::MAX). Id value `0` (zero) represents a virtual counter with a
    /// constant value of `0`.
    #[derive(HashStable)]
    #[max = 0xFFFF_FFFF]
    #[debug_format = "ExpressionOperandId({})"]
    pub struct ExpressionOperandId {
    }
}

impl ExpressionOperandId {
    /// An expression operand for a "zero counter", as described in the following references:
    ///
    /// * <https://github.com/rust-lang/llvm-project/blob/rustc/13.0-2021-09-30/llvm/docs/CoverageMappingFormat.rst#counter>
    /// * <https://github.com/rust-lang/llvm-project/blob/rustc/13.0-2021-09-30/llvm/docs/CoverageMappingFormat.rst#tag>
    /// * <https://github.com/rust-lang/llvm-project/blob/rustc/13.0-2021-09-30/llvm/docs/CoverageMappingFormat.rst#counter-expressions>
    ///
    /// This operand can be used to count two or more separate code regions with a single counter,
    /// if they run sequentially with no branches, by injecting the `Counter` in a `BasicBlock` for
    /// one of the code regions, and inserting `CounterExpression`s ("add ZERO to the counter") in
    /// the coverage map for the other code regions.
    pub const ZERO: Self = Self::from_u32(0);
}

rustc_index::newtype_index! {
    #[derive(HashStable)]
    #[max = 0xFFFF_FFFF]
    #[debug_format = "CounterValueReference({})"]
    pub struct CounterValueReference {}
}

impl CounterValueReference {
    /// Counters start at 1 to reserve 0 for ExpressionOperandId::ZERO.
    pub const START: Self = Self::from_u32(1);

    /// Returns explicitly-requested zero-based version of the counter id, used
    /// during codegen. LLVM expects zero-based indexes.
    pub fn zero_based_index(self) -> u32 {
        let one_based_index = self.as_u32();
        debug_assert!(one_based_index > 0);
        one_based_index - 1
    }
}

rustc_index::newtype_index! {
    /// InjectedExpressionId.as_u32() converts to ExpressionOperandId.as_u32()
    ///
    /// Values descend from u32::MAX.
    #[derive(HashStable)]
    #[max = 0xFFFF_FFFF]
    #[debug_format = "InjectedExpressionId({})"]
    pub struct InjectedExpressionId {}
}

rustc_index::newtype_index! {
    /// InjectedExpressionIndex.as_u32() translates to u32::MAX - ExpressionOperandId.as_u32()
    ///
    /// Values ascend from 0.
    #[derive(HashStable)]
    #[max = 0xFFFF_FFFF]
    #[debug_format = "InjectedExpressionIndex({})"]
    pub struct InjectedExpressionIndex {}
}

rustc_index::newtype_index! {
    /// MappedExpressionIndex values ascend from zero, and are recalculated indexes based on their
    /// array position in the LLVM coverage map "Expressions" array, which is assembled during the
    /// "mapgen" process. They cannot be computed algorithmically, from the other `newtype_index`s.
    #[derive(HashStable)]
    #[max = 0xFFFF_FFFF]
    #[debug_format = "MappedExpressionIndex({})"]
    pub struct MappedExpressionIndex {}
}

impl From<CounterValueReference> for ExpressionOperandId {
    #[inline]
    fn from(v: CounterValueReference) -> ExpressionOperandId {
        ExpressionOperandId::from(v.as_u32())
    }
}

impl From<InjectedExpressionId> for ExpressionOperandId {
    #[inline]
    fn from(v: InjectedExpressionId) -> ExpressionOperandId {
        ExpressionOperandId::from(v.as_u32())
    }
}

#[derive(Clone, PartialEq, TyEncodable, TyDecodable, Hash, HashStable, TypeFoldable, TypeVisitable)]
pub enum CoverageKind {
    Counter {
        function_source_hash: u64,
        id: CounterValueReference,
    },
    Expression {
        id: InjectedExpressionId,
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
            Unreachable => bug!("Unreachable coverage cannot be part of an expression"),
        }
    }

    pub fn is_expression(&self) -> bool {
        matches!(self, Self::Expression { .. })
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
                match op {
                    Op::Add => "+",
                    Op::Subtract => "-",
                },
                rhs.index(),
            ),
            Unreachable => write!(fmt, "Unreachable"),
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
