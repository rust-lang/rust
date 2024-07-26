use derive_where::derive_where;
use rustc_type_ir_macros::{TypeFoldable_Generic, TypeVisitable_Generic};

use crate::solve::NoSolution;
use crate::{self as ty, Interner};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[derive(TypeFoldable_Generic, TypeVisitable_Generic)]
pub struct ExpectedFound<T> {
    pub expected: T,
    pub found: T,
}

impl<T> ExpectedFound<T> {
    pub fn new(a_is_expected: bool, a: T, b: T) -> Self {
        if a_is_expected {
            ExpectedFound { expected: a, found: b }
        } else {
            ExpectedFound { expected: b, found: a }
        }
    }
}

// Data structures used in type unification
#[derive_where(Clone, Copy, PartialEq, Eq, Debug; I: Interner)]
#[derive(TypeVisitable_Generic)]
#[cfg_attr(feature = "nightly", rustc_pass_by_value)]
pub enum TypeError<I: Interner> {
    Mismatch,
    ConstnessMismatch(ExpectedFound<ty::BoundConstness>),
    PolarityMismatch(ExpectedFound<ty::PredicatePolarity>),
    SafetyMismatch(ExpectedFound<I::Safety>),
    AbiMismatch(ExpectedFound<I::Abi>),
    Mutability,
    ArgumentMutability(usize),
    TupleSize(ExpectedFound<usize>),
    FixedArraySize(ExpectedFound<u64>),
    ArgCount,

    RegionsDoesNotOutlive(I::Region, I::Region),
    RegionsInsufficientlyPolymorphic(I::BoundRegion, I::Region),
    RegionsPlaceholderMismatch,

    Sorts(ExpectedFound<I::Ty>),
    ArgumentSorts(ExpectedFound<I::Ty>, usize),
    Traits(ExpectedFound<I::DefId>),
    VariadicMismatch(ExpectedFound<bool>),

    /// Instantiating a type variable with the given type would have
    /// created a cycle (because it appears somewhere within that
    /// type).
    CyclicTy(I::Ty),
    CyclicConst(I::Const),
    ProjectionMismatched(ExpectedFound<I::DefId>),
    ExistentialMismatch(ExpectedFound<I::BoundExistentialPredicates>),
    ConstMismatch(ExpectedFound<I::Const>),

    IntrinsicCast,
    /// Safe `#[target_feature]` functions are not assignable to safe function pointers.
    TargetFeatureCast(I::DefId),
}

impl<I: Interner> TypeError<I> {
    pub fn involves_regions(self) -> bool {
        match self {
            TypeError::RegionsDoesNotOutlive(_, _)
            | TypeError::RegionsInsufficientlyPolymorphic(_, _)
            | TypeError::RegionsPlaceholderMismatch => true,
            _ => false,
        }
    }

    pub fn must_include_note(self) -> bool {
        use self::TypeError::*;
        match self {
            CyclicTy(_) | CyclicConst(_) | SafetyMismatch(_) | ConstnessMismatch(_)
            | PolarityMismatch(_) | Mismatch | AbiMismatch(_) | FixedArraySize(_)
            | ArgumentSorts(..) | Sorts(_) | VariadicMismatch(_) | TargetFeatureCast(_) => false,

            Mutability
            | ArgumentMutability(_)
            | TupleSize(_)
            | ArgCount
            | RegionsDoesNotOutlive(..)
            | RegionsInsufficientlyPolymorphic(..)
            | RegionsPlaceholderMismatch
            | Traits(_)
            | ProjectionMismatched(_)
            | ExistentialMismatch(_)
            | ConstMismatch(_)
            | IntrinsicCast => true,
        }
    }
}

impl<I: Interner> From<TypeError<I>> for NoSolution {
    fn from(_: TypeError<I>) -> NoSolution {
        NoSolution
    }
}
