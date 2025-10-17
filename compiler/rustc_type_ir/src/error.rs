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
    pub fn new(expected: T, found: T) -> Self {
        ExpectedFound { expected, found }
    }
}

// Data structures used in type unification
#[derive_where(Clone, Copy, PartialEq, Debug; I: Interner)]
#[derive(TypeVisitable_Generic)]
#[cfg_attr(feature = "nightly", rustc_pass_by_value)]
pub enum TypeError<I: Interner> {
    Mismatch,
    PolarityMismatch(#[type_visitable(ignore)] ExpectedFound<ty::PredicatePolarity>),
    SafetyMismatch(#[type_visitable(ignore)] ExpectedFound<I::Safety>),
    AbiMismatch(#[type_visitable(ignore)] ExpectedFound<I::Abi>),
    Mutability,
    ArgumentMutability(usize),
    TupleSize(ExpectedFound<usize>),
    ArraySize(ExpectedFound<I::Const>),
    ArgCount,

    RegionsDoesNotOutlive(I::Region, I::Region),
    RegionsInsufficientlyPolymorphic(I::BoundRegion, I::Region),
    RegionsPlaceholderMismatch,

    Sorts(ExpectedFound<I::Ty>),
    ArgumentSorts(ExpectedFound<I::Ty>, usize),
    Traits(ExpectedFound<I::TraitId>),
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
    /// `#[rustc_force_inline]` functions must be inlined and must not be codegened independently,
    /// so casting to a function pointer must be prohibited.
    ForceInlineCast,
    /// Safe `#[target_feature]` functions are not assignable to safe function pointers.
    TargetFeatureCast(I::DefId),
}

impl<I: Interner> Eq for TypeError<I> {}

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
            CyclicTy(_) | CyclicConst(_) | SafetyMismatch(_) | PolarityMismatch(_) | Mismatch
            | AbiMismatch(_) | ArraySize(_) | ArgumentSorts(..) | Sorts(_)
            | VariadicMismatch(_) | TargetFeatureCast(_) => false,

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
            | ForceInlineCast
            | IntrinsicCast => true,
        }
    }
}

impl<I: Interner> From<TypeError<I>> for NoSolution {
    fn from(_: TypeError<I>) -> NoSolution {
        NoSolution
    }
}
