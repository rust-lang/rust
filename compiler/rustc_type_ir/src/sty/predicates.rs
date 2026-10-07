use std::fmt;

use derive_where::derive_where;
#[cfg(feature = "nightly")]
use rustc_macros::StableHash_NoContext;

use crate::inherent::*;
use crate::intern::Interned as _;
use crate::{Binder, ClauseKind, DebruijnIndex, Flags, Interner, PredicateKind, TypeFlags};

/// A statement that can be proven by a trait solver.
///
/// This is the interner-generic representation of a predicate. The interned
/// representation also stores cached type flags and binder information.
#[derive_where(Clone, Copy, PartialEq, Eq, Hash; I: Interner)]
#[cfg_attr(feature = "nightly", derive(StableHash_NoContext))]
#[cfg_attr(feature = "nightly", rustc_pass_by_value)]
pub struct Predicate<I: Interner>(pub I::InternedPredicateKind);

impl<I: Interner> Predicate<I> {
    #[inline]
    pub fn new(interner: I, kind: Binder<I, PredicateKind<I>>) -> Self {
        interner.intern_predicate(kind)
    }

    #[inline]
    pub fn kind(self) -> Binder<I, PredicateKind<I>> {
        self.0.get().internee
    }

    /// Matches a `PredicateKind::Clause` and turns it into a `Clause`.
    #[inline]
    pub fn as_clause(self) -> Option<Clause<I>> {
        match self.kind().skip_binder() {
            PredicateKind::Clause(_) => Some(Clause(self)),
            _ => None,
        }
    }

    /// Assert that this predicate is a clause.
    #[inline]
    pub fn expect_clause(self) -> Clause<I> {
        match self.as_clause() {
            Some(clause) => clause,
            None => panic!("{self:?} is not a clause"),
        }
    }

    /// Whether this predicate can be soundly normalized.
    #[inline]
    pub fn allow_normalization(self) -> bool {
        match self.kind().skip_binder() {
            PredicateKind::Clause(ClauseKind::WellFormed(_)) => false,

            PredicateKind::Clause(ClauseKind::Trait(_))
            | PredicateKind::Clause(ClauseKind::HostEffect(..))
            | PredicateKind::Clause(ClauseKind::RegionOutlives(_))
            | PredicateKind::Clause(ClauseKind::TypeOutlives(_))
            | PredicateKind::Clause(ClauseKind::Projection(_))
            | PredicateKind::Clause(ClauseKind::ConstArgHasType(..))
            | PredicateKind::Clause(ClauseKind::UnstableFeature(_))
            | PredicateKind::DynCompatible(_)
            | PredicateKind::Subtype(_)
            | PredicateKind::Coerce(_)
            | PredicateKind::Clause(ClauseKind::ConstEvaluatable(_))
            | PredicateKind::ConstEquate(_, _)
            | PredicateKind::NormalizesTo(..)
            | PredicateKind::Ambiguous => true,
        }
    }
}

impl<I: Interner> fmt::Debug for Predicate<I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.kind().fmt(f)
    }
}

impl<I: Interner> Flags for Predicate<I> {
    #[inline]
    fn flags(&self) -> TypeFlags {
        self.0.get().flags
    }

    #[inline]
    fn outer_exclusive_binder(&self) -> DebruijnIndex {
        self.0.get().outer_exclusive_binder
    }
}

impl<I: Interner> IntoKind for Predicate<I> {
    type Kind = Binder<I, PredicateKind<I>>;

    #[inline]
    fn kind(self) -> Self::Kind {
        self.kind()
    }
}

/// A predicate which can be assumed by the trait solver.
///
/// `Clause` deliberately shares the same interned representation as
/// `Predicate`; construction is restricted to predicates whose kind is
/// `PredicateKind::Clause`.
#[derive_where(Clone, Copy, PartialEq, Eq, Hash; I: Interner)]
#[cfg_attr(feature = "nightly", derive(StableHash_NoContext))]
#[cfg_attr(feature = "nightly", rustc_pass_by_value)]
pub struct Clause<I: Interner>(Predicate<I>);

impl<I: Interner> Clause<I> {
    #[inline]
    pub fn as_predicate(self) -> Predicate<I> {
        self.0
    }

    #[inline]
    pub fn kind(self) -> Binder<I, ClauseKind<I>> {
        #[cold]
        #[inline(never)]
        fn unreachable_inner() -> ! {
            unreachable!()
        }

        self.0.kind().map_bound_no_validate_bound_vars(|kind| match kind {
            PredicateKind::Clause(clause) => clause,
            _ => unreachable_inner(),
        })
    }
}

impl<I: Interner> fmt::Debug for Clause<I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.kind().fmt(f)
    }
}

impl<I: Interner> Flags for Clause<I> {
    #[inline]
    fn flags(&self) -> TypeFlags {
        self.0.flags()
    }

    #[inline]
    fn outer_exclusive_binder(&self) -> DebruijnIndex {
        self.0.outer_exclusive_binder()
    }
}

impl<I: Interner> IntoKind for Clause<I> {
    type Kind = Binder<I, ClauseKind<I>>;

    #[inline]
    fn kind(self) -> Self::Kind {
        self.kind()
    }
}
