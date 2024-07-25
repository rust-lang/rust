#![allow(clippy::derived_hash_with_manual_eq)]

use derive_where::derive_where;
#[cfg(feature = "nightly")]
use rustc_macros::{Decodable, Encodable, HashStable_NoContext, TyDecodable, TyEncodable};
use rustc_type_ir_macros::{TypeFoldable_Generic, TypeVisitable_Generic};
use std::fmt;

use crate::{self as ty, Interner};

/// A clause is something that can appear in where bounds or be inferred
/// by implied bounds.
#[derive_where(Clone, Copy, Hash, Eq; I: Interner)]
#[derive(TypeVisitable_Generic, TypeFoldable_Generic)]
#[cfg_attr(feature = "nightly", derive(TyEncodable, TyDecodable, HashStable_NoContext))]
pub enum ClauseKind<I: Interner> {
    /// Corresponds to `where Foo: Bar<A, B, C>`. `Foo` here would be
    /// the `Self` type of the trait reference and `A`, `B`, and `C`
    /// would be the type parameters.
    Trait(ty::TraitPredicate<I>),

    /// `where 'a: 'r`
    RegionOutlives(ty::OutlivesPredicate<I, I::Region>),

    /// `where T: 'r`
    TypeOutlives(ty::OutlivesPredicate<I, I::Ty>),

    /// `where <T as TraitRef>::Name == X`, approximately.
    /// See the `ProjectionPredicate` struct for details.
    Projection(ty::ProjectionPredicate<I>),

    /// Ensures that a const generic argument to a parameter `const N: u8`
    /// is of type `u8`.
    ConstArgHasType(I::Const, I::Ty),

    /// No syntax: `T` well-formed.
    WellFormed(I::GenericArg),

    /// Constant initializer must evaluate successfully.
    ConstEvaluatable(I::Const),
}

// FIXME(GrigorenkoPV): consider not implementing PartialEq manually
impl<I: Interner> PartialEq for ClauseKind<I> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Trait(l0), Self::Trait(r0)) => l0 == r0,
            (Self::RegionOutlives(l0), Self::RegionOutlives(r0)) => l0 == r0,
            (Self::TypeOutlives(l0), Self::TypeOutlives(r0)) => l0 == r0,
            (Self::Projection(l0), Self::Projection(r0)) => l0 == r0,
            (Self::ConstArgHasType(l0, l1), Self::ConstArgHasType(r0, r1)) => l0 == r0 && l1 == r1,
            (Self::WellFormed(l0), Self::WellFormed(r0)) => l0 == r0,
            (Self::ConstEvaluatable(l0), Self::ConstEvaluatable(r0)) => l0 == r0,
            _ => false,
        }
    }
}

#[derive_where(Clone, Copy, Hash, PartialEq, Eq; I: Interner)]
#[derive(TypeVisitable_Generic, TypeFoldable_Generic)]
#[cfg_attr(feature = "nightly", derive(TyEncodable, TyDecodable, HashStable_NoContext))]
pub enum PredicateKind<I: Interner> {
    /// Prove a clause
    Clause(ClauseKind<I>),

    /// Trait must be object-safe.
    ObjectSafe(I::DefId),

    /// `T1 <: T2`
    ///
    /// This obligation is created most often when we have two
    /// unresolved type variables and hence don't have enough
    /// information to process the subtyping obligation yet.
    Subtype(ty::SubtypePredicate<I>),

    /// `T1` coerced to `T2`
    ///
    /// Like a subtyping obligation, this is created most often
    /// when we have two unresolved type variables and hence
    /// don't have enough information to process the coercion
    /// obligation yet. At the moment, we actually process coercions
    /// very much like subtyping and don't handle the full coercion
    /// logic.
    Coerce(ty::CoercePredicate<I>),

    /// Constants must be equal. The first component is the const that is expected.
    ConstEquate(I::Const, I::Const),

    /// A marker predicate that is always ambiguous.
    /// Used for coherence to mark opaque types as possibly equal to each other but ambiguous.
    Ambiguous,

    /// This should only be used inside of the new solver for `AliasRelate` and expects
    /// the `term` to be an unconstrained inference variable.
    ///
    /// The alias normalizes to `term`. Unlike `Projection`, this always fails if the
    /// alias cannot be normalized in the current context. For the rigid alias
    /// `T as Trait>::Assoc`, `Projection(<T as Trait>::Assoc, ?x)` constrains `?x`
    /// to `<T as Trait>::Assoc` while `NormalizesTo(<T as Trait>::Assoc, ?x)`
    /// results in `NoSolution`.
    NormalizesTo(ty::NormalizesTo<I>),

    /// Separate from `ClauseKind::Projection` which is used for normalization in new solver.
    /// This predicate requires two terms to be equal to eachother.
    ///
    /// Only used for new solver.
    AliasRelate(I::Term, I::Term, AliasRelationDirection),
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Copy)]
#[cfg_attr(feature = "nightly", derive(HashStable_NoContext, Encodable, Decodable))]
pub enum AliasRelationDirection {
    Equate,
    Subtype,
}

impl std::fmt::Display for AliasRelationDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AliasRelationDirection::Equate => write!(f, "=="),
            AliasRelationDirection::Subtype => write!(f, "<:"),
        }
    }
}

impl<I: Interner> fmt::Debug for ClauseKind<I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ClauseKind::ConstArgHasType(ct, ty) => write!(f, "ConstArgHasType({ct:?}, {ty:?})"),
            ClauseKind::Trait(a) => a.fmt(f),
            ClauseKind::RegionOutlives(pair) => pair.fmt(f),
            ClauseKind::TypeOutlives(pair) => pair.fmt(f),
            ClauseKind::Projection(pair) => pair.fmt(f),
            ClauseKind::WellFormed(data) => write!(f, "WellFormed({data:?})"),
            ClauseKind::ConstEvaluatable(ct) => {
                write!(f, "ConstEvaluatable({ct:?})")
            }
        }
    }
}

impl<I: Interner> fmt::Debug for PredicateKind<I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PredicateKind::Clause(a) => a.fmt(f),
            PredicateKind::Subtype(pair) => pair.fmt(f),
            PredicateKind::Coerce(pair) => pair.fmt(f),
            PredicateKind::ObjectSafe(trait_def_id) => {
                write!(f, "ObjectSafe({trait_def_id:?})")
            }
            PredicateKind::ConstEquate(c1, c2) => write!(f, "ConstEquate({c1:?}, {c2:?})"),
            PredicateKind::Ambiguous => write!(f, "Ambiguous"),
            PredicateKind::NormalizesTo(p) => p.fmt(f),
            PredicateKind::AliasRelate(t1, t2, dir) => {
                write!(f, "AliasRelate({t1:?}, {dir:?}, {t2:?})")
            }
        }
    }
}
