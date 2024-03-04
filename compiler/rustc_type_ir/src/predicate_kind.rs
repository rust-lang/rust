use rustc_ast_ir::try_visit;
use rustc_ast_ir::visit::VisitorResult;
use std::fmt;

use crate::fold::{FallibleTypeFolder, TypeFoldable};
use crate::visit::{TypeVisitable, TypeVisitor};
use crate::Interner;

/// A clause is something that can appear in where bounds or be inferred
/// by implied bounds.
#[derive(derivative::Derivative)]
#[derivative(Clone(bound = ""), Copy(bound = ""), Hash(bound = ""))]
#[cfg_attr(feature = "nightly", derive(TyEncodable, TyDecodable, HashStable_NoContext))]
pub enum ClauseKind<I: Interner> {
    /// Corresponds to `where Foo: Bar<A, B, C>`. `Foo` here would be
    /// the `Self` type of the trait reference and `A`, `B`, and `C`
    /// would be the type parameters.
    Trait(I::TraitPredicate),

    /// `where 'a: 'b`
    RegionOutlives(I::RegionOutlivesPredicate),

    /// `where T: 'a`
    TypeOutlives(I::TypeOutlivesPredicate),

    /// `where <T as TraitRef>::Name == X`, approximately.
    /// See the `ProjectionPredicate` struct for details.
    Projection(I::ProjectionPredicate),

    /// Ensures that a const generic argument to a parameter `const N: u8`
    /// is of type `u8`.
    ConstArgHasType(I::Const, I::Ty),

    /// No syntax: `T` well-formed.
    WellFormed(I::GenericArg),

    /// Constant initializer must evaluate successfully.
    ConstEvaluatable(I::Const),
}

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

impl<I: Interner> Eq for ClauseKind<I> {}

impl<I: Interner> TypeFoldable<I> for ClauseKind<I>
where
    I::Ty: TypeFoldable<I>,
    I::Const: TypeFoldable<I>,
    I::GenericArg: TypeFoldable<I>,
    I::TraitPredicate: TypeFoldable<I>,
    I::ProjectionPredicate: TypeFoldable<I>,
    I::TypeOutlivesPredicate: TypeFoldable<I>,
    I::RegionOutlivesPredicate: TypeFoldable<I>,
{
    fn try_fold_with<F: FallibleTypeFolder<I>>(self, folder: &mut F) -> Result<Self, F::Error> {
        Ok(match self {
            ClauseKind::Trait(p) => ClauseKind::Trait(p.try_fold_with(folder)?),
            ClauseKind::RegionOutlives(p) => ClauseKind::RegionOutlives(p.try_fold_with(folder)?),
            ClauseKind::TypeOutlives(p) => ClauseKind::TypeOutlives(p.try_fold_with(folder)?),
            ClauseKind::Projection(p) => ClauseKind::Projection(p.try_fold_with(folder)?),
            ClauseKind::ConstArgHasType(c, t) => {
                ClauseKind::ConstArgHasType(c.try_fold_with(folder)?, t.try_fold_with(folder)?)
            }
            ClauseKind::WellFormed(p) => ClauseKind::WellFormed(p.try_fold_with(folder)?),
            ClauseKind::ConstEvaluatable(p) => {
                ClauseKind::ConstEvaluatable(p.try_fold_with(folder)?)
            }
        })
    }
}

impl<I: Interner> TypeVisitable<I> for ClauseKind<I>
where
    I::Ty: TypeVisitable<I>,
    I::Const: TypeVisitable<I>,
    I::GenericArg: TypeVisitable<I>,
    I::TraitPredicate: TypeVisitable<I>,
    I::ProjectionPredicate: TypeVisitable<I>,
    I::TypeOutlivesPredicate: TypeVisitable<I>,
    I::RegionOutlivesPredicate: TypeVisitable<I>,
{
    fn visit_with<V: TypeVisitor<I>>(&self, visitor: &mut V) -> V::Result {
        match self {
            ClauseKind::Trait(p) => p.visit_with(visitor),
            ClauseKind::RegionOutlives(p) => p.visit_with(visitor),
            ClauseKind::TypeOutlives(p) => p.visit_with(visitor),
            ClauseKind::Projection(p) => p.visit_with(visitor),
            ClauseKind::ConstArgHasType(c, t) => {
                try_visit!(c.visit_with(visitor));
                t.visit_with(visitor)
            }
            ClauseKind::WellFormed(p) => p.visit_with(visitor),
            ClauseKind::ConstEvaluatable(p) => p.visit_with(visitor),
        }
    }
}

#[derive(derivative::Derivative)]
#[derivative(
    Clone(bound = ""),
    Copy(bound = ""),
    Hash(bound = ""),
    PartialEq(bound = ""),
    Eq(bound = "")
)]
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
    Subtype(I::SubtypePredicate),

    /// `T1` coerced to `T2`
    ///
    /// Like a subtyping obligation, this is created most often
    /// when we have two unresolved type variables and hence
    /// don't have enough information to process the coercion
    /// obligation yet. At the moment, we actually process coercions
    /// very much like subtyping and don't handle the full coercion
    /// logic.
    Coerce(I::CoercePredicate),

    /// Constants must be equal. The first component is the const that is expected.
    ConstEquate(I::Const, I::Const),

    /// A marker predicate that is always ambiguous.
    /// Used for coherence to mark opaque types as possibly equal to each other but ambiguous.
    Ambiguous,

    /// The alias normalizes to `term`. Unlike `Projection`, this always fails if the alias
    /// cannot be normalized in the current context.
    ///
    /// `Projection(<T as Trait>::Assoc, ?x)` results in `?x == <T as Trait>::Assoc` while
    /// `NormalizesTo(<T as Trait>::Assoc, ?x)` results in `NoSolution`.
    ///
    /// Only used in the new solver.
    NormalizesTo(I::NormalizesTo),

    /// Separate from `ClauseKind::Projection` which is used for normalization in new solver.
    /// This predicate requires two terms to be equal to eachother.
    ///
    /// Only used for new solver
    AliasRelate(I::Term, I::Term, AliasRelationDirection),
}

impl<I: Interner> TypeFoldable<I> for PredicateKind<I>
where
    I::DefId: TypeFoldable<I>,
    I::Const: TypeFoldable<I>,
    I::GenericArgs: TypeFoldable<I>,
    I::Term: TypeFoldable<I>,
    I::CoercePredicate: TypeFoldable<I>,
    I::SubtypePredicate: TypeFoldable<I>,
    I::NormalizesTo: TypeFoldable<I>,
    ClauseKind<I>: TypeFoldable<I>,
{
    fn try_fold_with<F: FallibleTypeFolder<I>>(self, folder: &mut F) -> Result<Self, F::Error> {
        Ok(match self {
            PredicateKind::Clause(c) => PredicateKind::Clause(c.try_fold_with(folder)?),
            PredicateKind::ObjectSafe(d) => PredicateKind::ObjectSafe(d.try_fold_with(folder)?),
            PredicateKind::Subtype(s) => PredicateKind::Subtype(s.try_fold_with(folder)?),
            PredicateKind::Coerce(s) => PredicateKind::Coerce(s.try_fold_with(folder)?),
            PredicateKind::ConstEquate(a, b) => {
                PredicateKind::ConstEquate(a.try_fold_with(folder)?, b.try_fold_with(folder)?)
            }
            PredicateKind::Ambiguous => PredicateKind::Ambiguous,
            PredicateKind::NormalizesTo(p) => PredicateKind::NormalizesTo(p.try_fold_with(folder)?),
            PredicateKind::AliasRelate(a, b, d) => PredicateKind::AliasRelate(
                a.try_fold_with(folder)?,
                b.try_fold_with(folder)?,
                d.try_fold_with(folder)?,
            ),
        })
    }
}

impl<I: Interner> TypeVisitable<I> for PredicateKind<I>
where
    I::DefId: TypeVisitable<I>,
    I::Const: TypeVisitable<I>,
    I::GenericArgs: TypeVisitable<I>,
    I::Term: TypeVisitable<I>,
    I::CoercePredicate: TypeVisitable<I>,
    I::SubtypePredicate: TypeVisitable<I>,
    I::NormalizesTo: TypeVisitable<I>,
    ClauseKind<I>: TypeVisitable<I>,
{
    fn visit_with<V: TypeVisitor<I>>(&self, visitor: &mut V) -> V::Result {
        match self {
            PredicateKind::Clause(p) => p.visit_with(visitor),
            PredicateKind::ObjectSafe(d) => d.visit_with(visitor),
            PredicateKind::Subtype(s) => s.visit_with(visitor),
            PredicateKind::Coerce(s) => s.visit_with(visitor),
            PredicateKind::ConstEquate(a, b) => {
                try_visit!(a.visit_with(visitor));
                b.visit_with(visitor)
            }
            PredicateKind::Ambiguous => V::Result::output(),
            PredicateKind::NormalizesTo(p) => p.visit_with(visitor),
            PredicateKind::AliasRelate(a, b, d) => {
                try_visit!(a.visit_with(visitor));
                try_visit!(b.visit_with(visitor));
                d.visit_with(visitor)
            }
        }
    }
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

// FIXME: Convert to DebugWithInfcx impl
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

// FIXME: Convert to DebugWithInfcx impl
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
