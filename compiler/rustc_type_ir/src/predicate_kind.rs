use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use std::fmt;
use std::ops::ControlFlow;

use crate::fold::{FallibleTypeFolder, TypeFoldable};
use crate::visit::{TypeVisitable, TypeVisitor};
use crate::{HashStableContext, Interner};

/// A clause is something that can appear in where bounds or be inferred
/// by implied bounds.
#[derive(derivative::Derivative)]
#[derivative(Clone(bound = ""), Hash(bound = ""))]
#[derive(TyEncodable, TyDecodable)]
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

impl<I: Interner> Copy for ClauseKind<I>
where
    I::Ty: Copy,
    I::Const: Copy,
    I::GenericArg: Copy,
    I::TraitPredicate: Copy,
    I::ProjectionPredicate: Copy,
    I::TypeOutlivesPredicate: Copy,
    I::RegionOutlivesPredicate: Copy,
{
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

fn clause_kind_discriminant<I: Interner>(value: &ClauseKind<I>) -> usize {
    match value {
        ClauseKind::Trait(_) => 0,
        ClauseKind::RegionOutlives(_) => 1,
        ClauseKind::TypeOutlives(_) => 2,
        ClauseKind::Projection(_) => 3,
        ClauseKind::ConstArgHasType(_, _) => 4,
        ClauseKind::WellFormed(_) => 5,
        ClauseKind::ConstEvaluatable(_) => 6,
    }
}

impl<CTX: HashStableContext, I: Interner> HashStable<CTX> for ClauseKind<I>
where
    I::Ty: HashStable<CTX>,
    I::Const: HashStable<CTX>,
    I::GenericArg: HashStable<CTX>,
    I::TraitPredicate: HashStable<CTX>,
    I::ProjectionPredicate: HashStable<CTX>,
    I::TypeOutlivesPredicate: HashStable<CTX>,
    I::RegionOutlivesPredicate: HashStable<CTX>,
{
    fn hash_stable(&self, hcx: &mut CTX, hasher: &mut StableHasher) {
        clause_kind_discriminant(self).hash_stable(hcx, hasher);
        match self {
            ClauseKind::Trait(p) => p.hash_stable(hcx, hasher),
            ClauseKind::RegionOutlives(p) => p.hash_stable(hcx, hasher),
            ClauseKind::TypeOutlives(p) => p.hash_stable(hcx, hasher),
            ClauseKind::Projection(p) => p.hash_stable(hcx, hasher),
            ClauseKind::ConstArgHasType(c, t) => {
                c.hash_stable(hcx, hasher);
                t.hash_stable(hcx, hasher);
            }
            ClauseKind::WellFormed(t) => t.hash_stable(hcx, hasher),
            ClauseKind::ConstEvaluatable(c) => c.hash_stable(hcx, hasher),
        }
    }
}

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
    fn visit_with<V: TypeVisitor<I>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        match self {
            ClauseKind::Trait(p) => p.visit_with(visitor),
            ClauseKind::RegionOutlives(p) => p.visit_with(visitor),
            ClauseKind::TypeOutlives(p) => p.visit_with(visitor),
            ClauseKind::Projection(p) => p.visit_with(visitor),
            ClauseKind::ConstArgHasType(c, t) => {
                c.visit_with(visitor)?;
                t.visit_with(visitor)
            }
            ClauseKind::WellFormed(p) => p.visit_with(visitor),
            ClauseKind::ConstEvaluatable(p) => p.visit_with(visitor),
        }
    }
}

#[derive(derivative::Derivative)]
#[derivative(Clone(bound = ""), Hash(bound = ""))]
#[derive(TyEncodable, TyDecodable)]
pub enum PredicateKind<I: Interner> {
    /// Prove a clause
    Clause(ClauseKind<I>),

    /// Trait must be object-safe.
    ObjectSafe(I::DefId),

    /// No direct syntax. May be thought of as `where T: FnFoo<...>`
    /// for some generic args `...` and `T` being a closure type.
    /// Satisfied (or refuted) once we know the closure's kind.
    ClosureKind(I::DefId, I::GenericArgs, I::ClosureKind),

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

    /// Separate from `ClauseKind::Projection` which is used for normalization in new solver.
    /// This predicate requires two terms to be equal to eachother.
    ///
    /// Only used for new solver
    AliasRelate(I::Term, I::Term, AliasRelationDirection),
}

impl<I: Interner> Copy for PredicateKind<I>
where
    I::DefId: Copy,
    I::Const: Copy,
    I::GenericArgs: Copy,
    I::Term: Copy,
    I::CoercePredicate: Copy,
    I::SubtypePredicate: Copy,
    I::ClosureKind: Copy,
    ClauseKind<I>: Copy,
{
}

impl<I: Interner> PartialEq for PredicateKind<I> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Clause(l0), Self::Clause(r0)) => l0 == r0,
            (Self::ObjectSafe(l0), Self::ObjectSafe(r0)) => l0 == r0,
            (Self::ClosureKind(l0, l1, l2), Self::ClosureKind(r0, r1, r2)) => {
                l0 == r0 && l1 == r1 && l2 == r2
            }
            (Self::Subtype(l0), Self::Subtype(r0)) => l0 == r0,
            (Self::Coerce(l0), Self::Coerce(r0)) => l0 == r0,
            (Self::ConstEquate(l0, l1), Self::ConstEquate(r0, r1)) => l0 == r0 && l1 == r1,
            (Self::AliasRelate(l0, l1, l2), Self::AliasRelate(r0, r1, r2)) => {
                l0 == r0 && l1 == r1 && l2 == r2
            }
            _ => core::mem::discriminant(self) == core::mem::discriminant(other),
        }
    }
}

impl<I: Interner> Eq for PredicateKind<I> {}

fn predicate_kind_discriminant<I: Interner>(value: &PredicateKind<I>) -> usize {
    match value {
        PredicateKind::Clause(_) => 0,
        PredicateKind::ObjectSafe(_) => 1,
        PredicateKind::ClosureKind(_, _, _) => 2,
        PredicateKind::Subtype(_) => 3,
        PredicateKind::Coerce(_) => 4,
        PredicateKind::ConstEquate(_, _) => 5,
        PredicateKind::Ambiguous => 6,
        PredicateKind::AliasRelate(_, _, _) => 7,
    }
}

impl<CTX: HashStableContext, I: Interner> HashStable<CTX> for PredicateKind<I>
where
    I::DefId: HashStable<CTX>,
    I::Const: HashStable<CTX>,
    I::GenericArgs: HashStable<CTX>,
    I::Term: HashStable<CTX>,
    I::CoercePredicate: HashStable<CTX>,
    I::SubtypePredicate: HashStable<CTX>,
    I::ClosureKind: HashStable<CTX>,
    ClauseKind<I>: HashStable<CTX>,
{
    fn hash_stable(&self, hcx: &mut CTX, hasher: &mut StableHasher) {
        predicate_kind_discriminant(self).hash_stable(hcx, hasher);
        match self {
            PredicateKind::Clause(p) => p.hash_stable(hcx, hasher),
            PredicateKind::ObjectSafe(d) => d.hash_stable(hcx, hasher),
            PredicateKind::ClosureKind(d, g, k) => {
                d.hash_stable(hcx, hasher);
                g.hash_stable(hcx, hasher);
                k.hash_stable(hcx, hasher);
            }
            PredicateKind::Subtype(p) => p.hash_stable(hcx, hasher),
            PredicateKind::Coerce(p) => p.hash_stable(hcx, hasher),
            PredicateKind::ConstEquate(c1, c2) => {
                c1.hash_stable(hcx, hasher);
                c2.hash_stable(hcx, hasher);
            }
            PredicateKind::Ambiguous => {}
            PredicateKind::AliasRelate(t1, t2, r) => {
                t1.hash_stable(hcx, hasher);
                t2.hash_stable(hcx, hasher);
                r.hash_stable(hcx, hasher);
            }
        }
    }
}

impl<I: Interner> TypeFoldable<I> for PredicateKind<I>
where
    I::DefId: TypeFoldable<I>,
    I::Const: TypeFoldable<I>,
    I::GenericArgs: TypeFoldable<I>,
    I::Term: TypeFoldable<I>,
    I::CoercePredicate: TypeFoldable<I>,
    I::SubtypePredicate: TypeFoldable<I>,
    I::ClosureKind: TypeFoldable<I>,
    ClauseKind<I>: TypeFoldable<I>,
{
    fn try_fold_with<F: FallibleTypeFolder<I>>(self, folder: &mut F) -> Result<Self, F::Error> {
        Ok(match self {
            PredicateKind::Clause(c) => PredicateKind::Clause(c.try_fold_with(folder)?),
            PredicateKind::ObjectSafe(d) => PredicateKind::ObjectSafe(d.try_fold_with(folder)?),
            PredicateKind::ClosureKind(d, g, k) => PredicateKind::ClosureKind(
                d.try_fold_with(folder)?,
                g.try_fold_with(folder)?,
                k.try_fold_with(folder)?,
            ),
            PredicateKind::Subtype(s) => PredicateKind::Subtype(s.try_fold_with(folder)?),
            PredicateKind::Coerce(s) => PredicateKind::Coerce(s.try_fold_with(folder)?),
            PredicateKind::ConstEquate(a, b) => {
                PredicateKind::ConstEquate(a.try_fold_with(folder)?, b.try_fold_with(folder)?)
            }
            PredicateKind::Ambiguous => PredicateKind::Ambiguous,
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
    I::ClosureKind: TypeVisitable<I>,
    ClauseKind<I>: TypeVisitable<I>,
{
    fn visit_with<V: TypeVisitor<I>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        match self {
            PredicateKind::Clause(p) => p.visit_with(visitor),
            PredicateKind::ObjectSafe(d) => d.visit_with(visitor),
            PredicateKind::ClosureKind(d, g, k) => {
                d.visit_with(visitor)?;
                g.visit_with(visitor)?;
                k.visit_with(visitor)
            }
            PredicateKind::Subtype(s) => s.visit_with(visitor),
            PredicateKind::Coerce(s) => s.visit_with(visitor),
            PredicateKind::ConstEquate(a, b) => {
                a.visit_with(visitor)?;
                b.visit_with(visitor)
            }
            PredicateKind::Ambiguous => ControlFlow::Continue(()),
            PredicateKind::AliasRelate(a, b, d) => {
                a.visit_with(visitor)?;
                b.visit_with(visitor)?;
                d.visit_with(visitor)
            }
        }
    }
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Copy)]
#[derive(HashStable_Generic, Encodable, Decodable)]
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
            PredicateKind::ClosureKind(closure_def_id, closure_args, kind) => {
                write!(f, "ClosureKind({closure_def_id:?}, {closure_args:?}, {kind:?})")
            }
            PredicateKind::ConstEquate(c1, c2) => write!(f, "ConstEquate({c1:?}, {c2:?})"),
            PredicateKind::Ambiguous => write!(f, "Ambiguous"),
            PredicateKind::AliasRelate(t1, t2, dir) => {
                write!(f, "AliasRelate({t1:?}, {dir:?}, {t2:?})")
            }
        }
    }
}
