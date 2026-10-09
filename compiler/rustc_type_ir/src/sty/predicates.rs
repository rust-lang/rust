use std::fmt;

use derive_where::derive_where;
#[cfg(feature = "nightly")]
use rustc_macros::StableHash_NoContext;
use rustc_type_ir_macros::Lift_Generic;

use crate::inherent::*;
use crate::intern::Interned as _;
use crate::{
    Binder, ClauseKind, DebruijnIndex, EarlyBinder, FallibleTypeFolder, Flags, HostEffectClause,
    Interner, NormalizesTo, OutlivesClause, PredicateKind, PredicateProxy, ProjectionClause,
    Region, TraitClause, TraitRef, TypeFlags, TypeFoldable, TypeFolder, TypeSuperFoldable,
    TypeSuperVisitable, TypeVisitable, TypeVisitor, Upcast, UpcastFrom, shift_bound_var_indices,
};

/// A statement that can be proven by a trait solver.
///
/// This is the interner-generic representation of a predicate. The interned
/// representation also stores cached type flags and binder information.
#[derive_where(Clone, Copy, PartialEq, Eq, Hash; I: Interner)]
#[cfg_attr(feature = "nightly", derive(StableHash_NoContext))]
#[cfg_attr(feature = "nightly", rustc_pass_by_value)]
#[derive(Lift_Generic)]
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

    /// Flips the polarity of a trait predicate.
    ///
    /// Given `T: Trait`, returns `T: !Trait`, and vice versa.
    pub fn flip_polarity(self, interner: I) -> Option<Self> {
        let kind = self
            .kind()
            .map_bound(|kind| match kind {
                PredicateKind::Clause(ClauseKind::Trait(TraitClause { trait_ref, polarity })) => {
                    Some(PredicateKind::Clause(ClauseKind::Trait(TraitClause {
                        trait_ref,
                        polarity: polarity.flip(),
                    })))
                }

                _ => None,
            })
            .transpose()?;

        Some(Self::new(interner, kind))
    }

    pub fn as_trait_clause(self) -> Option<Binder<I, TraitClause<I>>> {
        let predicate = self.kind();
        match predicate.skip_binder() {
            PredicateKind::Clause(ClauseKind::Trait(trait_clause)) => {
                Some(predicate.rebind(trait_clause))
            }
            _ => None,
        }
    }

    pub fn as_projection_clause(self) -> Option<Binder<I, ProjectionClause<I>>> {
        let predicate = self.kind();
        match predicate.skip_binder() {
            PredicateKind::Clause(ClauseKind::Projection(projection_clause)) => {
                Some(predicate.rebind(projection_clause))
            }
            _ => None,
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
#[derive(Lift_Generic)]
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

    pub fn as_trait_clause(self) -> Option<Binder<I, TraitClause<I>>> {
        let clause = self.kind();
        if let ClauseKind::Trait(trait_clause) = clause.skip_binder() {
            Some(clause.rebind(trait_clause))
        } else {
            None
        }
    }

    pub fn as_projection_clause(self) -> Option<Binder<I, ProjectionClause<I>>> {
        let clause = self.kind();
        if let ClauseKind::Projection(projection_clause) = clause.skip_binder() {
            Some(clause.rebind(projection_clause))
        } else {
            None
        }
    }

    pub fn as_type_outlives_clause(self) -> Option<Binder<I, OutlivesClause<I, I::Ty>>> {
        let clause = self.kind();
        if let ClauseKind::TypeOutlives(outlives) = clause.skip_binder() {
            Some(clause.rebind(outlives))
        } else {
            None
        }
    }

    pub fn as_region_outlives_clause(self) -> Option<Binder<I, OutlivesClause<I, Region<I>>>> {
        let clause = self.kind();
        if let ClauseKind::RegionOutlives(outlives) = clause.skip_binder() {
            Some(clause.rebind(outlives))
        } else {
            None
        }
    }

    pub fn as_host_effect_clause(self) -> Option<Binder<I, HostEffectClause<I>>> {
        let clause = self.kind();
        if let ClauseKind::HostEffect(host_effect) = clause.skip_binder() {
            Some(clause.rebind(host_effect))
        } else {
            None
        }
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

impl<I: Interner> UpcastFrom<I, PredicateKind<I>> for Predicate<I> {
    fn upcast_from(from: PredicateKind<I>, interner: I) -> Self {
        Self::new(interner, Binder::dummy(from))
    }
}

impl<I: Interner> UpcastFrom<I, Binder<I, PredicateKind<I>>> for Predicate<I> {
    fn upcast_from(from: Binder<I, PredicateKind<I>>, interner: I) -> Self {
        Self::new(interner, from)
    }
}

impl<I: Interner> UpcastFrom<I, ClauseKind<I>> for Predicate<I> {
    fn upcast_from(from: ClauseKind<I>, interner: I) -> Self {
        Self::new(interner, Binder::dummy(PredicateKind::Clause(from)))
    }
}

impl<I: Interner> UpcastFrom<I, Binder<I, ClauseKind<I>>> for Predicate<I> {
    fn upcast_from(from: Binder<I, ClauseKind<I>>, interner: I) -> Self {
        Self::new(interner, from.map_bound(PredicateKind::Clause))
    }
}

impl<I: Interner> UpcastFrom<I, Clause<I>> for Predicate<I> {
    fn upcast_from(from: Clause<I>, _interner: I) -> Self {
        from.as_predicate()
    }
}

impl<I: Interner> UpcastFrom<I, ClauseKind<I>> for Clause<I> {
    fn upcast_from(from: ClauseKind<I>, interner: I) -> Self {
        Predicate::new(interner, Binder::dummy(PredicateKind::Clause(from))).expect_clause()
    }
}

impl<I: Interner> UpcastFrom<I, Binder<I, ClauseKind<I>>> for Clause<I> {
    fn upcast_from(from: Binder<I, ClauseKind<I>>, interner: I) -> Self {
        Predicate::new(interner, from.map_bound(PredicateKind::Clause)).expect_clause()
    }
}

impl<I: Interner> UpcastFrom<I, TraitRef<I>> for Predicate<I> {
    fn upcast_from(from: TraitRef<I>, interner: I) -> Self {
        let trait_clause: TraitClause<I> = from.upcast(interner);
        PredicateKind::Clause(ClauseKind::Trait(trait_clause)).upcast(interner)
    }
}

impl<I: Interner> UpcastFrom<I, TraitRef<I>> for Clause<I> {
    fn upcast_from(from: TraitRef<I>, interner: I) -> Self {
        let predicate: Predicate<I> = from.upcast(interner);
        predicate.expect_clause()
    }
}

impl<I: Interner> UpcastFrom<I, Binder<I, TraitRef<I>>> for Predicate<I> {
    fn upcast_from(from: Binder<I, TraitRef<I>>, interner: I) -> Self {
        let trait_clause: Binder<I, TraitClause<I>> = from.upcast(interner);
        trait_clause.upcast(interner)
    }
}

impl<I: Interner> UpcastFrom<I, Binder<I, TraitRef<I>>> for Clause<I> {
    fn upcast_from(from: Binder<I, TraitRef<I>>, interner: I) -> Self {
        let predicate: Predicate<I> = from.upcast(interner);
        predicate.expect_clause()
    }
}

impl<I: Interner> UpcastFrom<I, TraitClause<I>> for Predicate<I> {
    fn upcast_from(from: TraitClause<I>, interner: I) -> Self {
        PredicateKind::Clause(ClauseKind::Trait(from)).upcast(interner)
    }
}

impl<I: Interner> UpcastFrom<I, Binder<I, TraitClause<I>>> for Predicate<I> {
    fn upcast_from(from: Binder<I, TraitClause<I>>, interner: I) -> Self {
        from.map_bound_no_validate_bound_vars(|trait_clause| {
            PredicateKind::Clause(ClauseKind::Trait(trait_clause))
        })
        .upcast(interner)
    }
}

impl<I: Interner> UpcastFrom<I, TraitClause<I>> for Clause<I> {
    fn upcast_from(from: TraitClause<I>, interner: I) -> Self {
        let predicate: Predicate<I> = from.upcast(interner);
        predicate.expect_clause()
    }
}

impl<I: Interner> UpcastFrom<I, Binder<I, TraitClause<I>>> for Clause<I> {
    fn upcast_from(from: Binder<I, TraitClause<I>>, interner: I) -> Self {
        let predicate: Predicate<I> = from.upcast(interner);
        predicate.expect_clause()
    }
}

impl<I: Interner> UpcastFrom<I, ProjectionClause<I>> for Predicate<I> {
    fn upcast_from(from: ProjectionClause<I>, interner: I) -> Self {
        PredicateKind::Clause(ClauseKind::Projection(from)).upcast(interner)
    }
}

impl<I: Interner> UpcastFrom<I, Binder<I, ProjectionClause<I>>> for Predicate<I> {
    fn upcast_from(from: Binder<I, ProjectionClause<I>>, interner: I) -> Self {
        from.map_bound(|projection| PredicateKind::Clause(ClauseKind::Projection(projection)))
            .upcast(interner)
    }
}

impl<I: Interner> UpcastFrom<I, ProjectionClause<I>> for Clause<I> {
    fn upcast_from(from: ProjectionClause<I>, interner: I) -> Self {
        let predicate: Predicate<I> = from.upcast(interner);
        predicate.expect_clause()
    }
}

impl<I: Interner> UpcastFrom<I, Binder<I, ProjectionClause<I>>> for Clause<I> {
    fn upcast_from(from: Binder<I, ProjectionClause<I>>, interner: I) -> Self {
        let predicate: Predicate<I> = from.upcast(interner);
        predicate.expect_clause()
    }
}

impl<I: Interner> UpcastFrom<I, Binder<I, HostEffectClause<I>>> for Predicate<I> {
    fn upcast_from(from: Binder<I, HostEffectClause<I>>, interner: I) -> Self {
        from.map_bound(ClauseKind::HostEffect).upcast(interner)
    }
}

impl<I: Interner> UpcastFrom<I, Binder<I, HostEffectClause<I>>> for Clause<I> {
    fn upcast_from(from: Binder<I, HostEffectClause<I>>, interner: I) -> Self {
        let predicate: Predicate<I> = from.map_bound(ClauseKind::HostEffect).upcast(interner);
        predicate.expect_clause()
    }
}

impl<I: Interner> UpcastFrom<I, NormalizesTo<I>> for Predicate<I> {
    fn upcast_from(from: NormalizesTo<I>, interner: I) -> Self {
        PredicateKind::NormalizesTo(from).upcast(interner)
    }
}

impl<I: Interner> TypeFoldable<I> for Predicate<I> {
    fn try_fold_with<F: FallibleTypeFolder<I>>(self, folder: &mut F) -> Result<Self, F::Error> {
        folder.try_fold_predicate(self)
    }

    fn fold_with<F: TypeFolder<I>>(self, folder: &mut F) -> Self {
        folder.fold_predicate(self)
    }
}

impl<I: Interner> TypeVisitable<I> for Predicate<I> {
    fn visit_with<V: TypeVisitor<I>>(&self, visitor: &mut V) -> V::Result {
        visitor.visit_predicate(*self)
    }
}

impl<I: Interner> TypeSuperFoldable<I> for Predicate<I> {
    fn try_super_fold_with<F: FallibleTypeFolder<I>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        let new = self.kind().try_fold_with(folder)?;

        if new == self.kind() { Ok(self) } else { Ok(Self::new(folder.cx(), new)) }
    }

    fn super_fold_with<F: TypeFolder<I>>(self, folder: &mut F) -> Self {
        let new = self.kind().fold_with(folder);

        if new == self.kind() { self } else { Self::new(folder.cx(), new) }
    }
}

impl<I: Interner> TypeSuperVisitable<I> for Predicate<I> {
    fn super_visit_with<V: TypeVisitor<I>>(&self, visitor: &mut V) -> V::Result {
        self.kind().visit_with(visitor)
    }
}

impl<I: Interner> PredicateProxy<I> for Predicate<I> {
    fn allow_normalization(&self) -> bool {
        (*self).allow_normalization()
    }

    fn clause_kind_unchecked(&self) -> Option<Binder<I, ClauseKind<I>>> {
        self.as_clause().map(Clause::kind)
    }

    fn map_projection(
        self,
        cx: I,
        f: impl FnOnce(Binder<I, ProjectionClause<I>>) -> Binder<I, ProjectionClause<I>>,
    ) -> Option<Self> {
        self.as_projection_clause().map(|projection| f(projection).upcast(cx))
    }
}

impl<I: Interner> TypeFoldable<I> for Clause<I> {
    fn try_fold_with<F: FallibleTypeFolder<I>>(self, folder: &mut F) -> Result<Self, F::Error> {
        folder.try_fold_predicate(self)
    }

    fn fold_with<F: TypeFolder<I>>(self, folder: &mut F) -> Self {
        folder.fold_predicate(self)
    }
}

impl<I: Interner> TypeVisitable<I> for Clause<I> {
    fn visit_with<V: TypeVisitor<I>>(&self, visitor: &mut V) -> V::Result {
        visitor.visit_predicate(*self)
    }
}

impl<I: Interner> TypeSuperFoldable<I> for Clause<I> {
    fn try_super_fold_with<F: FallibleTypeFolder<I>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        self.as_predicate().try_super_fold_with(folder).map(Predicate::expect_clause)
    }

    fn super_fold_with<F: TypeFolder<I>>(self, folder: &mut F) -> Self {
        self.as_predicate().super_fold_with(folder).expect_clause()
    }
}

impl<I: Interner> TypeSuperVisitable<I> for Clause<I> {
    fn super_visit_with<V: TypeVisitor<I>>(&self, visitor: &mut V) -> V::Result {
        self.as_predicate().super_visit_with(visitor)
    }
}

impl<I: Interner> PredicateProxy<I> for Clause<I> {
    fn allow_normalization(&self) -> bool {
        self.as_predicate().allow_normalization()
    }

    fn clause_kind_unchecked(&self) -> Option<Binder<I, ClauseKind<I>>> {
        Some(self.kind())
    }

    fn map_projection(
        self,
        cx: I,
        f: impl FnOnce(Binder<I, ProjectionClause<I>>) -> Binder<I, ProjectionClause<I>>,
    ) -> Option<Self> {
        self.as_projection_clause().map(|projection| f(projection).upcast(cx))
    }
}

impl<I: Interner> Clause<I> {
    /// Instantiate a supertrait clause using the arguments of `trait_ref`.
    ///
    /// Both binders' variables become bound by the resulting binder.
    /// Shift the supertrait's bound-variable indices before substituting
    /// the trait arguments, so the two sets of variables do not overlap.
    pub fn instantiate_supertrait(self, cx: I, trait_ref: Binder<I, TraitRef<I>>) -> Self {
        let bound_pred = self.kind();
        let pred_bound_vars = bound_pred.bound_vars();
        let trait_bound_vars = trait_ref.bound_vars();

        let shifted_pred =
            shift_bound_var_indices(cx, trait_bound_vars.len(), bound_pred.skip_binder());

        let new = EarlyBinder::bind(cx, shifted_pred)
            .instantiate(cx, trait_ref.skip_binder().args)
            .skip_norm_wip();

        let bound_vars =
            I::BoundVarKinds::from_vars(cx, trait_bound_vars.iter().chain(pred_bound_vars.iter()));

        let new_kind = Binder::bind_with_vars(PredicateKind::Clause(new), bound_vars);
        let old_predicate = self.as_predicate();

        if old_predicate.kind() == new_kind {
            self
        } else {
            Predicate::new(cx, new_kind).expect_clause()
        }
    }
}
