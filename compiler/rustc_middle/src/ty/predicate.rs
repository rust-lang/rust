use std::cmp::Ordering;

use rustc_data_structures::intern::Interned;
use rustc_hir::def_id::DefId;
use rustc_macros::{HashStable, extension};
use rustc_type_ir as ir;

use crate::ty::{
    self, DebruijnIndex, EarlyBinder, Ty, TyCtxt, TypeFlags, Upcast, UpcastFrom, WithCachedTypeInfo,
};

pub type TraitRef<'tcx> = ir::TraitRef<TyCtxt<'tcx>>;
pub type AliasTerm<'tcx> = ir::AliasTerm<TyCtxt<'tcx>>;
pub type ProjectionPredicate<'tcx> = ir::ProjectionPredicate<TyCtxt<'tcx>>;
pub type ExistentialPredicate<'tcx> = ir::ExistentialPredicate<TyCtxt<'tcx>>;
pub type ExistentialTraitRef<'tcx> = ir::ExistentialTraitRef<TyCtxt<'tcx>>;
pub type ExistentialProjection<'tcx> = ir::ExistentialProjection<TyCtxt<'tcx>>;
pub type TraitPredicate<'tcx> = ir::TraitPredicate<TyCtxt<'tcx>>;
pub type HostEffectPredicate<'tcx> = ir::HostEffectPredicate<TyCtxt<'tcx>>;
pub type ClauseKind<'tcx> = ir::ClauseKind<TyCtxt<'tcx>>;
pub type PredicateKind<'tcx> = ir::PredicateKind<TyCtxt<'tcx>>;
pub type NormalizesTo<'tcx> = ir::NormalizesTo<TyCtxt<'tcx>>;
pub type CoercePredicate<'tcx> = ir::CoercePredicate<TyCtxt<'tcx>>;
pub type SubtypePredicate<'tcx> = ir::SubtypePredicate<TyCtxt<'tcx>>;
pub type OutlivesPredicate<'tcx, T> = ir::OutlivesPredicate<TyCtxt<'tcx>, T>;
pub type RegionOutlivesPredicate<'tcx> = OutlivesPredicate<'tcx, ty::Region<'tcx>>;
pub type TypeOutlivesPredicate<'tcx> = OutlivesPredicate<'tcx, Ty<'tcx>>;
pub type ArgOutlivesPredicate<'tcx> = OutlivesPredicate<'tcx, ty::GenericArg<'tcx>>;
pub type PolyTraitPredicate<'tcx> = ty::Binder<'tcx, TraitPredicate<'tcx>>;
pub type PolyRegionOutlivesPredicate<'tcx> = ty::Binder<'tcx, RegionOutlivesPredicate<'tcx>>;
pub type PolyTypeOutlivesPredicate<'tcx> = ty::Binder<'tcx, TypeOutlivesPredicate<'tcx>>;
pub type PolySubtypePredicate<'tcx> = ty::Binder<'tcx, SubtypePredicate<'tcx>>;
pub type PolyCoercePredicate<'tcx> = ty::Binder<'tcx, CoercePredicate<'tcx>>;
pub type PolyProjectionPredicate<'tcx> = ty::Binder<'tcx, ProjectionPredicate<'tcx>>;

/// A statement that can be proven by a trait solver. This includes things that may
/// show up in where clauses, such as trait predicates and projection predicates,
/// and also things that are emitted as part of type checking such as `DynCompatible`
/// predicate which is emitted when a type is coerced to a trait object.
///
/// Use this rather than `PredicateKind`, whenever possible.
#[derive(Clone, Copy, PartialEq, Eq, Hash, HashStable)]
#[rustc_pass_by_value]
pub struct Predicate<'tcx>(
    pub(super) Interned<'tcx, WithCachedTypeInfo<ty::Binder<'tcx, PredicateKind<'tcx>>>>,
);

impl<'tcx> rustc_type_ir::inherent::Predicate<TyCtxt<'tcx>> for Predicate<'tcx> {
    fn as_clause(self) -> Option<ty::Clause<'tcx>> {
        self.as_clause()
    }

    fn allow_normalization(self) -> bool {
        self.allow_normalization()
    }
}

impl<'tcx> rustc_type_ir::inherent::IntoKind for Predicate<'tcx> {
    type Kind = ty::Binder<'tcx, ty::PredicateKind<'tcx>>;

    fn kind(self) -> Self::Kind {
        self.kind()
    }
}

impl<'tcx> rustc_type_ir::Flags for Predicate<'tcx> {
    fn flags(&self) -> TypeFlags {
        self.0.flags
    }

    fn outer_exclusive_binder(&self) -> ty::DebruijnIndex {
        self.0.outer_exclusive_binder
    }
}

impl<'tcx> Predicate<'tcx> {
    /// Gets the inner `ty::Binder<'tcx, PredicateKind<'tcx>>`.
    #[inline]
    pub fn kind(self) -> ty::Binder<'tcx, PredicateKind<'tcx>> {
        self.0.internee
    }

    // FIXME(compiler-errors): Think about removing this.
    #[inline(always)]
    pub fn flags(self) -> TypeFlags {
        self.0.flags
    }

    // FIXME(compiler-errors): Think about removing this.
    #[inline(always)]
    pub fn outer_exclusive_binder(self) -> DebruijnIndex {
        self.0.outer_exclusive_binder
    }

    /// Flips the polarity of a Predicate.
    ///
    /// Given `T: Trait` predicate it returns `T: !Trait` and given `T: !Trait` returns `T: Trait`.
    pub fn flip_polarity(self, tcx: TyCtxt<'tcx>) -> Option<Predicate<'tcx>> {
        let kind = self
            .kind()
            .map_bound(|kind| match kind {
                PredicateKind::Clause(ClauseKind::Trait(TraitPredicate {
                    trait_ref,
                    polarity,
                })) => Some(PredicateKind::Clause(ClauseKind::Trait(TraitPredicate {
                    trait_ref,
                    polarity: polarity.flip(),
                }))),

                _ => None,
            })
            .transpose()?;

        Some(tcx.mk_predicate(kind))
    }

    /// Whether this projection can be soundly normalized.
    ///
    /// Wf predicates must not be normalized, as normalization
    /// can remove required bounds which would cause us to
    /// unsoundly accept some programs. See #91068.
    #[inline]
    pub fn allow_normalization(self) -> bool {
        match self.kind().skip_binder() {
            PredicateKind::Clause(ClauseKind::WellFormed(_)) | PredicateKind::AliasRelate(..) => {
                false
            }
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

impl<'tcx> rustc_errors::IntoDiagArg for Predicate<'tcx> {
    fn into_diag_arg(self, path: &mut Option<std::path::PathBuf>) -> rustc_errors::DiagArgValue {
        ty::tls::with(|tcx| {
            let pred = tcx.short_string(self, path);
            rustc_errors::DiagArgValue::Str(std::borrow::Cow::Owned(pred))
        })
    }
}

impl<'tcx> rustc_errors::IntoDiagArg for Clause<'tcx> {
    fn into_diag_arg(self, path: &mut Option<std::path::PathBuf>) -> rustc_errors::DiagArgValue {
        ty::tls::with(|tcx| {
            let clause = tcx.short_string(self, path);
            rustc_errors::DiagArgValue::Str(std::borrow::Cow::Owned(clause))
        })
    }
}

/// A subset of predicates which can be assumed by the trait solver. They show up in
/// an item's where clauses, hence the name `Clause`, and may either be user-written
/// (such as traits) or may be inserted during lowering.
#[derive(Clone, Copy, PartialEq, Eq, Hash, HashStable)]
#[rustc_pass_by_value]
pub struct Clause<'tcx>(
    pub(super) Interned<'tcx, WithCachedTypeInfo<ty::Binder<'tcx, PredicateKind<'tcx>>>>,
);

impl<'tcx> rustc_type_ir::inherent::Clause<TyCtxt<'tcx>> for Clause<'tcx> {
    fn as_predicate(self) -> Predicate<'tcx> {
        self.as_predicate()
    }

    fn instantiate_supertrait(self, tcx: TyCtxt<'tcx>, trait_ref: ty::PolyTraitRef<'tcx>) -> Self {
        self.instantiate_supertrait(tcx, trait_ref)
    }
}

impl<'tcx> rustc_type_ir::inherent::IntoKind for Clause<'tcx> {
    type Kind = ty::Binder<'tcx, ClauseKind<'tcx>>;

    fn kind(self) -> Self::Kind {
        self.kind()
    }
}

impl<'tcx> Clause<'tcx> {
    pub fn as_predicate(self) -> Predicate<'tcx> {
        Predicate(self.0)
    }

    pub fn kind(self) -> ty::Binder<'tcx, ClauseKind<'tcx>> {
        self.0.internee.map_bound(|kind| match kind {
            PredicateKind::Clause(clause) => clause,
            _ => unreachable!(),
        })
    }

    pub fn as_trait_clause(self) -> Option<ty::Binder<'tcx, TraitPredicate<'tcx>>> {
        let clause = self.kind();
        if let ty::ClauseKind::Trait(trait_clause) = clause.skip_binder() {
            Some(clause.rebind(trait_clause))
        } else {
            None
        }
    }

    pub fn as_projection_clause(self) -> Option<ty::Binder<'tcx, ProjectionPredicate<'tcx>>> {
        let clause = self.kind();
        if let ty::ClauseKind::Projection(projection_clause) = clause.skip_binder() {
            Some(clause.rebind(projection_clause))
        } else {
            None
        }
    }

    pub fn as_type_outlives_clause(self) -> Option<ty::Binder<'tcx, TypeOutlivesPredicate<'tcx>>> {
        let clause = self.kind();
        if let ty::ClauseKind::TypeOutlives(o) = clause.skip_binder() {
            Some(clause.rebind(o))
        } else {
            None
        }
    }

    pub fn as_region_outlives_clause(
        self,
    ) -> Option<ty::Binder<'tcx, RegionOutlivesPredicate<'tcx>>> {
        let clause = self.kind();
        if let ty::ClauseKind::RegionOutlives(o) = clause.skip_binder() {
            Some(clause.rebind(o))
        } else {
            None
        }
    }
}

impl<'tcx> rustc_type_ir::inherent::Clauses<TyCtxt<'tcx>> for ty::Clauses<'tcx> {}

#[extension(pub trait ExistentialPredicateStableCmpExt<'tcx>)]
impl<'tcx> ExistentialPredicate<'tcx> {
    /// Compares via an ordering that will not change if modules are reordered or other changes are
    /// made to the tree. In particular, this ordering is preserved across incremental compilations.
    fn stable_cmp(&self, tcx: TyCtxt<'tcx>, other: &Self) -> Ordering {
        match (*self, *other) {
            (ExistentialPredicate::Trait(_), ExistentialPredicate::Trait(_)) => Ordering::Equal,
            (ExistentialPredicate::Projection(ref a), ExistentialPredicate::Projection(ref b)) => {
                tcx.def_path_hash(a.def_id).cmp(&tcx.def_path_hash(b.def_id))
            }
            (ExistentialPredicate::AutoTrait(ref a), ExistentialPredicate::AutoTrait(ref b)) => {
                tcx.def_path_hash(*a).cmp(&tcx.def_path_hash(*b))
            }
            (ExistentialPredicate::Trait(_), _) => Ordering::Less,
            (ExistentialPredicate::Projection(_), ExistentialPredicate::Trait(_)) => {
                Ordering::Greater
            }
            (ExistentialPredicate::Projection(_), _) => Ordering::Less,
            (ExistentialPredicate::AutoTrait(_), _) => Ordering::Greater,
        }
    }
}

pub type PolyExistentialPredicate<'tcx> = ty::Binder<'tcx, ExistentialPredicate<'tcx>>;

impl<'tcx> rustc_type_ir::inherent::BoundExistentialPredicates<TyCtxt<'tcx>>
    for &'tcx ty::List<ty::PolyExistentialPredicate<'tcx>>
{
    fn principal_def_id(self) -> Option<DefId> {
        self.principal_def_id()
    }

    fn principal(self) -> Option<ty::PolyExistentialTraitRef<'tcx>> {
        self.principal()
    }

    fn auto_traits(self) -> impl IntoIterator<Item = DefId> {
        self.auto_traits()
    }

    fn projection_bounds(
        self,
    ) -> impl IntoIterator<Item = ty::Binder<'tcx, ExistentialProjection<'tcx>>> {
        self.projection_bounds()
    }
}

impl<'tcx> ty::List<ty::PolyExistentialPredicate<'tcx>> {
    /// Returns the "principal `DefId`" of this set of existential predicates.
    ///
    /// A Rust trait object type consists (in addition to a lifetime bound)
    /// of a set of trait bounds, which are separated into any number
    /// of auto-trait bounds, and at most one non-auto-trait bound. The
    /// non-auto-trait bound is called the "principal" of the trait
    /// object.
    ///
    /// Only the principal can have methods or type parameters (because
    /// auto traits can have neither of them). This is important, because
    /// it means the auto traits can be treated as an unordered set (methods
    /// would force an order for the vtable, while relating traits with
    /// type parameters without knowing the order to relate them in is
    /// a rather non-trivial task).
    ///
    /// For example, in the trait object `dyn std::fmt::Debug + Sync`, the
    /// principal bound is `Some(std::fmt::Debug)`, while the auto-trait bounds
    /// are the set `{Sync}`.
    ///
    /// It is also possible to have a "trivial" trait object that
    /// consists only of auto traits, with no principal - for example,
    /// `dyn Send + Sync`. In that case, the set of auto-trait bounds
    /// is `{Send, Sync}`, while there is no principal. These trait objects
    /// have a "trivial" vtable consisting of just the size, alignment,
    /// and destructor.
    pub fn principal(&self) -> Option<ty::Binder<'tcx, ExistentialTraitRef<'tcx>>> {
        self[0]
            .map_bound(|this| match this {
                ExistentialPredicate::Trait(tr) => Some(tr),
                _ => None,
            })
            .transpose()
    }

    pub fn principal_def_id(&self) -> Option<DefId> {
        self.principal().map(|trait_ref| trait_ref.skip_binder().def_id)
    }

    #[inline]
    pub fn projection_bounds(
        &self,
    ) -> impl Iterator<Item = ty::Binder<'tcx, ExistentialProjection<'tcx>>> {
        self.iter().filter_map(|predicate| {
            predicate
                .map_bound(|pred| match pred {
                    ExistentialPredicate::Projection(projection) => Some(projection),
                    _ => None,
                })
                .transpose()
        })
    }

    #[inline]
    pub fn auto_traits(&self) -> impl Iterator<Item = DefId> {
        self.iter().filter_map(|predicate| match predicate.skip_binder() {
            ExistentialPredicate::AutoTrait(did) => Some(did),
            _ => None,
        })
    }

    pub fn without_auto_traits(&self) -> impl Iterator<Item = ty::PolyExistentialPredicate<'tcx>> {
        self.iter().filter(|predicate| {
            !matches!(predicate.as_ref().skip_binder(), ExistentialPredicate::AutoTrait(_))
        })
    }
}

pub type PolyTraitRef<'tcx> = ty::Binder<'tcx, TraitRef<'tcx>>;
pub type PolyExistentialTraitRef<'tcx> = ty::Binder<'tcx, ExistentialTraitRef<'tcx>>;
pub type PolyExistentialProjection<'tcx> = ty::Binder<'tcx, ExistentialProjection<'tcx>>;

impl<'tcx> Clause<'tcx> {
    /// Performs a instantiation suitable for going from a
    /// poly-trait-ref to supertraits that must hold if that
    /// poly-trait-ref holds. This is slightly different from a normal
    /// instantiation in terms of what happens with bound regions. See
    /// lengthy comment below for details.
    pub fn instantiate_supertrait(
        self,
        tcx: TyCtxt<'tcx>,
        trait_ref: ty::PolyTraitRef<'tcx>,
    ) -> Clause<'tcx> {
        // The interaction between HRTB and supertraits is not entirely
        // obvious. Let me walk you (and myself) through an example.
        //
        // Let's start with an easy case. Consider two traits:
        //
        //     trait Foo<'a>: Bar<'a,'a> { }
        //     trait Bar<'b,'c> { }
        //
        // Now, if we have a trait reference `for<'x> T: Foo<'x>`, then
        // we can deduce that `for<'x> T: Bar<'x,'x>`. Basically, if we
        // knew that `Foo<'x>` (for any 'x) then we also know that
        // `Bar<'x,'x>` (for any 'x). This more-or-less falls out from
        // normal instantiation.
        //
        // In terms of why this is sound, the idea is that whenever there
        // is an impl of `T:Foo<'a>`, it must show that `T:Bar<'a,'a>`
        // holds. So if there is an impl of `T:Foo<'a>` that applies to
        // all `'a`, then we must know that `T:Bar<'a,'a>` holds for all
        // `'a`.
        //
        // Another example to be careful of is this:
        //
        //     trait Foo1<'a>: for<'b> Bar1<'a,'b> { }
        //     trait Bar1<'b,'c> { }
        //
        // Here, if we have `for<'x> T: Foo1<'x>`, then what do we know?
        // The answer is that we know `for<'x,'b> T: Bar1<'x,'b>`. The
        // reason is similar to the previous example: any impl of
        // `T:Foo1<'x>` must show that `for<'b> T: Bar1<'x, 'b>`. So
        // basically we would want to collapse the bound lifetimes from
        // the input (`trait_ref`) and the supertraits.
        //
        // To achieve this in practice is fairly straightforward. Let's
        // consider the more complicated scenario:
        //
        // - We start out with `for<'x> T: Foo1<'x>`. In this case, `'x`
        //   has a De Bruijn index of 1. We want to produce `for<'x,'b> T: Bar1<'x,'b>`,
        //   where both `'x` and `'b` would have a DB index of 1.
        //   The instantiation from the input trait-ref is therefore going to be
        //   `'a => 'x` (where `'x` has a DB index of 1).
        // - The supertrait-ref is `for<'b> Bar1<'a,'b>`, where `'a` is an
        //   early-bound parameter and `'b` is a late-bound parameter with a
        //   DB index of 1.
        // - If we replace `'a` with `'x` from the input, it too will have
        //   a DB index of 1, and thus we'll have `for<'x,'b> Bar1<'x,'b>`
        //   just as we wanted.
        //
        // There is only one catch. If we just apply the instantiation `'a
        // => 'x` to `for<'b> Bar1<'a,'b>`, the instantiation code will
        // adjust the DB index because we instantiating into a binder (it
        // tries to be so smart...) resulting in `for<'x> for<'b>
        // Bar1<'x,'b>` (we have no syntax for this, so use your
        // imagination). Basically the 'x will have DB index of 2 and 'b
        // will have DB index of 1. Not quite what we want. So we apply
        // the instantiation to the *contents* of the trait reference,
        // rather than the trait reference itself (put another way, the
        // instantiation code expects equal binding levels in the values
        // from the instantiation and the value being instantiated into, and
        // this trick achieves that).

        // Working through the second example:
        // trait_ref: for<'x> T: Foo1<'^0.0>; args: [T, '^0.0]
        // predicate: for<'b> Self: Bar1<'a, '^0.0>; args: [Self, 'a, '^0.0]
        // We want to end up with:
        //     for<'x, 'b> T: Bar1<'^0.0, '^0.1>
        // To do this:
        // 1) We must shift all bound vars in predicate by the length
        //    of trait ref's bound vars. So, we would end up with predicate like
        //    Self: Bar1<'a, '^0.1>
        // 2) We can then apply the trait args to this, ending up with
        //    T: Bar1<'^0.0, '^0.1>
        // 3) Finally, to create the final bound vars, we concatenate the bound
        //    vars of the trait ref with those of the predicate:
        //    ['x, 'b]
        let bound_pred = self.kind();
        let pred_bound_vars = bound_pred.bound_vars();
        let trait_bound_vars = trait_ref.bound_vars();
        // 1) Self: Bar1<'a, '^0.0> -> Self: Bar1<'a, '^0.1>
        let shifted_pred =
            tcx.shift_bound_var_indices(trait_bound_vars.len(), bound_pred.skip_binder());
        // 2) Self: Bar1<'a, '^0.1> -> T: Bar1<'^0.0, '^0.1>
        let new = EarlyBinder::bind(shifted_pred).instantiate(tcx, trait_ref.skip_binder().args);
        // 3) ['x] + ['b] -> ['x, 'b]
        let bound_vars =
            tcx.mk_bound_variable_kinds_from_iter(trait_bound_vars.iter().chain(pred_bound_vars));

        // FIXME: Is it really perf sensitive to use reuse_or_mk_predicate here?
        tcx.reuse_or_mk_predicate(
            self.as_predicate(),
            ty::Binder::bind_with_vars(PredicateKind::Clause(new), bound_vars),
        )
        .expect_clause()
    }
}

impl<'tcx> UpcastFrom<TyCtxt<'tcx>, PredicateKind<'tcx>> for Predicate<'tcx> {
    fn upcast_from(from: PredicateKind<'tcx>, tcx: TyCtxt<'tcx>) -> Self {
        ty::Binder::dummy(from).upcast(tcx)
    }
}

impl<'tcx> UpcastFrom<TyCtxt<'tcx>, ty::Binder<'tcx, PredicateKind<'tcx>>> for Predicate<'tcx> {
    fn upcast_from(from: ty::Binder<'tcx, PredicateKind<'tcx>>, tcx: TyCtxt<'tcx>) -> Self {
        tcx.mk_predicate(from)
    }
}

impl<'tcx> UpcastFrom<TyCtxt<'tcx>, ClauseKind<'tcx>> for Predicate<'tcx> {
    fn upcast_from(from: ClauseKind<'tcx>, tcx: TyCtxt<'tcx>) -> Self {
        tcx.mk_predicate(ty::Binder::dummy(PredicateKind::Clause(from)))
    }
}

impl<'tcx> UpcastFrom<TyCtxt<'tcx>, ty::Binder<'tcx, ClauseKind<'tcx>>> for Predicate<'tcx> {
    fn upcast_from(from: ty::Binder<'tcx, ClauseKind<'tcx>>, tcx: TyCtxt<'tcx>) -> Self {
        tcx.mk_predicate(from.map_bound(PredicateKind::Clause))
    }
}

impl<'tcx> UpcastFrom<TyCtxt<'tcx>, Clause<'tcx>> for Predicate<'tcx> {
    fn upcast_from(from: Clause<'tcx>, _tcx: TyCtxt<'tcx>) -> Self {
        from.as_predicate()
    }
}

impl<'tcx> UpcastFrom<TyCtxt<'tcx>, ClauseKind<'tcx>> for Clause<'tcx> {
    fn upcast_from(from: ClauseKind<'tcx>, tcx: TyCtxt<'tcx>) -> Self {
        tcx.mk_predicate(ty::Binder::dummy(PredicateKind::Clause(from))).expect_clause()
    }
}

impl<'tcx> UpcastFrom<TyCtxt<'tcx>, ty::Binder<'tcx, ClauseKind<'tcx>>> for Clause<'tcx> {
    fn upcast_from(from: ty::Binder<'tcx, ClauseKind<'tcx>>, tcx: TyCtxt<'tcx>) -> Self {
        tcx.mk_predicate(from.map_bound(|clause| PredicateKind::Clause(clause))).expect_clause()
    }
}

impl<'tcx> UpcastFrom<TyCtxt<'tcx>, TraitRef<'tcx>> for Predicate<'tcx> {
    fn upcast_from(from: TraitRef<'tcx>, tcx: TyCtxt<'tcx>) -> Self {
        ty::Binder::dummy(from).upcast(tcx)
    }
}

impl<'tcx> UpcastFrom<TyCtxt<'tcx>, TraitRef<'tcx>> for Clause<'tcx> {
    fn upcast_from(from: TraitRef<'tcx>, tcx: TyCtxt<'tcx>) -> Self {
        let p: Predicate<'tcx> = from.upcast(tcx);
        p.expect_clause()
    }
}

impl<'tcx> UpcastFrom<TyCtxt<'tcx>, ty::Binder<'tcx, TraitRef<'tcx>>> for Predicate<'tcx> {
    fn upcast_from(from: ty::Binder<'tcx, TraitRef<'tcx>>, tcx: TyCtxt<'tcx>) -> Self {
        let pred: PolyTraitPredicate<'tcx> = from.upcast(tcx);
        pred.upcast(tcx)
    }
}

impl<'tcx> UpcastFrom<TyCtxt<'tcx>, ty::Binder<'tcx, TraitRef<'tcx>>> for Clause<'tcx> {
    fn upcast_from(from: ty::Binder<'tcx, TraitRef<'tcx>>, tcx: TyCtxt<'tcx>) -> Self {
        let pred: PolyTraitPredicate<'tcx> = from.upcast(tcx);
        pred.upcast(tcx)
    }
}

impl<'tcx> UpcastFrom<TyCtxt<'tcx>, TraitPredicate<'tcx>> for Predicate<'tcx> {
    fn upcast_from(from: TraitPredicate<'tcx>, tcx: TyCtxt<'tcx>) -> Self {
        PredicateKind::Clause(ClauseKind::Trait(from)).upcast(tcx)
    }
}

impl<'tcx> UpcastFrom<TyCtxt<'tcx>, PolyTraitPredicate<'tcx>> for Predicate<'tcx> {
    fn upcast_from(from: PolyTraitPredicate<'tcx>, tcx: TyCtxt<'tcx>) -> Self {
        from.map_bound(|p| PredicateKind::Clause(ClauseKind::Trait(p))).upcast(tcx)
    }
}

impl<'tcx> UpcastFrom<TyCtxt<'tcx>, TraitPredicate<'tcx>> for Clause<'tcx> {
    fn upcast_from(from: TraitPredicate<'tcx>, tcx: TyCtxt<'tcx>) -> Self {
        let p: Predicate<'tcx> = from.upcast(tcx);
        p.expect_clause()
    }
}

impl<'tcx> UpcastFrom<TyCtxt<'tcx>, PolyTraitPredicate<'tcx>> for Clause<'tcx> {
    fn upcast_from(from: PolyTraitPredicate<'tcx>, tcx: TyCtxt<'tcx>) -> Self {
        let p: Predicate<'tcx> = from.upcast(tcx);
        p.expect_clause()
    }
}

impl<'tcx> UpcastFrom<TyCtxt<'tcx>, RegionOutlivesPredicate<'tcx>> for Predicate<'tcx> {
    fn upcast_from(from: RegionOutlivesPredicate<'tcx>, tcx: TyCtxt<'tcx>) -> Self {
        ty::Binder::dummy(PredicateKind::Clause(ClauseKind::RegionOutlives(from))).upcast(tcx)
    }
}

impl<'tcx> UpcastFrom<TyCtxt<'tcx>, PolyRegionOutlivesPredicate<'tcx>> for Predicate<'tcx> {
    fn upcast_from(from: PolyRegionOutlivesPredicate<'tcx>, tcx: TyCtxt<'tcx>) -> Self {
        from.map_bound(|p| PredicateKind::Clause(ClauseKind::RegionOutlives(p))).upcast(tcx)
    }
}

impl<'tcx> UpcastFrom<TyCtxt<'tcx>, TypeOutlivesPredicate<'tcx>> for Predicate<'tcx> {
    fn upcast_from(from: TypeOutlivesPredicate<'tcx>, tcx: TyCtxt<'tcx>) -> Self {
        ty::Binder::dummy(PredicateKind::Clause(ClauseKind::TypeOutlives(from))).upcast(tcx)
    }
}

impl<'tcx> UpcastFrom<TyCtxt<'tcx>, ProjectionPredicate<'tcx>> for Predicate<'tcx> {
    fn upcast_from(from: ProjectionPredicate<'tcx>, tcx: TyCtxt<'tcx>) -> Self {
        ty::Binder::dummy(PredicateKind::Clause(ClauseKind::Projection(from))).upcast(tcx)
    }
}

impl<'tcx> UpcastFrom<TyCtxt<'tcx>, PolyProjectionPredicate<'tcx>> for Predicate<'tcx> {
    fn upcast_from(from: PolyProjectionPredicate<'tcx>, tcx: TyCtxt<'tcx>) -> Self {
        from.map_bound(|p| PredicateKind::Clause(ClauseKind::Projection(p))).upcast(tcx)
    }
}

impl<'tcx> UpcastFrom<TyCtxt<'tcx>, ProjectionPredicate<'tcx>> for Clause<'tcx> {
    fn upcast_from(from: ProjectionPredicate<'tcx>, tcx: TyCtxt<'tcx>) -> Self {
        let p: Predicate<'tcx> = from.upcast(tcx);
        p.expect_clause()
    }
}

impl<'tcx> UpcastFrom<TyCtxt<'tcx>, PolyProjectionPredicate<'tcx>> for Clause<'tcx> {
    fn upcast_from(from: PolyProjectionPredicate<'tcx>, tcx: TyCtxt<'tcx>) -> Self {
        let p: Predicate<'tcx> = from.upcast(tcx);
        p.expect_clause()
    }
}

impl<'tcx> UpcastFrom<TyCtxt<'tcx>, ty::Binder<'tcx, ty::HostEffectPredicate<'tcx>>>
    for Predicate<'tcx>
{
    fn upcast_from(
        from: ty::Binder<'tcx, ty::HostEffectPredicate<'tcx>>,
        tcx: TyCtxt<'tcx>,
    ) -> Self {
        from.map_bound(ty::ClauseKind::HostEffect).upcast(tcx)
    }
}

impl<'tcx> UpcastFrom<TyCtxt<'tcx>, ty::Binder<'tcx, ty::HostEffectPredicate<'tcx>>>
    for Clause<'tcx>
{
    fn upcast_from(
        from: ty::Binder<'tcx, ty::HostEffectPredicate<'tcx>>,
        tcx: TyCtxt<'tcx>,
    ) -> Self {
        from.map_bound(ty::ClauseKind::HostEffect).upcast(tcx)
    }
}

impl<'tcx> UpcastFrom<TyCtxt<'tcx>, NormalizesTo<'tcx>> for Predicate<'tcx> {
    fn upcast_from(from: NormalizesTo<'tcx>, tcx: TyCtxt<'tcx>) -> Self {
        PredicateKind::NormalizesTo(from).upcast(tcx)
    }
}

impl<'tcx> Predicate<'tcx> {
    pub fn as_trait_clause(self) -> Option<PolyTraitPredicate<'tcx>> {
        let predicate = self.kind();
        match predicate.skip_binder() {
            PredicateKind::Clause(ClauseKind::Trait(t)) => Some(predicate.rebind(t)),
            _ => None,
        }
    }

    pub fn as_projection_clause(self) -> Option<PolyProjectionPredicate<'tcx>> {
        let predicate = self.kind();
        match predicate.skip_binder() {
            PredicateKind::Clause(ClauseKind::Projection(t)) => Some(predicate.rebind(t)),
            _ => None,
        }
    }

    /// Matches a `PredicateKind::Clause` and turns it into a `Clause`, otherwise returns `None`.
    pub fn as_clause(self) -> Option<Clause<'tcx>> {
        match self.kind().skip_binder() {
            PredicateKind::Clause(..) => Some(self.expect_clause()),
            _ => None,
        }
    }

    /// Assert that the predicate is a clause.
    pub fn expect_clause(self) -> Clause<'tcx> {
        match self.kind().skip_binder() {
            PredicateKind::Clause(..) => Clause(self.0),
            _ => bug!("{self} is not a clause"),
        }
    }
}

// Some types are used a lot. Make sure they don't unintentionally get bigger.
#[cfg(target_pointer_width = "64")]
mod size_asserts {
    use rustc_data_structures::static_assert_size;

    use super::*;
    // tidy-alphabetical-start
    static_assert_size!(PredicateKind<'_>, 32);
    static_assert_size!(WithCachedTypeInfo<PredicateKind<'_>>, 56);
    // tidy-alphabetical-end
}
