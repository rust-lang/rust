use std::cmp::Ordering;

use rustc_hir::def_id::DefId;
use rustc_macros::extension;
use rustc_type_ir as ir;

use crate::ty::{self, Ty, TyCtxt, Upcast, UpcastFrom};

pub type TraitRef<'tcx> = ir::TraitRef<TyCtxt<'tcx>>;
pub type AliasTerm<'tcx> = ir::AliasTerm<TyCtxt<'tcx>>;
pub type AliasTermKind<'tcx> = ir::AliasTermKind<TyCtxt<'tcx>>;
pub type ProjectionClause<'tcx> = ir::ProjectionClause<TyCtxt<'tcx>>;
pub type ExistentialPredicate<'tcx> = ir::ExistentialPredicate<TyCtxt<'tcx>>;
pub type ExistentialTraitRef<'tcx> = ir::ExistentialTraitRef<TyCtxt<'tcx>>;
pub type ExistentialProjection<'tcx> = ir::ExistentialProjection<TyCtxt<'tcx>>;
pub type TraitClause<'tcx> = ir::TraitClause<TyCtxt<'tcx>>;
pub type HostEffectClause<'tcx> = ir::HostEffectClause<TyCtxt<'tcx>>;
pub type ClauseKind<'tcx> = ir::ClauseKind<TyCtxt<'tcx>>;
pub type PredicateKind<'tcx> = ir::PredicateKind<TyCtxt<'tcx>>;
pub type NormalizesTo<'tcx> = ir::NormalizesTo<TyCtxt<'tcx>>;
pub type CoercePredicate<'tcx> = ir::CoercePredicate<TyCtxt<'tcx>>;
pub type SubtypePredicate<'tcx> = ir::SubtypePredicate<TyCtxt<'tcx>>;
pub type OutlivesClause<'tcx, T> = ir::OutlivesClause<TyCtxt<'tcx>, T>;
pub type RegionOutlivesClause<'tcx> = OutlivesClause<'tcx, ty::Region<'tcx>>;
pub type TypeOutlivesClause<'tcx> = OutlivesClause<'tcx, Ty<'tcx>>;
pub type ArgOutlivesClause<'tcx> = OutlivesClause<'tcx, ty::GenericArg<'tcx>>;
pub type RegionEqPredicate<'tcx> = ir::RegionEqPredicate<TyCtxt<'tcx>>;
pub type RegionConstraint<'tcx> = ir::RegionConstraint<TyCtxt<'tcx>>;
pub type PolyTraitClause<'tcx> = ty::Binder<'tcx, TraitClause<'tcx>>;
pub type PolyRegionOutlivesClause<'tcx> = ty::Binder<'tcx, RegionOutlivesClause<'tcx>>;
pub type PolyTypeOutlivesClause<'tcx> = ty::Binder<'tcx, TypeOutlivesClause<'tcx>>;
pub type PolySubtypePredicate<'tcx> = ty::Binder<'tcx, SubtypePredicate<'tcx>>;
pub type PolyCoercePredicate<'tcx> = ty::Binder<'tcx, CoercePredicate<'tcx>>;
pub type PolyProjectionClause<'tcx> = ty::Binder<'tcx, ProjectionClause<'tcx>>;

/// An interned statement the trait solver can prove.
pub type Predicate<'tcx> = ir::predicates::Predicate<TyCtxt<'tcx>>;

/// An interned predicate that can be assumed by the solver.
pub type Clause<'tcx> = ir::predicates::Clause<TyCtxt<'tcx>>;

impl<'tcx> rustc_type_ir::inherent::Clause<TyCtxt<'tcx>> for Clause<'tcx> {
    fn as_predicate(self) -> Predicate<'tcx> {
        self.as_predicate()
    }

    fn instantiate_supertrait(self, tcx: TyCtxt<'tcx>, trait_ref: ty::PolyTraitRef<'tcx>) -> Self {
        self.instantiate_supertrait(tcx, trait_ref)
    }
}

// Preserve rustc's existing diagnostic formatting while the concrete
// predicate wrappers are being migrated into rustc_type_ir.
impl<'tcx> ir::ir_print::PredicateDiagFormatter for TyCtxt<'tcx> {
    fn predicate_diag_string(
        predicate: ir::predicates::Predicate<Self>,
        path: &mut Option<std::path::PathBuf>,
    ) -> String {
        ty::tls::with(|tcx| tcx.short_string(tcx.lift(predicate), path))
    }

    fn clause_diag_string(
        clause: ir::predicates::Clause<Self>,
        path: &mut Option<std::path::PathBuf>,
    ) -> String {
        ty::tls::with(|tcx| tcx.short_string(tcx.lift(clause), path))
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

impl<'tcx> UpcastFrom<TyCtxt<'tcx>, RegionOutlivesClause<'tcx>> for Predicate<'tcx> {
    fn upcast_from(from: RegionOutlivesClause<'tcx>, tcx: TyCtxt<'tcx>) -> Self {
        ty::Binder::dummy(PredicateKind::Clause(ClauseKind::RegionOutlives(from))).upcast(tcx)
    }
}

impl<'tcx> UpcastFrom<TyCtxt<'tcx>, PolyRegionOutlivesClause<'tcx>> for Predicate<'tcx> {
    fn upcast_from(from: PolyRegionOutlivesClause<'tcx>, tcx: TyCtxt<'tcx>) -> Self {
        from.map_bound(|c| PredicateKind::Clause(ClauseKind::RegionOutlives(c))).upcast(tcx)
    }
}

impl<'tcx> UpcastFrom<TyCtxt<'tcx>, TypeOutlivesClause<'tcx>> for Predicate<'tcx> {
    fn upcast_from(from: TypeOutlivesClause<'tcx>, tcx: TyCtxt<'tcx>) -> Self {
        ty::Binder::dummy(PredicateKind::Clause(ClauseKind::TypeOutlives(from))).upcast(tcx)
    }
}

// Some types are used a lot. Make sure they don't unintentionally get bigger.
#[cfg(target_pointer_width = "64")]
mod size_asserts {
    use rustc_data_structures::static_assert_size;

    use super::*;
    // tidy-alphabetical-start
    static_assert_size!(PredicateKind<'_>, 40);
    static_assert_size!(rustc_type_ir::WithCachedTypeInfo<PredicateKind<'_>>, 48);
    // tidy-alphabetical-end
}
