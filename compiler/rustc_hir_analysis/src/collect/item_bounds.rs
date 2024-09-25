use rustc_data_structures::fx::FxIndexSet;
use rustc_hir as hir;
use rustc_infer::traits::util;
use rustc_middle::ty::{
    self, GenericArgs, Ty, TyCtxt, TypeFoldable, TypeFolder, TypeSuperFoldable,
};
use rustc_middle::{bug, span_bug};
use rustc_span::Span;
use rustc_span::def_id::{DefId, LocalDefId};
use rustc_type_ir::Upcast;
use tracing::{debug, instrument};

use super::ItemCtxt;
use super::predicates_of::assert_only_contains_predicates_from;
use crate::hir_ty_lowering::{HirTyLowerer, PredicateFilter};

/// For associated types we include both bounds written on the type
/// (`type X: Trait`) and predicates from the trait: `where Self::X: Trait`.
///
/// Note that this filtering is done with the items identity args to
/// simplify checking that these bounds are met in impls. This means that
/// a bound such as `for<'b> <Self as X<'b>>::U: Clone` can't be used, as in
/// `hr-associated-type-bound-1.rs`.
fn associated_type_bounds<'tcx>(
    tcx: TyCtxt<'tcx>,
    assoc_item_def_id: LocalDefId,
    hir_bounds: &'tcx [hir::GenericBound<'tcx>],
    span: Span,
    filter: PredicateFilter,
) -> &'tcx [(ty::Clause<'tcx>, Span)] {
    let item_ty = Ty::new_projection_from_args(
        tcx,
        assoc_item_def_id.to_def_id(),
        GenericArgs::identity_for_item(tcx, assoc_item_def_id),
    );

    let icx = ItemCtxt::new(tcx, assoc_item_def_id);
    let mut bounds = icx.lowerer().lower_mono_bounds(item_ty, hir_bounds, filter);
    // Associated types are implicitly sized unless a `?Sized` bound is found
    icx.lowerer().add_sized_bound(&mut bounds, item_ty, hir_bounds, None, span);

    let trait_def_id = tcx.local_parent(assoc_item_def_id);
    let trait_predicates = tcx.trait_explicit_predicates_and_bounds(trait_def_id);

    let bounds_from_parent = trait_predicates.predicates.iter().copied().filter(|(pred, _)| {
        match pred.kind().skip_binder() {
            ty::ClauseKind::Trait(tr) => tr.self_ty() == item_ty,
            ty::ClauseKind::Projection(proj) => proj.projection_term.self_ty() == item_ty,
            ty::ClauseKind::TypeOutlives(outlives) => outlives.0 == item_ty,
            _ => false,
        }
    });

    let all_bounds = tcx.arena.alloc_from_iter(bounds.clauses(tcx).chain(bounds_from_parent));
    debug!(
        "associated_type_bounds({}) = {:?}",
        tcx.def_path_str(assoc_item_def_id.to_def_id()),
        all_bounds
    );

    assert_only_contains_predicates_from(filter, all_bounds, item_ty);

    all_bounds
}

/// Opaque types don't inherit bounds from their parent: for return position
/// impl trait it isn't possible to write a suitable predicate on the
/// containing function and for type-alias impl trait we don't have a backwards
/// compatibility issue.
#[instrument(level = "trace", skip(tcx, item_ty))]
fn opaque_type_bounds<'tcx>(
    tcx: TyCtxt<'tcx>,
    opaque_def_id: LocalDefId,
    hir_bounds: &'tcx [hir::GenericBound<'tcx>],
    item_ty: Ty<'tcx>,
    span: Span,
    filter: PredicateFilter,
) -> &'tcx [(ty::Clause<'tcx>, Span)] {
    ty::print::with_reduced_queries!({
        let icx = ItemCtxt::new(tcx, opaque_def_id);
        let mut bounds = icx.lowerer().lower_mono_bounds(item_ty, hir_bounds, filter);
        // Opaque types are implicitly sized unless a `?Sized` bound is found
        icx.lowerer().add_sized_bound(&mut bounds, item_ty, hir_bounds, None, span);
        debug!(?bounds);

        tcx.arena.alloc_from_iter(bounds.clauses(tcx))
    })
}

pub(super) fn explicit_item_bounds(
    tcx: TyCtxt<'_>,
    def_id: LocalDefId,
) -> ty::EarlyBinder<'_, &'_ [(ty::Clause<'_>, Span)]> {
    explicit_item_bounds_with_filter(tcx, def_id, PredicateFilter::All)
}

pub(super) fn explicit_item_super_predicates(
    tcx: TyCtxt<'_>,
    def_id: LocalDefId,
) -> ty::EarlyBinder<'_, &'_ [(ty::Clause<'_>, Span)]> {
    explicit_item_bounds_with_filter(tcx, def_id, PredicateFilter::SelfOnly)
}

pub(super) fn explicit_item_bounds_with_filter(
    tcx: TyCtxt<'_>,
    def_id: LocalDefId,
    filter: PredicateFilter,
) -> ty::EarlyBinder<'_, &'_ [(ty::Clause<'_>, Span)]> {
    match tcx.opt_rpitit_info(def_id.to_def_id()) {
        // RPITIT's bounds are the same as opaque type bounds, but with
        // a projection self type.
        Some(ty::ImplTraitInTraitData::Trait { opaque_def_id, .. }) => {
            let item = tcx.hir_node_by_def_id(opaque_def_id.expect_local()).expect_item();
            let opaque_ty = item.expect_opaque_ty();
            let item_ty = Ty::new_projection_from_args(
                tcx,
                def_id.to_def_id(),
                ty::GenericArgs::identity_for_item(tcx, def_id),
            );
            let bounds = opaque_type_bounds(
                tcx,
                opaque_def_id.expect_local(),
                opaque_ty.bounds,
                item_ty,
                item.span,
                filter,
            );
            assert_only_contains_predicates_from(filter, bounds, item_ty);
            return ty::EarlyBinder::bind(bounds);
        }
        Some(ty::ImplTraitInTraitData::Impl { .. }) => span_bug!(
            tcx.def_span(def_id),
            "item bounds for RPITIT in impl to be fed on def-id creation"
        ),
        None => {}
    }

    if tcx.is_effects_desugared_assoc_ty(def_id.to_def_id()) {
        let mut predicates = Vec::new();

        let parent = tcx.local_parent(def_id);

        let preds = tcx.explicit_predicates_of(parent);

        if let ty::AssocItemContainer::TraitContainer = tcx.associated_item(def_id).container {
            // for traits, emit `type Effects: TyCompat<<(T1::Effects, ..) as Min>::Output>`
            let tup = Ty::new(tcx, ty::Tuple(preds.effects_min_tys));
            // FIXME(effects) span
            let span = tcx.def_span(def_id);
            let assoc = tcx.require_lang_item(hir::LangItem::EffectsIntersectionOutput, Some(span));
            let proj = Ty::new_projection(tcx, assoc, [tup]);
            let self_proj = Ty::new_projection(
                tcx,
                def_id.to_def_id(),
                ty::GenericArgs::identity_for_item(tcx, def_id),
            );
            let trait_ = tcx.require_lang_item(hir::LangItem::EffectsTyCompat, Some(span));
            let trait_ref = ty::TraitRef::new(tcx, trait_, [self_proj, proj]);
            predicates.push((ty::Binder::dummy(trait_ref).upcast(tcx), span));
        }
        return ty::EarlyBinder::bind(tcx.arena.alloc_from_iter(predicates));
    }

    let bounds = match tcx.hir_node_by_def_id(def_id) {
        hir::Node::TraitItem(hir::TraitItem {
            kind: hir::TraitItemKind::Type(bounds, _),
            span,
            ..
        }) => associated_type_bounds(tcx, def_id, bounds, *span, filter),
        hir::Node::Item(hir::Item {
            kind: hir::ItemKind::OpaqueTy(hir::OpaqueTy { bounds, in_trait: false, .. }),
            span,
            ..
        }) => {
            let args = GenericArgs::identity_for_item(tcx, def_id);
            let item_ty = Ty::new_opaque(tcx, def_id.to_def_id(), args);
            let bounds = opaque_type_bounds(tcx, def_id, bounds, item_ty, *span, filter);
            assert_only_contains_predicates_from(filter, bounds, item_ty);
            bounds
        }
        // Since RPITITs are lowered as projections in `<dyn HirTyLowerer>::lower_ty`, when we're
        // asking for the item bounds of the *opaques* in a trait's default method signature, we
        // need to map these projections back to opaques.
        hir::Node::Item(hir::Item {
            kind: hir::ItemKind::OpaqueTy(hir::OpaqueTy { bounds, in_trait: true, origin, .. }),
            span,
            ..
        }) => {
            let (hir::OpaqueTyOrigin::FnReturn(fn_def_id)
            | hir::OpaqueTyOrigin::AsyncFn(fn_def_id)) = *origin
            else {
                span_bug!(*span, "RPITIT cannot be a TAIT, but got origin {origin:?}");
            };
            let args = GenericArgs::identity_for_item(tcx, def_id);
            let item_ty = Ty::new_opaque(tcx, def_id.to_def_id(), args);
            let bounds = &*tcx.arena.alloc_slice(
                &opaque_type_bounds(tcx, def_id, bounds, item_ty, *span, filter)
                    .to_vec()
                    .fold_with(&mut AssocTyToOpaque { tcx, fn_def_id: fn_def_id.to_def_id() }),
            );
            assert_only_contains_predicates_from(filter, bounds, item_ty);
            bounds
        }
        hir::Node::Item(hir::Item { kind: hir::ItemKind::TyAlias(..), .. }) => &[],
        _ => bug!("item_bounds called on {:?}", def_id),
    };

    ty::EarlyBinder::bind(bounds)
}

pub(super) fn item_bounds(tcx: TyCtxt<'_>, def_id: DefId) -> ty::EarlyBinder<'_, ty::Clauses<'_>> {
    tcx.explicit_item_bounds(def_id).map_bound(|bounds| {
        tcx.mk_clauses_from_iter(util::elaborate(tcx, bounds.iter().map(|&(bound, _span)| bound)))
    })
}

pub(super) fn item_super_predicates(
    tcx: TyCtxt<'_>,
    def_id: DefId,
) -> ty::EarlyBinder<'_, ty::Clauses<'_>> {
    tcx.explicit_item_super_predicates(def_id).map_bound(|bounds| {
        tcx.mk_clauses_from_iter(
            util::elaborate(tcx, bounds.iter().map(|&(bound, _span)| bound)).filter_only_self(),
        )
    })
}

/// This exists as an optimization to compute only the item bounds of the item
/// that are not `Self` bounds.
pub(super) fn item_non_self_assumptions(
    tcx: TyCtxt<'_>,
    def_id: DefId,
) -> ty::EarlyBinder<'_, ty::Clauses<'_>> {
    let all_bounds: FxIndexSet<_> = tcx.item_bounds(def_id).skip_binder().iter().collect();
    let own_bounds: FxIndexSet<_> =
        tcx.item_super_predicates(def_id).skip_binder().iter().collect();
    if all_bounds.len() == own_bounds.len() {
        ty::EarlyBinder::bind(ty::ListWithCachedTypeInfo::empty())
    } else {
        ty::EarlyBinder::bind(tcx.mk_clauses_from_iter(all_bounds.difference(&own_bounds).copied()))
    }
}

/// This exists as an optimization to compute only the supertraits of this impl's
/// trait that are outlives bounds.
pub(super) fn impl_super_outlives(
    tcx: TyCtxt<'_>,
    def_id: DefId,
) -> ty::EarlyBinder<'_, ty::Clauses<'_>> {
    tcx.impl_trait_header(def_id).expect("expected an impl of trait").trait_ref.map_bound(
        |trait_ref| {
            let clause: ty::Clause<'_> = trait_ref.upcast(tcx);
            tcx.mk_clauses_from_iter(util::elaborate(tcx, [clause]).filter(|clause| {
                matches!(
                    clause.kind().skip_binder(),
                    ty::ClauseKind::TypeOutlives(_) | ty::ClauseKind::RegionOutlives(_)
                )
            }))
        },
    )
}

struct AssocTyToOpaque<'tcx> {
    tcx: TyCtxt<'tcx>,
    fn_def_id: DefId,
}

impl<'tcx> TypeFolder<TyCtxt<'tcx>> for AssocTyToOpaque<'tcx> {
    fn cx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
        if let ty::Alias(ty::Projection, projection_ty) = ty.kind()
            && let Some(ty::ImplTraitInTraitData::Trait { fn_def_id, .. }) =
                self.tcx.opt_rpitit_info(projection_ty.def_id)
            && fn_def_id == self.fn_def_id
        {
            self.tcx.type_of(projection_ty.def_id).instantiate(self.tcx, projection_ty.args)
        } else {
            ty.super_fold_with(self)
        }
    }
}
