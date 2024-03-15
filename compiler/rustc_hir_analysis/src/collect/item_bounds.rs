use super::ItemCtxt;
use crate::astconv::{HirTyLowerer, PredicateFilter};
use rustc_data_structures::fx::FxIndexSet;
use rustc_hir as hir;
use rustc_infer::traits::util;
use rustc_middle::ty::GenericArgs;
use rustc_middle::ty::{self, Ty, TyCtxt, TypeFoldable, TypeFolder};
use rustc_span::def_id::{DefId, LocalDefId};
use rustc_span::Span;

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
    ast_bounds: &'tcx [hir::GenericBound<'tcx>],
    span: Span,
    filter: PredicateFilter,
) -> &'tcx [(ty::Clause<'tcx>, Span)] {
    let item_ty = Ty::new_projection(
        tcx,
        assoc_item_def_id.to_def_id(),
        GenericArgs::identity_for_item(tcx, assoc_item_def_id),
    );

    let icx = ItemCtxt::new(tcx, assoc_item_def_id);
    let mut bounds = icx.lowerer().lower_mono_bounds(item_ty, ast_bounds, filter);
    // Associated types are implicitly sized unless a `?Sized` bound is found
    icx.lowerer().add_sized_bound(&mut bounds, item_ty, ast_bounds, None, span);

    let trait_def_id = tcx.local_parent(assoc_item_def_id);
    let trait_predicates = tcx.trait_explicit_predicates_and_bounds(trait_def_id);

    let bounds_from_parent = trait_predicates.predicates.iter().copied().filter(|(pred, _)| {
        match pred.kind().skip_binder() {
            ty::ClauseKind::Trait(tr) => tr.self_ty() == item_ty,
            ty::ClauseKind::Projection(proj) => proj.projection_ty.self_ty() == item_ty,
            ty::ClauseKind::TypeOutlives(outlives) => outlives.0 == item_ty,
            _ => false,
        }
    });

    let all_bounds = tcx.arena.alloc_from_iter(bounds.clauses().chain(bounds_from_parent));
    debug!(
        "associated_type_bounds({}) = {:?}",
        tcx.def_path_str(assoc_item_def_id.to_def_id()),
        all_bounds
    );
    all_bounds
}

/// Opaque types don't inherit bounds from their parent: for return position
/// impl trait it isn't possible to write a suitable predicate on the
/// containing function and for type-alias impl trait we don't have a backwards
/// compatibility issue.
#[instrument(level = "trace", skip(tcx), ret)]
fn opaque_type_bounds<'tcx>(
    tcx: TyCtxt<'tcx>,
    opaque_def_id: LocalDefId,
    ast_bounds: &'tcx [hir::GenericBound<'tcx>],
    item_ty: Ty<'tcx>,
    span: Span,
    filter: PredicateFilter,
) -> &'tcx [(ty::Clause<'tcx>, Span)] {
    ty::print::with_reduced_queries!({
        let icx = ItemCtxt::new(tcx, opaque_def_id);
        let mut bounds = icx.lowerer().lower_mono_bounds(item_ty, ast_bounds, filter);
        // Opaque types are implicitly sized unless a `?Sized` bound is found
        icx.lowerer().add_sized_bound(&mut bounds, item_ty, ast_bounds, None, span);
        debug!(?bounds);

        tcx.arena.alloc_from_iter(bounds.clauses())
    })
}

pub(super) fn explicit_item_bounds(
    tcx: TyCtxt<'_>,
    def_id: LocalDefId,
) -> ty::EarlyBinder<&'_ [(ty::Clause<'_>, Span)]> {
    explicit_item_bounds_with_filter(tcx, def_id, PredicateFilter::All)
}

pub(super) fn explicit_item_super_predicates(
    tcx: TyCtxt<'_>,
    def_id: LocalDefId,
) -> ty::EarlyBinder<&'_ [(ty::Clause<'_>, Span)]> {
    explicit_item_bounds_with_filter(tcx, def_id, PredicateFilter::SelfOnly)
}

pub(super) fn explicit_item_bounds_with_filter(
    tcx: TyCtxt<'_>,
    def_id: LocalDefId,
    filter: PredicateFilter,
) -> ty::EarlyBinder<&'_ [(ty::Clause<'_>, Span)]> {
    match tcx.opt_rpitit_info(def_id.to_def_id()) {
        // RPITIT's bounds are the same as opaque type bounds, but with
        // a projection self type.
        Some(ty::ImplTraitInTraitData::Trait { opaque_def_id, .. }) => {
            let item = tcx.hir_node_by_def_id(opaque_def_id.expect_local()).expect_item();
            let opaque_ty = item.expect_opaque_ty();
            return ty::EarlyBinder::bind(opaque_type_bounds(
                tcx,
                opaque_def_id.expect_local(),
                opaque_ty.bounds,
                Ty::new_projection(
                    tcx,
                    def_id.to_def_id(),
                    ty::GenericArgs::identity_for_item(tcx, def_id),
                ),
                item.span,
                filter,
            ));
        }
        Some(ty::ImplTraitInTraitData::Impl { .. }) => span_bug!(
            tcx.def_span(def_id),
            "item bounds for RPITIT in impl to be fed on def-id creation"
        ),
        None => {}
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
            opaque_type_bounds(tcx, def_id, bounds, item_ty, *span, filter)
        }
        // Since RPITITs are astconv'd as projections in `ast_ty_to_ty`, when we're asking
        // for the item bounds of the *opaques* in a trait's default method signature, we
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
            tcx.arena.alloc_slice(
                &opaque_type_bounds(tcx, def_id, bounds, item_ty, *span, filter)
                    .to_vec()
                    .fold_with(&mut AssocTyToOpaque { tcx, fn_def_id: fn_def_id.to_def_id() }),
            )
        }
        hir::Node::Item(hir::Item { kind: hir::ItemKind::TyAlias(..), .. }) => &[],
        _ => bug!("item_bounds called on {:?}", def_id),
    };
    ty::EarlyBinder::bind(bounds)
}

pub(super) fn item_bounds(
    tcx: TyCtxt<'_>,
    def_id: DefId,
) -> ty::EarlyBinder<&'_ ty::List<ty::Clause<'_>>> {
    tcx.explicit_item_bounds(def_id).map_bound(|bounds| {
        tcx.mk_clauses_from_iter(util::elaborate(tcx, bounds.iter().map(|&(bound, _span)| bound)))
    })
}

pub(super) fn item_super_predicates(
    tcx: TyCtxt<'_>,
    def_id: DefId,
) -> ty::EarlyBinder<&'_ ty::List<ty::Clause<'_>>> {
    tcx.explicit_item_super_predicates(def_id).map_bound(|bounds| {
        tcx.mk_clauses_from_iter(
            util::elaborate(tcx, bounds.iter().map(|&(bound, _span)| bound)).filter_only_self(),
        )
    })
}

pub(super) fn item_non_self_assumptions(
    tcx: TyCtxt<'_>,
    def_id: DefId,
) -> ty::EarlyBinder<&'_ ty::List<ty::Clause<'_>>> {
    let all_bounds: FxIndexSet<_> = tcx.item_bounds(def_id).skip_binder().iter().collect();
    let own_bounds: FxIndexSet<_> =
        tcx.item_super_predicates(def_id).skip_binder().iter().collect();
    if all_bounds.len() == own_bounds.len() {
        ty::EarlyBinder::bind(ty::List::empty())
    } else {
        ty::EarlyBinder::bind(tcx.mk_clauses_from_iter(all_bounds.difference(&own_bounds).copied()))
    }
}

struct AssocTyToOpaque<'tcx> {
    tcx: TyCtxt<'tcx>,
    fn_def_id: DefId,
}

impl<'tcx> TypeFolder<TyCtxt<'tcx>> for AssocTyToOpaque<'tcx> {
    fn interner(&self) -> TyCtxt<'tcx> {
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
            ty
        }
    }
}
