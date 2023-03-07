use super::ItemCtxt;
use crate::astconv::{AstConv, OnlySelfBounds};
use rustc_hir as hir;
use rustc_infer::traits::util;
use rustc_middle::ty::subst::InternalSubsts;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_span::def_id::{DefId, LocalDefId};
use rustc_span::Span;

/// For associated types we include both bounds written on the type
/// (`type X: Trait`) and predicates from the trait: `where Self::X: Trait`.
///
/// Note that this filtering is done with the items identity substs to
/// simplify checking that these bounds are met in impls. This means that
/// a bound such as `for<'b> <Self as X<'b>>::U: Clone` can't be used, as in
/// `hr-associated-type-bound-1.rs`.
fn associated_type_bounds<'tcx>(
    tcx: TyCtxt<'tcx>,
    assoc_item_def_id: LocalDefId,
    ast_bounds: &'tcx [hir::GenericBound<'tcx>],
    span: Span,
) -> &'tcx [(ty::Predicate<'tcx>, Span)] {
    let item_ty = tcx.mk_projection(
        assoc_item_def_id.to_def_id(),
        InternalSubsts::identity_for_item(tcx, assoc_item_def_id),
    );

    let icx = ItemCtxt::new(tcx, assoc_item_def_id);
    let mut bounds = icx.astconv().compute_bounds(item_ty, ast_bounds, OnlySelfBounds(false));
    // Associated types are implicitly sized unless a `?Sized` bound is found
    icx.astconv().add_implicitly_sized(&mut bounds, item_ty, ast_bounds, None, span);

    let trait_def_id = tcx.local_parent(assoc_item_def_id);
    let trait_predicates = tcx.trait_explicit_predicates_and_bounds(trait_def_id);

    let bounds_from_parent = trait_predicates.predicates.iter().copied().filter(|(pred, _)| {
        match pred.kind().skip_binder() {
            ty::PredicateKind::Clause(ty::Clause::Trait(tr)) => tr.self_ty() == item_ty,
            ty::PredicateKind::Clause(ty::Clause::Projection(proj)) => {
                proj.projection_ty.self_ty() == item_ty
            }
            ty::PredicateKind::Clause(ty::Clause::TypeOutlives(outlives)) => outlives.0 == item_ty,
            _ => false,
        }
    });

    let all_bounds = tcx.arena.alloc_from_iter(bounds.predicates().chain(bounds_from_parent));
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
) -> &'tcx [(ty::Predicate<'tcx>, Span)] {
    ty::print::with_no_queries!({
        let icx = ItemCtxt::new(tcx, opaque_def_id);
        let mut bounds = icx.astconv().compute_bounds(item_ty, ast_bounds, OnlySelfBounds(false));
        // Opaque types are implicitly sized unless a `?Sized` bound is found
        icx.astconv().add_implicitly_sized(&mut bounds, item_ty, ast_bounds, None, span);
        debug!(?bounds);

        tcx.arena.alloc_from_iter(bounds.predicates())
    })
}

pub(super) fn explicit_item_bounds(
    tcx: TyCtxt<'_>,
    def_id: LocalDefId,
) -> ty::EarlyBinder<&'_ [(ty::Predicate<'_>, Span)]> {
    match tcx.opt_rpitit_info(def_id.to_def_id()) {
        // RPITIT's bounds are the same as opaque type bounds, but with
        // a projection self type.
        Some(ty::ImplTraitInTraitData::Trait { opaque_def_id, .. }) => {
            let item = tcx.hir().get_by_def_id(opaque_def_id.expect_local()).expect_item();
            let opaque_ty = item.expect_opaque_ty();
            return ty::EarlyBinder::bind(opaque_type_bounds(
                tcx,
                opaque_def_id.expect_local(),
                opaque_ty.bounds,
                tcx.mk_projection(
                    def_id.to_def_id(),
                    ty::InternalSubsts::identity_for_item(tcx, def_id),
                ),
                item.span,
            ));
        }
        // These should have been fed!
        Some(ty::ImplTraitInTraitData::Impl { .. }) => unreachable!(),
        None => {}
    }

    let hir_id = tcx.hir().local_def_id_to_hir_id(def_id);
    let bounds = match tcx.hir().get(hir_id) {
        hir::Node::TraitItem(hir::TraitItem {
            kind: hir::TraitItemKind::Type(bounds, _),
            span,
            ..
        }) => associated_type_bounds(tcx, def_id, bounds, *span),
        hir::Node::Item(hir::Item {
            kind: hir::ItemKind::OpaqueTy(hir::OpaqueTy { bounds, in_trait, .. }),
            span,
            ..
        }) => {
            let substs = InternalSubsts::identity_for_item(tcx, def_id);
            let item_ty = if *in_trait && !tcx.lower_impl_trait_in_trait_to_assoc_ty() {
                tcx.mk_projection(def_id.to_def_id(), substs)
            } else {
                tcx.mk_opaque(def_id.to_def_id(), substs)
            };
            opaque_type_bounds(tcx, def_id, bounds, item_ty, *span)
        }
        hir::Node::Item(hir::Item { kind: hir::ItemKind::TyAlias(..), .. }) => &[],
        _ => bug!("item_bounds called on {:?}", def_id),
    };
    ty::EarlyBinder::bind(bounds)
}

pub(super) fn item_bounds(
    tcx: TyCtxt<'_>,
    def_id: DefId,
) -> ty::EarlyBinder<&'_ ty::List<ty::Predicate<'_>>> {
    tcx.explicit_item_bounds(def_id).map_bound(|bounds| {
        tcx.mk_predicates_from_iter(util::elaborate(
            tcx,
            bounds.iter().map(|&(bound, _span)| bound),
        ))
    })
}
