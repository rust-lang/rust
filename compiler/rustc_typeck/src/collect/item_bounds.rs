use rustc_hir::def::DefKind;
use rustc_infer::traits::util;
use rustc_middle::ty::subst::InternalSubsts;
use rustc_middle::ty::{self, TyCtxt};
use rustc_span::def_id::DefId;

/// For associated types we allow bounds written on the associated type
/// (`type X: Trait`) to be used as candidates. We also allow the same bounds
/// when desugared as bounds on the trait `where Self::X: Trait`.
///
/// Note that this filtering is done with the items identity substs to
/// simplify checking that these bounds are met in impls. This means that
/// a bound such as `for<'b> <Self as X<'b>>::U: Clone` can't be used, as in
/// `hr-associated-type-bound-1.rs`.
fn associated_type_bounds(
    tcx: TyCtxt<'_>,
    assoc_item_def_id: DefId,
) -> &'_ ty::List<ty::Predicate<'_>> {
    let generic_trait_bounds = tcx.predicates_of(assoc_item_def_id);
    // We include predicates from the trait as well to handle
    // `where Self::X: Trait`.
    let item_bounds = generic_trait_bounds.instantiate_identity(tcx);
    let item_predicates = util::elaborate_predicates(tcx, item_bounds.predicates.into_iter());

    let assoc_item_ty = ty::ProjectionTy {
        item_def_id: assoc_item_def_id,
        substs: InternalSubsts::identity_for_item(tcx, assoc_item_def_id),
    };

    let predicates = item_predicates.filter_map(|obligation| {
        let pred = obligation.predicate;
        match pred.kind() {
            ty::PredicateKind::Trait(tr, _) => {
                if let ty::Projection(p) = *tr.skip_binder().self_ty().kind() {
                    if p == assoc_item_ty {
                        return Some(pred);
                    }
                }
            }
            ty::PredicateKind::Projection(proj) => {
                if let ty::Projection(p) = *proj.skip_binder().projection_ty.self_ty().kind() {
                    if p == assoc_item_ty {
                        return Some(pred);
                    }
                }
            }
            ty::PredicateKind::TypeOutlives(outlives) => {
                if let ty::Projection(p) = *outlives.skip_binder().0.kind() {
                    if p == assoc_item_ty {
                        return Some(pred);
                    }
                }
            }
            _ => {}
        }
        None
    });

    let result = tcx.mk_predicates(predicates);
    debug!("associated_type_bounds({}) = {:?}", tcx.def_path_str(assoc_item_def_id), result);
    result
}

/// Opaque types don't have the same issues as associated types: the only
/// predicates on an opaque type (excluding those it inherits from its parent
/// item) should be of the form we're expecting.
fn opaque_type_bounds(tcx: TyCtxt<'_>, def_id: DefId) -> &'_ ty::List<ty::Predicate<'_>> {
    let substs = InternalSubsts::identity_for_item(tcx, def_id);

    let bounds = tcx.predicates_of(def_id);
    let predicates =
        util::elaborate_predicates(tcx, bounds.predicates.iter().map(|&(pred, _)| pred));

    let filtered_predicates = predicates.filter_map(|obligation| {
        let pred = obligation.predicate;
        match pred.kind() {
            ty::PredicateKind::Trait(tr, _) => {
                if let ty::Opaque(opaque_def_id, opaque_substs) = *tr.skip_binder().self_ty().kind()
                {
                    if opaque_def_id == def_id && opaque_substs == substs {
                        return Some(pred);
                    }
                }
            }
            ty::PredicateKind::Projection(proj) => {
                if let ty::Opaque(opaque_def_id, opaque_substs) =
                    *proj.skip_binder().projection_ty.self_ty().kind()
                {
                    if opaque_def_id == def_id && opaque_substs == substs {
                        return Some(pred);
                    }
                }
            }
            ty::PredicateKind::TypeOutlives(outlives) => {
                if let ty::Opaque(opaque_def_id, opaque_substs) = *outlives.skip_binder().0.kind() {
                    if opaque_def_id == def_id && opaque_substs == substs {
                        return Some(pred);
                    }
                } else {
                    // These can come from elaborating other predicates
                    return None;
                }
            }
            // These can come from elaborating other predicates
            ty::PredicateKind::RegionOutlives(_) => return None,
            _ => {}
        }
        tcx.sess.delay_span_bug(
            obligation.cause.span(tcx),
            &format!("unexpected predicate {:?} on opaque type", pred),
        );
        None
    });

    let result = tcx.mk_predicates(filtered_predicates);
    debug!("opaque_type_bounds({}) = {:?}", tcx.def_path_str(def_id), result);
    result
}

pub(super) fn item_bounds(tcx: TyCtxt<'_>, def_id: DefId) -> &'_ ty::List<ty::Predicate<'_>> {
    match tcx.def_kind(def_id) {
        DefKind::AssocTy => associated_type_bounds(tcx, def_id),
        DefKind::OpaqueTy => opaque_type_bounds(tcx, def_id),
        k => bug!("item_bounds called on {}", k.descr(def_id)),
    }
}
