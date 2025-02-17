use rustc_hir::LangItem;
use rustc_hir::def_id::DefId;
use rustc_middle::ty::fast_reject::SimplifiedType;
use rustc_middle::ty::{
    self, GenericParamDefKind, Ty, TyCtxt, TypeSuperVisitable, TypeVisitable, TypeVisitor,
};
use rustc_span::{Span, sym};
use rustc_trait_selection::traits::FulfillmentError;
use tracing::instrument;

use crate::errors;

#[instrument(level = "debug", skip(tcx), ret)]
pub(super) fn extract_coerce_pointee_data<'tcx>(
    tcx: TyCtxt<'tcx>,
    adt_did: DefId,
) -> Option<(usize, DefId)> {
    // It is decided that a query to cache these results is not necessary
    // for error reporting.
    // We can afford to recompute it on-demand.
    if let Some(impls) = tcx.lang_items().get(LangItem::CoercePointeeValidated).and_then(|did| {
        tcx.trait_impls_of(did).non_blanket_impls().get(&SimplifiedType::Adt(adt_did))
    }) && let [impl_did, ..] = impls[..]
    {
        // Search for the `#[pointee]`
        let mut first_type = None;
        for (idx, param) in tcx.generics_of(adt_did).own_params.iter().enumerate() {
            if let GenericParamDefKind::Type { .. } = param.kind {
                first_type = if first_type.is_some() { None } else { Some(idx) };
            }
            if tcx.has_attr(param.def_id, sym::pointee) {
                return Some((idx, impl_did));
            }
        }
        if let Some(idx) = first_type {
            return Some((idx, impl_did));
        }
    }
    None
}

fn contains_coerce_pointee_target_pointee<'tcx>(ty: Ty<'tcx>, target_pointee_ty: Ty<'tcx>) -> bool {
    struct Search<'tcx> {
        pointee: Ty<'tcx>,
        found: bool,
    }
    impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for Search<'tcx> {
        fn visit_binder<T: ty::TypeVisitable<TyCtxt<'tcx>>>(
            &mut self,
            t: &rustc_type_ir::Binder<TyCtxt<'tcx>, T>,
        ) {
            if self.found {
                return;
            }
            t.super_visit_with(self)
        }

        fn visit_ty(&mut self, t: Ty<'tcx>) {
            if self.found {
                return;
            }
            if t == self.pointee {
                self.found = true;
            } else {
                t.super_visit_with(self)
            }
        }

        fn visit_region(&mut self, r: <TyCtxt<'tcx> as ty::Interner>::Region) {
            if self.found {
                return;
            }
            if let rustc_type_ir::ReError(guar) = r.kind() {
                self.visit_error(guar)
            }
        }

        fn visit_const(&mut self, c: <TyCtxt<'tcx> as ty::Interner>::Const) {
            if self.found {
                return;
            }
            c.super_visit_with(self)
        }

        fn visit_predicate(&mut self, p: <TyCtxt<'tcx> as ty::Interner>::Predicate) {
            if self.found {
                return;
            }
            p.super_visit_with(self)
        }

        fn visit_clauses(&mut self, p: <TyCtxt<'tcx> as ty::Interner>::Clauses) {
            if self.found {
                return;
            }
            p.super_visit_with(self)
        }
    }
    let mut search = Search { pointee: target_pointee_ty, found: false };
    ty.visit_with(&mut search);
    search.found
}

#[instrument(level = "debug", skip(tcx))]
pub(super) fn redact_fulfillment_err_for_coerce_pointee<'tcx>(
    tcx: TyCtxt<'tcx>,
    err: FulfillmentError<'tcx>,
    target_pointee_ty: Ty<'tcx>,
    span: Span,
) -> Option<FulfillmentError<'tcx>> {
    if let ty::PredicateKind::Clause(ty::ClauseKind::Trait(pred)) =
        err.obligation.predicate.kind().skip_binder()
    {
        let mentions_pointee = || {
            contains_coerce_pointee_target_pointee(
                pred.trait_ref.args.type_at(1),
                target_pointee_ty,
            )
        };
        let source = pred.trait_ref.self_ty();
        if tcx.is_lang_item(pred.def_id(), LangItem::Unsize) {
            if mentions_pointee() {
                // We should redact it
                tcx.dcx()
                    .emit_err(errors::CoercePointeeCannotUnsize { ty: source.to_string(), span });
                return None;
            }
        } else if tcx.is_lang_item(pred.def_id(), LangItem::CoerceUnsized) && mentions_pointee() {
            // We should redact it
            tcx.dcx()
                .emit_err(errors::CoercePointeeCannotCoerceUnsize { ty: source.to_string(), span });
            return None;
        }
    }
    Some(err)
}
