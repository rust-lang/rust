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
) -> Option<usize> {
    // It is decided that a query to cache these results is not necessary
    // for error reporting.
    // We can afford to recompute it on-demand.
    if tcx.lang_items().get(LangItem::CoercePointeeValidated).map_or(false, |did| {
        tcx.trait_impls_of(did).non_blanket_impls().contains_key(&SimplifiedType::Adt(adt_did))
    }) {
        // Search for the `#[pointee]`
        enum Pointee {
            None,
            First(usize),
            Ambiguous,
        }
        let mut first_type = Pointee::None;
        for (idx, param) in tcx.generics_of(adt_did).own_params.iter().enumerate() {
            if let GenericParamDefKind::Type { .. } = param.kind {
                match first_type {
                    Pointee::None => {
                        first_type = Pointee::First(idx);
                    }
                    Pointee::First(_) => first_type = Pointee::Ambiguous,
                    Pointee::Ambiguous => {}
                }
            }
            if tcx.has_attr(param.def_id, sym::pointee) {
                return Some(idx);
            }
        }
        if let Pointee::First(idx) = first_type {
            return Some(idx);
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
        fn visit_ty(&mut self, t: Ty<'tcx>) {
            if t == self.pointee {
                self.found = true;
            } else {
                t.super_visit_with(self)
            }
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
        if tcx.is_lang_item(pred.def_id(), LangItem::DispatchFromDyn) && mentions_pointee() {
            tcx.dcx().emit_err(errors::CoercePointeeCannotDispatchFromDyn {
                ty: source.to_string(),
                span,
            });
            return None;
        }
        if tcx.is_lang_item(pred.def_id(), LangItem::Unsize) && mentions_pointee() {
            // We should redact it
            tcx.dcx().emit_err(errors::CoercePointeeCannotUnsize { ty: source.to_string(), span });
            return None;
        }
        if tcx.is_lang_item(pred.def_id(), LangItem::CoerceUnsized) && mentions_pointee() {
            // We should redact it
            tcx.dcx()
                .emit_err(errors::CoercePointeeCannotCoerceUnsize { ty: source.to_string(), span });
            return None;
        }
    }
    Some(err)
}
