use rustc_hir::lang_items::LangItem;
use rustc_infer::infer::TyCtxtInferExt;
use rustc_middle::query::Providers;
use rustc_middle::ty::{self, Ty, TyCtxt, TypingMode};
use rustc_trait_selection::traits::{ObligationCause, ObligationCtxt};

/// This method returns true if and only if `adt_ty` itself has been marked as
/// eligible for structural-match: namely, if it implements
/// `StructuralPartialEq` (which is injected by `#[derive(PartialEq)]`).
///
/// Note that this does *not* recursively check if the substructure of `adt_ty`
/// implements the trait.
fn has_structural_eq_impl<'tcx>(tcx: TyCtxt<'tcx>, adt_ty: Ty<'tcx>) -> bool {
    let infcx = &tcx.infer_ctxt().build(TypingMode::non_body_analysis());
    let cause = ObligationCause::dummy();

    let ocx = ObligationCtxt::new(infcx);
    // require `#[derive(PartialEq)]`
    let structural_peq_def_id = infcx.tcx.require_lang_item(LangItem::StructuralPeq, cause.span);
    ocx.register_bound(cause.clone(), ty::ParamEnv::empty(), adt_ty, structural_peq_def_id);

    // We deliberately skip *reporting* fulfillment errors (via
    // `report_fulfillment_errors`), for two reasons:
    //
    // 1. The error messages would mention `std::marker::StructuralPartialEq`
    //    (a trait which is solely meant as an implementation detail
    //    for now), and
    //
    // 2. We are sometimes doing future-incompatibility lints for
    //    now, so we do not want unconditional errors here.
    ocx.select_all_or_error().is_empty()
}

pub(crate) fn provide(providers: &mut Providers) {
    providers.has_structural_eq_impl = has_structural_eq_impl;
}
