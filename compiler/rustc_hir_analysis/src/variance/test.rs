use rustc_hir::def::DefKind;
use rustc_hir::def_id::CRATE_DEF_ID;
use rustc_middle::ty::TyCtxt;
use rustc_span::symbol::sym;
use rustc_span::ErrorGuaranteed;

use crate::errors;

pub fn test_variance(tcx: TyCtxt<'_>) -> Result<(), ErrorGuaranteed> {
    let mut res = Ok(());
    if tcx.has_attr(CRATE_DEF_ID, sym::rustc_variance_of_opaques) {
        for id in tcx.hir().items() {
            if matches!(tcx.def_kind(id.owner_id), DefKind::OpaqueTy) {
                let variances_of = tcx.variances_of(id.owner_id);

                res = Err(tcx.dcx().emit_err(errors::VariancesOf {
                    span: tcx.def_span(id.owner_id),
                    variances_of: format!("{variances_of:?}"),
                }));
            }
        }
    }

    // For unit testing: check for a special "rustc_variance"
    // attribute and report an error with various results if found.
    for id in tcx.hir().items() {
        if tcx.has_attr(id.owner_id, sym::rustc_variance) {
            let variances_of = tcx.variances_of(id.owner_id);

            res = Err(tcx.dcx().emit_err(errors::VariancesOf {
                span: tcx.def_span(id.owner_id),
                variances_of: format!("{variances_of:?}"),
            }));
        }
    }
    res
}
