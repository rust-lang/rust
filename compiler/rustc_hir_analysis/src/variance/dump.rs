use rustc_hir::def::DefKind;
use rustc_hir::def_id::CRATE_DEF_ID;
use rustc_middle::ty::TyCtxt;
use rustc_span::symbol::sym;

pub(crate) fn variances(tcx: TyCtxt<'_>) {
    if tcx.has_attr(CRATE_DEF_ID, sym::rustc_variance_of_opaques) {
        for id in tcx.hir().items() {
            let DefKind::OpaqueTy = tcx.def_kind(id.owner_id) else { continue };

            let variances = tcx.variances_of(id.owner_id);

            tcx.dcx().emit_err(crate::errors::VariancesOf {
                span: tcx.def_span(id.owner_id),
                variances: format!("{variances:?}"),
            });
        }
    }

    for id in tcx.hir().items() {
        if !tcx.has_attr(id.owner_id, sym::rustc_variance) {
            continue;
        }

        let variances = tcx.variances_of(id.owner_id);

        tcx.dcx().emit_err(crate::errors::VariancesOf {
            span: tcx.def_span(id.owner_id),
            variances: format!("{variances:?}"),
        });
    }
}
