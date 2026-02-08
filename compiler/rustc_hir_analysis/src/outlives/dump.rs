use rustc_hir::attrs::AttributeKind;
use rustc_hir::find_attr;
use rustc_middle::bug;
use rustc_middle::ty::{self, TyCtxt};
use rustc_span::sym;

pub(crate) fn inferred_outlives(tcx: TyCtxt<'_>) {
    for id in tcx.hir_free_items() {
        if !find_attr!(tcx.get_all_attrs(id.owner_id), AttributeKind::RustcOutlives) {
            continue;
        }

        let preds = tcx.inferred_outlives_of(id.owner_id);
        let mut preds: Vec<_> = preds
            .iter()
            .map(|(pred, _)| match pred.kind().skip_binder() {
                ty::ClauseKind::RegionOutlives(p) => p.to_string(),
                ty::ClauseKind::TypeOutlives(p) => p.to_string(),
                err => bug!("unexpected clause {:?}", err),
            })
            .collect();
        preds.sort();

        let span = tcx.def_span(id.owner_id);
        let mut err = tcx.dcx().struct_span_err(span, sym::rustc_outlives.as_str());
        for pred in preds {
            err.note(pred);
        }
        err.emit();
    }
}
