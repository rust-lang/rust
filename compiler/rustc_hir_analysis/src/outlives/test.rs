use rustc_middle::ty::{self, TyCtxt};
use rustc_span::symbol::sym;

pub fn test_inferred_outlives(tcx: TyCtxt<'_>) {
    for id in tcx.hir().items() {
        // For unit testing: check for a special "rustc_outlives"
        // attribute and report an error with various results if found.
        if tcx.has_attr(id.owner_id, sym::rustc_outlives) {
            let predicates = tcx.inferred_outlives_of(id.owner_id);
            let mut pred: Vec<String> = predicates
                .iter()
                .map(|(out_pred, _)| match out_pred.kind().skip_binder() {
                    ty::ClauseKind::RegionOutlives(p) => p.to_string(),
                    ty::ClauseKind::TypeOutlives(p) => p.to_string(),
                    err => bug!("unexpected clause {:?}", err),
                })
                .collect();
            pred.sort();

            let span = tcx.def_span(id.owner_id);
            let mut err = tcx.dcx().struct_span_err(span, "rustc_outlives");
            for p in pred {
                err.note(p);
            }
            err.emit();
        }
    }
}
