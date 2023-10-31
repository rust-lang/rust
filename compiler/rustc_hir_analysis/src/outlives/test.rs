use rustc_errors::struct_span_err;
use rustc_middle::ty::TyCtxt;
use rustc_span::symbol::sym;

pub fn test_inferred_outlives(tcx: TyCtxt<'_>) {
    for id in tcx.hir().items() {
        // For unit testing: check for a special "rustc_outlives"
        // attribute and report an error with various results if found.
        if tcx.has_attr(id.owner_id, sym::rustc_outlives) {
            let inferred_outlives_of = tcx.inferred_outlives_of(id.owner_id);
            let mut err =
                struct_span_err!(tcx.sess, tcx.def_span(id.owner_id), E0640, "rustc_outlives");
            for &(clause, span) in inferred_outlives_of {
                err.span_note(span, format!("{clause}"));
            }
            err.emit();
        }
    }
}
