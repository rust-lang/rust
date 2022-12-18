use rustc_errors::struct_span_err;
use rustc_middle::ty::TyCtxt;
use rustc_span::symbol::sym;

pub fn test_variance(tcx: TyCtxt<'_>) {
    // For unit testing: check for a special "rustc_variance"
    // attribute and report an error with various results if found.
    for id in tcx.hir().items() {
        if tcx.has_attr(id.owner_id.to_def_id(), sym::rustc_variance) {
            let variances_of = tcx.variances_of(id.owner_id);
            struct_span_err!(tcx.sess, tcx.def_span(id.owner_id), E0208, "{:?}", variances_of)
                .emit();
        }
    }
}
