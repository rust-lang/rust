use rustc_middle::ty::TyCtxt;
use rustc_span::def_id::LocalDefId;
use rustc_span::symbol::sym;

pub fn test_variance(tcx: TyCtxt<'_>, module: LocalDefId) {
    let _timer = tcx.sess.timer("variance_testing");

    // For unit testing: check for a special "rustc_variance"
    // attribute and report an error with various results if found.
    for id in tcx.hir().module_items(module) {
        if tcx.has_attr(id.owner_id.to_def_id(), sym::rustc_variance) {
            let variances_of = tcx.variances_of(id.owner_id);
            tcx.sess.struct_span_err(tcx.def_span(id.owner_id), format!("{variances_of:?}")).emit();
        }
    }
}
