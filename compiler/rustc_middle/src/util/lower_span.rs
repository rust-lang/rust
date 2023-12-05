use rustc_hir::def_id::LocalDefId;
use rustc_span::Span;

use crate::ty::TyCtxt;

pub fn lower_span(tcx: TyCtxt<'_>, span: Span, parent: LocalDefId) -> Span {
    if tcx.sess.opts.incremental.is_some() {
        span.with_parent(Some(parent))
    } else {
        // Do not make spans relative when not using incremental compilation.
        span
    }
}
