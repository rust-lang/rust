use crate::ty::TyCtxt;

/// A query to trigger a delayed bug. Clearly, if one has a `tcx` one can already trigger a
/// delayed bug, so what is the point of this? It exists to help us test the interaction of delayed
/// bugs with the query system and incremental.
pub fn trigger_delayed_bug(tcx: TyCtxt<'_>, key: rustc_hir::def_id::DefId) {
    tcx.dcx().span_delayed_bug(
        tcx.def_span(key),
        "delayed bug triggered by #[rustc_delayed_bug_from_inside_query]",
    );
}

pub fn provide(providers: &mut crate::query::Providers) {
    *providers = crate::query::Providers { trigger_delayed_bug, ..*providers };
}
