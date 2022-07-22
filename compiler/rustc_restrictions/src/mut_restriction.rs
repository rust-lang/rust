use rustc_hir::def_id::LocalDefId;
use rustc_hir::Node;
use rustc_middle::query::Providers;
use rustc_middle::span_bug;
use rustc_middle::ty::{Restriction, TyCtxt};

pub(crate) fn provide(providers: &mut Providers) {
    *providers = Providers { mut_restriction, ..*providers };
}

fn mut_restriction(tcx: TyCtxt<'_>, def_id: LocalDefId) -> Restriction {
    tracing::debug!("mut_restriction({def_id:?})");

    match (&tcx.resolutions(()).mut_restrictions).get(&def_id) {
        Some(restriction) => *restriction,
        None => {
            let hir_id = tcx.local_def_id_to_hir_id(def_id);
            match tcx.hir().get(hir_id) {
                Node::Field(..) => {
                    tracing::debug!("mut restriction not found; assuming unrestricted");
                    Restriction::Unrestricted
                }
                _ => {
                    span_bug!(
                        tcx.def_span(def_id),
                        "called `mut_restriction` on invalid item: {def_id:?}",
                    )
                }
            }
        }
    }
}
