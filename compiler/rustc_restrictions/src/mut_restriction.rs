use rustc_hir::def_id::LocalDefId;
use rustc_hir::{Item, ItemKind, Node};
use rustc_middle::query::Providers;
use rustc_middle::span_bug;
use rustc_middle::ty::{Restriction, TyCtxt};

pub(crate) fn provide(providers: &mut Providers) {
    *providers = Providers { mut_restriction, ..*providers };
}

fn mut_restriction(tcx: TyCtxt<'_>, def_id: LocalDefId) -> Restriction {
    match tcx.resolutions(()).mut_restrictions.get(&def_id) {
        Some(restriction) => *restriction,
        None => {
            let hir_id = tcx.hir().local_def_id_to_hir_id(def_id);
            match tcx.hir().get(hir_id) {
                Node::Item(Item { kind: ItemKind::Struct(..), .. }) => {
                    span_bug!(
                        tcx.def_span(def_id),
                        "mut restriction table unexpectedly missing a def-id: {def_id:?}",
                    )
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
