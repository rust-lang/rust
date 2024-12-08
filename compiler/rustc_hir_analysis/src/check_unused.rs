use rustc_data_structures::unord::{ExtendUnord, UnordSet};
use rustc_hir::def::DefKind;
use rustc_hir::def_id::LocalDefId;
use rustc_middle::query::Providers;
use rustc_middle::ty::TyCtxt;
use rustc_session::lint;
use tracing::debug;

pub(crate) fn provide(providers: &mut Providers) {
    *providers = Providers { check_unused_traits, ..*providers };
}

fn check_unused_traits(tcx: TyCtxt<'_>, (): ()) {
    let mut used_trait_imports = UnordSet::<LocalDefId>::default();

    // FIXME: Use `tcx.hir().par_body_owners()` when we implement creating `DefId`s
    // for anon constants during their parents' typeck.
    // Doing so at current will produce queries cycle errors because it may typeck
    // on anon constants directly.
    for item_def_id in tcx.hir().body_owners() {
        let imports = tcx.used_trait_imports(item_def_id);
        debug!("GatherVisitor: item_def_id={:?} with imports {:#?}", item_def_id, imports);
        used_trait_imports.extend_unord(imports.items().copied());
    }

    for &id in tcx.maybe_unused_trait_imports(()) {
        debug_assert_eq!(tcx.def_kind(id), DefKind::Use);
        if tcx.visibility(id).is_public() {
            continue;
        }
        if used_trait_imports.contains(&id) {
            continue;
        }
        let item = tcx.hir().expect_item(id);
        if item.span.is_dummy() {
            continue;
        }
        let (path, _) = item.expect_use();
        tcx.node_span_lint(lint::builtin::UNUSED_IMPORTS, item.hir_id(), path.span, |lint| {
            if let Ok(snippet) = tcx.sess.source_map().span_to_snippet(path.span) {
                lint.primary_message(format!("unused import: `{snippet}`"));
            } else {
                lint.primary_message("unused import");
            }
        });
    }
}
