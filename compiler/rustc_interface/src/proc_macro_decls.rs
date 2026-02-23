use rustc_hir::def_id::LocalDefId;
use rustc_hir::find_attr;
use rustc_middle::query::Providers;
use rustc_middle::ty::TyCtxt;

fn proc_macro_decls_static(tcx: TyCtxt<'_>, (): ()) -> Option<LocalDefId> {
    let mut decls = None;

    for id in tcx.hir_free_items() {
        if find_attr!(tcx.hir_attrs(id.hir_id()), RustcProcMacroDecls) {
            decls = Some(id.owner_id.def_id);
        }
    }

    decls
}

pub(crate) fn provide(providers: &mut Providers) {
    *providers = Providers { proc_macro_decls_static, ..*providers };
}
