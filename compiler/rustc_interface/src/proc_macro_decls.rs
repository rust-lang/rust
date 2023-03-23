use rustc_ast::attr;
use rustc_hir::def_id::LocalDefId;
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::TyCtxt;
use rustc_span::symbol::sym;

fn proc_macro_decls_static(tcx: TyCtxt<'_>, (): ()) -> Option<LocalDefId> {
    let mut decls = None;

    for id in tcx.hir().items() {
        let attrs = tcx.hir().attrs(id.hir_id());
        if attr::contains_name(attrs, sym::rustc_proc_macro_decls) {
            decls = Some(id.owner_id.def_id);
        }
    }

    decls
}

pub(crate) fn provide(providers: &mut Providers) {
    *providers = Providers { proc_macro_decls_static, ..*providers };
}
