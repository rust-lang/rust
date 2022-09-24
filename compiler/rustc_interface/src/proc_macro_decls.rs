use rustc_hir::def_id::LocalDefId;
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::TyCtxt;
use rustc_span::symbol::sym;

fn proc_macro_decls_static(tcx: TyCtxt<'_>, (): ()) -> Option<LocalDefId> {
    let mut finder = Finder { tcx, decls: None };

    for id in tcx.hir().items() {
        let attrs = finder.tcx.hir().attrs(id.hir_id());
        if finder.tcx.sess.contains_name(attrs, sym::rustc_proc_macro_decls) {
            finder.decls = Some(id.def_id.def_id);
        }
    }

    finder.decls
}

struct Finder<'tcx> {
    tcx: TyCtxt<'tcx>,
    decls: Option<LocalDefId>,
}

pub(crate) fn provide(providers: &mut Providers) {
    *providers = Providers { proc_macro_decls_static, ..*providers };
}
