use rustc_hir as hir;
use rustc_hir::def_id::LocalDefId;
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::TyCtxt;
use rustc_span::symbol::sym;

fn proc_macro_decls_static(tcx: TyCtxt<'_>, (): ()) -> Option<LocalDefId> {
    let mut finder = Finder { tcx, decls: None };

    for id in tcx.hir().items() {
        let attrs = finder.tcx.hir().attrs(id.hir_id());
        if finder.tcx.sess.contains_name(attrs, sym::rustc_proc_macro_decls) {
            finder.decls = Some(id.hir_id());
        }
    }

    finder.decls.map(|id| tcx.hir().local_def_id(id))
}

struct Finder<'tcx> {
    tcx: TyCtxt<'tcx>,
    decls: Option<hir::HirId>,
}

pub(crate) fn provide(providers: &mut Providers) {
    *providers = Providers { proc_macro_decls_static, ..*providers };
}
