use rustc_hir as hir;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::itemlikevisit::ItemLikeVisitor;
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::TyCtxt;
use rustc_span::symbol::sym;

fn proc_macro_decls_static(tcx: TyCtxt<'_>, (): ()) -> Option<LocalDefId> {
    let mut finder = Finder { tcx, decls: None };
    tcx.hir().visit_all_item_likes(&mut finder);

    finder.decls.map(|id| tcx.hir().local_def_id(id))
}

struct Finder<'tcx> {
    tcx: TyCtxt<'tcx>,
    decls: Option<hir::HirId>,
}

impl<'v> ItemLikeVisitor<'v> for Finder<'_> {
    fn visit_item(&mut self, item: &hir::Item<'_>) {
        let attrs = self.tcx.hir().attrs(item.hir_id());
        if self.tcx.sess.contains_name(attrs, sym::rustc_proc_macro_decls) {
            self.decls = Some(item.hir_id());
        }
    }

    fn visit_trait_item(&mut self, _trait_item: &hir::TraitItem<'_>) {}

    fn visit_impl_item(&mut self, _impl_item: &hir::ImplItem<'_>) {}

    fn visit_foreign_item(&mut self, _foreign_item: &hir::ForeignItem<'_>) {}
}

pub(crate) fn provide(providers: &mut Providers) {
    *providers = Providers { proc_macro_decls_static, ..*providers };
}
