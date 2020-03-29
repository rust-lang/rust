use rustc_hir as hir;
use rustc_hir::itemlikevisit::ItemLikeVisitor;
use rustc_middle::middle::cstore::ForeignModule;
use rustc_middle::ty::TyCtxt;

crate fn collect(tcx: TyCtxt<'_>) -> Vec<ForeignModule> {
    let mut collector = Collector { tcx, modules: Vec::new() };
    tcx.hir().krate().visit_all_item_likes(&mut collector);
    collector.modules
}

struct Collector<'tcx> {
    tcx: TyCtxt<'tcx>,
    modules: Vec<ForeignModule>,
}

impl ItemLikeVisitor<'tcx> for Collector<'tcx> {
    fn visit_item(&mut self, it: &'tcx hir::Item<'tcx>) {
        let fm = match it.kind {
            hir::ItemKind::ForeignMod(ref fm) => fm,
            _ => return,
        };

        let foreign_items =
            fm.items.iter().map(|it| self.tcx.hir().local_def_id(it.hir_id)).collect();
        self.modules
            .push(ForeignModule { foreign_items, def_id: self.tcx.hir().local_def_id(it.hir_id) });
    }

    fn visit_trait_item(&mut self, _it: &'tcx hir::TraitItem<'tcx>) {}
    fn visit_impl_item(&mut self, _it: &'tcx hir::ImplItem<'tcx>) {}
}
