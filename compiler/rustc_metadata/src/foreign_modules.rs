use rustc_hir as hir;
use rustc_hir::itemlikevisit::ItemLikeVisitor;
use rustc_middle::ty::TyCtxt;
use rustc_session::cstore::ForeignModule;

crate fn collect(tcx: TyCtxt<'_>) -> Vec<ForeignModule> {
    let mut collector = Collector { modules: Vec::new() };
    tcx.hir().visit_all_item_likes(&mut collector);
    collector.modules
}

struct Collector {
    modules: Vec<ForeignModule>,
}

impl ItemLikeVisitor<'tcx> for Collector {
    fn visit_item(&mut self, it: &'tcx hir::Item<'tcx>) {
        let items = match it.kind {
            hir::ItemKind::ForeignMod { items, .. } => items,
            _ => return,
        };

        let foreign_items = items.iter().map(|it| it.id.def_id.to_def_id()).collect();
        self.modules.push(ForeignModule { foreign_items, def_id: it.def_id.to_def_id() });
    }

    fn visit_trait_item(&mut self, _it: &'tcx hir::TraitItem<'tcx>) {}
    fn visit_impl_item(&mut self, _it: &'tcx hir::ImplItem<'tcx>) {}
    fn visit_foreign_item(&mut self, _it: &'tcx hir::ForeignItem<'tcx>) {}
}
