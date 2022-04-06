use rustc_hir as hir;
use rustc_hir::itemlikevisit::ItemLikeVisitor;
use rustc_middle::ty::TyCtxt;
use rustc_session::cstore::ForeignModule;

crate fn collect(tcx: TyCtxt<'_>) -> Vec<ForeignModule> {
    let mut modules = Vec::new();
    for id in tcx.hir().items() {
        let item = tcx.hir().item(id);
        let hir::ItemKind::ForeignMod { items, .. } = item.kind else {
            continue;
        };
        let foreign_items = items.iter().map(|it| it.id.def_id.to_def_id()).collect();
        modules.push(ForeignModule { foreign_items, def_id: id.def_id.to_def_id() });
    }
    modules
}

struct Collector {
    modules: Vec<ForeignModule>,
}

impl<'tcx> ItemLikeVisitor<'tcx> for Collector {
    fn visit_item(&mut self, it: &'tcx hir::Item<'tcx>) {
        let hir::ItemKind::ForeignMod { items, .. } = it.kind else {
            return;
        };

        let foreign_items = items.iter().map(|it| it.id.def_id.to_def_id()).collect();
        self.modules.push(ForeignModule { foreign_items, def_id: it.def_id.to_def_id() });
    }

    fn visit_trait_item(&mut self, _it: &'tcx hir::TraitItem<'tcx>) {}
    fn visit_impl_item(&mut self, _it: &'tcx hir::ImplItem<'tcx>) {}
    fn visit_foreign_item(&mut self, _it: &'tcx hir::ForeignItem<'tcx>) {}
}
