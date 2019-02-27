use rustc::hir::itemlikevisit::ItemLikeVisitor;
use rustc::hir;
use rustc::middle::cstore::ForeignModule;
use rustc::ty::TyCtxt;

pub fn collect<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>) -> Vec<ForeignModule> {
    let mut collector = Collector {
        tcx,
        modules: Vec::new(),
    };
    tcx.hir().krate().visit_all_item_likes(&mut collector);
    return collector.modules
}

struct Collector<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    modules: Vec<ForeignModule>,
}

impl<'a, 'tcx> ItemLikeVisitor<'tcx> for Collector<'a, 'tcx> {
    fn visit_item(&mut self, it: &'tcx hir::Item) {
        let fm = match it.node {
            hir::ItemKind::ForeignMod(ref fm) => fm,
            _ => return,
        };

        let foreign_items = fm.items.iter()
            .map(|it| self.tcx.hir().local_def_id_from_hir_id(it.hir_id))
            .collect();
        self.modules.push(ForeignModule {
            foreign_items,
            def_id: self.tcx.hir().local_def_id_from_hir_id(it.hir_id),
        });
    }

    fn visit_trait_item(&mut self, _it: &'tcx hir::TraitItem) {}
    fn visit_impl_item(&mut self, _it: &'tcx hir::ImplItem) {}
}
