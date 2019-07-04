use rustc::hir;
use rustc::hir::itemlikevisit::ItemLikeVisitor;
use rustc::ty::TyCtxt;
use syntax::symbol::sym;

pub fn test_inferred_outlives(tcx: TyCtxt<'_>) {
    tcx.hir()
       .krate()
       .visit_all_item_likes(&mut OutlivesTest { tcx });
}

struct OutlivesTest<'tcx> {
    tcx: TyCtxt<'tcx>,
}

impl ItemLikeVisitor<'tcx> for OutlivesTest<'tcx> {
    fn visit_item(&mut self, item: &'tcx hir::Item) {
        let item_def_id = self.tcx.hir().local_def_id_from_hir_id(item.hir_id);

        // For unit testing: check for a special "rustc_outlives"
        // attribute and report an error with various results if found.
        if self.tcx.has_attr(item_def_id, sym::rustc_outlives) {
            let inferred_outlives_of = self.tcx.inferred_outlives_of(item_def_id);
            span_err!(
                self.tcx.sess,
                item.span,
                E0640,
                "{:?}",
                inferred_outlives_of
            );
        }
    }

    fn visit_trait_item(&mut self, _: &'tcx hir::TraitItem) {}
    fn visit_impl_item(&mut self, _: &'tcx hir::ImplItem) {}
}
