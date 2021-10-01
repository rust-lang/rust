use rustc_errors::struct_span_err;
use rustc_hir as hir;
use rustc_hir::itemlikevisit::ItemLikeVisitor;
use rustc_middle::ty::TyCtxt;
use rustc_span::symbol::sym;

pub fn test_inferred_outlives(tcx: TyCtxt<'_>) {
    tcx.hir().visit_all_item_likes(&mut OutlivesTest { tcx });
}

struct OutlivesTest<'tcx> {
    tcx: TyCtxt<'tcx>,
}

impl ItemLikeVisitor<'tcx> for OutlivesTest<'tcx> {
    fn visit_item(&mut self, item: &'tcx hir::Item<'tcx>) {
        // For unit testing: check for a special "rustc_outlives"
        // attribute and report an error with various results if found.
        if self.tcx.has_attr(item.def_id.to_def_id(), sym::rustc_outlives) {
            let inferred_outlives_of = self.tcx.inferred_outlives_of(item.def_id);
            struct_span_err!(self.tcx.sess, item.span, E0640, "{:?}", inferred_outlives_of).emit();
        }
    }

    fn visit_trait_item(&mut self, _: &'tcx hir::TraitItem<'tcx>) {}
    fn visit_impl_item(&mut self, _: &'tcx hir::ImplItem<'tcx>) {}
    fn visit_foreign_item(&mut self, _: &'tcx hir::ForeignItem<'tcx>) {}
}
