use rustc::hir;
use rustc::hir::itemlikevisit::ItemLikeVisitor;
use rustc::ty::TyCtxt;

pub fn test_variance<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>) {
    tcx.hir().krate().visit_all_item_likes(&mut VarianceTest { tcx });
}

struct VarianceTest<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>
}

impl<'a, 'tcx> ItemLikeVisitor<'tcx> for VarianceTest<'a, 'tcx> {
    fn visit_item(&mut self, item: &'tcx hir::Item) {
        let item_def_id = self.tcx.hir().local_def_id_from_hir_id(item.hir_id);

        // For unit testing: check for a special "rustc_variance"
        // attribute and report an error with various results if found.
        if self.tcx.has_attr(item_def_id, "rustc_variance") {
            let variances_of = self.tcx.variances_of(item_def_id);
            span_err!(self.tcx.sess,
                      item.span,
                      E0208,
                      "{:?}",
                      variances_of);
        }
    }

    fn visit_trait_item(&mut self, _: &'tcx hir::TraitItem) { }
    fn visit_impl_item(&mut self, _: &'tcx hir::ImplItem) { }
}
