use rustc_errors::struct_span_err;
use rustc_hir as hir;
use rustc_hir::itemlikevisit::ParItemLikeVisitor;
use rustc_middle::ty::TyCtxt;
use rustc_span::def_id::LocalDefId;
use rustc_span::symbol::sym;

pub fn test_variance(tcx: TyCtxt<'_>, module: LocalDefId) {
    tcx.hir().par_visit_item_likes_in_module(module, &VarianceTest { tcx });
}

struct VarianceTest<'tcx> {
    tcx: TyCtxt<'tcx>,
}

impl<'tcx> ParItemLikeVisitor<'tcx> for VarianceTest<'tcx> {
    fn visit_item(&self, item: &'tcx hir::Item<'tcx>) {
        // For unit testing: check for a special "rustc_variance"
        // attribute and report an error with various results if found.
        if self.tcx.has_attr(item.def_id.to_def_id(), sym::rustc_variance) {
            let variances_of = self.tcx.variances_of(item.def_id);
            struct_span_err!(self.tcx.sess, item.span, E0208, "{:?}", variances_of).emit();
        }
    }

    fn visit_trait_item(&self, _: &'tcx hir::TraitItem<'tcx>) {}
    fn visit_impl_item(&self, _: &'tcx hir::ImplItem<'tcx>) {}
    fn visit_foreign_item(&self, _: &'tcx hir::ForeignItem<'tcx>) {}
}
