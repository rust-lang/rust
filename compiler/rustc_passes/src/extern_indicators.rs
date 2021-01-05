//! Detecting extern indicators.
//!
//! Some examples include `#[no_mangle]` and `#[link_name(...)]`.

use rustc_middle::ty::TyCtxt;

use rustc_hir as hir;
use rustc_hir::def_id::LOCAL_CRATE;
use rustc_hir::itemlikevisit::ItemLikeVisitor;
use rustc_hir::HirId;

use rustc_middle::ty::query::Providers;

struct ExternIndicatorCollector<'tcx> {
    has_extern_indicators: bool,
    tcx: TyCtxt<'tcx>,
}

impl ItemLikeVisitor<'v> for ExternIndicatorCollector<'tcx> {
    fn visit_item(&mut self, item: &hir::Item<'_>) {
        self.check(item.hir_id);
    }

    fn visit_trait_item(&mut self, item: &hir::TraitItem<'_>) {
        self.check(item.hir_id);
    }

    fn visit_impl_item(&mut self, item: &hir::ImplItem<'_>) {
        self.check(item.hir_id);
    }

    fn visit_foreign_item(&mut self, item: &hir::ForeignItem<'_>) {
        self.check(item.hir_id);
    }
}

impl<'tcx> ExternIndicatorCollector<'tcx> {
    fn new(tcx: TyCtxt<'tcx>) -> Self {
        ExternIndicatorCollector { tcx, has_extern_indicators: false }
    }

    fn check(&mut self, hir_id: HirId) {
        let def_id = self.tcx.hir().local_def_id(hir_id);
        if self.tcx.codegen_fn_attrs(def_id).contains_extern_indicator() {
            self.has_extern_indicators = true;
        }
    }
}

pub fn provide(providers: &mut Providers) {
    providers.has_extern_indicators = |tcx, id| {
        assert_eq!(id, LOCAL_CRATE);

        let mut collector = ExternIndicatorCollector::new(tcx);
        tcx.hir().krate().visit_all_item_likes(&mut collector);

        collector.has_extern_indicators
    };
}
