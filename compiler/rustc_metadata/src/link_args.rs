use rustc_hir as hir;
use rustc_hir::itemlikevisit::ItemLikeVisitor;
use rustc_middle::ty::TyCtxt;
use rustc_span::symbol::{sym, Symbol};
use rustc_target::spec::abi::Abi;

crate fn collect(tcx: TyCtxt<'_>) -> Vec<String> {
    let mut collector = Collector { tcx, args: Vec::new() };
    tcx.hir().krate().visit_all_item_likes(&mut collector);

    for attr in tcx.hir().krate().item.attrs.iter() {
        if attr.has_name(sym::link_args) {
            if let Some(linkarg) = attr.value_str() {
                collector.add_link_args(linkarg);
            }
        }
    }

    collector.args
}

struct Collector<'tcx> {
    tcx: TyCtxt<'tcx>,
    args: Vec<String>,
}

impl<'tcx> ItemLikeVisitor<'tcx> for Collector<'tcx> {
    fn visit_item(&mut self, it: &'tcx hir::Item<'tcx>) {
        let abi = match it.kind {
            hir::ItemKind::ForeignMod { abi, .. } => abi,
            _ => return,
        };
        if abi == Abi::Rust || abi == Abi::RustIntrinsic || abi == Abi::PlatformIntrinsic {
            return;
        }

        // First, add all of the custom #[link_args] attributes
        let sess = &self.tcx.sess;
        for m in it.attrs.iter().filter(|a| sess.check_name(a, sym::link_args)) {
            if let Some(linkarg) = m.value_str() {
                self.add_link_args(linkarg);
            }
        }
    }

    fn visit_trait_item(&mut self, _it: &'tcx hir::TraitItem<'tcx>) {}
    fn visit_impl_item(&mut self, _it: &'tcx hir::ImplItem<'tcx>) {}
    fn visit_foreign_item(&mut self, _it: &'tcx hir::ForeignItem<'tcx>) {}
}

impl<'tcx> Collector<'tcx> {
    fn add_link_args(&mut self, args: Symbol) {
        self.args.extend(args.as_str().split(' ').filter(|s| !s.is_empty()).map(|s| s.to_string()))
    }
}
