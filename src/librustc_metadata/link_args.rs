use rustc::ty::TyCtxt;
use rustc_hir as hir;
use rustc_hir::itemlikevisit::ItemLikeVisitor;
use rustc_span::symbol::sym;
use rustc_target::spec::abi::Abi;

crate fn collect(tcx: TyCtxt<'_>) -> Vec<String> {
    let mut collector = Collector { args: Vec::new() };
    tcx.hir().krate().visit_all_item_likes(&mut collector);

    for attr in tcx.hir().krate().item.attrs.iter() {
        if attr.has_name(sym::link_args) {
            if let Some(linkarg) = attr.value_str() {
                collector.add_link_args(&linkarg.as_str());
            }
        }
    }

    collector.args
}

struct Collector {
    args: Vec<String>,
}

impl<'tcx> ItemLikeVisitor<'tcx> for Collector {
    fn visit_item(&mut self, it: &'tcx hir::Item<'tcx>) {
        let fm = match it.kind {
            hir::ItemKind::ForeignMod(ref fm) => fm,
            _ => return,
        };
        if fm.abi == Abi::Rust || fm.abi == Abi::RustIntrinsic || fm.abi == Abi::PlatformIntrinsic {
            return;
        }

        // First, add all of the custom #[link_args] attributes
        for m in it.attrs.iter().filter(|a| a.check_name(sym::link_args)) {
            if let Some(linkarg) = m.value_str() {
                self.add_link_args(&linkarg.as_str());
            }
        }
    }

    fn visit_trait_item(&mut self, _it: &'tcx hir::TraitItem<'tcx>) {}
    fn visit_impl_item(&mut self, _it: &'tcx hir::ImplItem<'tcx>) {}
}

impl Collector {
    fn add_link_args(&mut self, args: &str) {
        self.args.extend(args.split(' ').filter(|s| !s.is_empty()).map(|s| s.to_string()))
    }
}
