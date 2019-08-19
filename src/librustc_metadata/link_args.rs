use rustc::hir::itemlikevisit::ItemLikeVisitor;
use rustc::hir;
use rustc::ty::TyCtxt;
use rustc_target::spec::abi::Abi;
use syntax::symbol::sym;

pub fn collect(tcx: TyCtxt<'_>) -> Vec<String> {
    let mut collector = Collector {
        args: Vec::new(),
    };
    tcx.hir().krate().visit_all_item_likes(&mut collector);

    for attr in tcx.hir().krate().attrs.iter() {
        if attr.path == sym::link_args {
            if let Some(linkarg) = attr.value_str() {
                collector.add_link_args(&linkarg.as_str());
            }
        }
    }

    return collector.args;
}

struct Collector {
    args: Vec<String>,
}

impl<'tcx> ItemLikeVisitor<'tcx> for Collector {
    fn visit_item(&mut self, it: &'tcx hir::Item) {
        let fm = match it.node {
            hir::ItemKind::ForeignMod(ref fm) => fm,
            _ => return,
        };
        if fm.abi == Abi::Rust ||
            fm.abi == Abi::RustIntrinsic ||
            fm.abi == Abi::PlatformIntrinsic {
            return
        }

        // First, add all of the custom #[link_args] attributes
        for m in it.attrs.iter().filter(|a| a.check_name(sym::link_args)) {
            if let Some(linkarg) = m.value_str() {
                self.add_link_args(&linkarg.as_str());
            }
        }
    }

    fn visit_trait_item(&mut self, _it: &'tcx hir::TraitItem) {}
    fn visit_impl_item(&mut self, _it: &'tcx hir::ImplItem) {}
}

impl Collector {
    fn add_link_args(&mut self, args: &str) {
        self.args.extend(args.split(' ').filter(|s| !s.is_empty()).map(|s| s.to_string()))
    }
}
