use rustc::hir::itemlikevisit::ItemLikeVisitor;
use rustc::hir::map::Map;
use rustc::hir;
use syntax::ast;
use syntax::attr;

pub fn find(hir_map: &Map) -> Option<ast::NodeId> {
    let krate = hir_map.krate();

    let mut finder = Finder { decls: None };
    krate.visit_all_item_likes(&mut finder);
    finder.decls
}

struct Finder {
    decls: Option<ast::NodeId>,
}

impl<'v> ItemLikeVisitor<'v> for Finder {
    fn visit_item(&mut self, item: &hir::Item) {
        if attr::contains_name(&item.attrs, "rustc_proc_macro_decls") {
            self.decls = Some(item.id);
        }
    }

    fn visit_trait_item(&mut self, _trait_item: &hir::TraitItem) {
    }

    fn visit_impl_item(&mut self, _impl_item: &hir::ImplItem) {
    }
}

