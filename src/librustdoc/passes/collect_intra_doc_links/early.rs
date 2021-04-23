use rustc_ast as ast;
use rustc_hir::def::Namespace::TypeNS;
use rustc_hir::def_id::{DefId, LocalDefId, CRATE_DEF_INDEX};
use rustc_interface::interface;

use std::cell::RefCell;
use std::mem;
use std::rc::Rc;

// Letting the resolver escape at the end of the function leads to inconsistencies between the
// crates the TyCtxt sees and the resolver sees (because the resolver could load more crates
// after escaping). Hopefully `IntraLinkCrateLoader` gets all the crates we need ...
crate struct IntraLinkCrateLoader {
    current_mod: DefId,
    crate resolver: Rc<RefCell<interface::BoxedResolver>>,
}

impl IntraLinkCrateLoader {
    crate fn new(resolver: Rc<RefCell<interface::BoxedResolver>>) -> Self {
        let crate_id = LocalDefId { local_def_index: CRATE_DEF_INDEX }.to_def_id();
        Self { current_mod: crate_id, resolver }
    }
}

impl ast::visit::Visitor<'_> for IntraLinkCrateLoader {
    fn visit_attribute(&mut self, attr: &ast::Attribute) {
        use crate::html::markdown::markdown_links;
        use crate::passes::collect_intra_doc_links::preprocess_link;

        if let Some(doc) = attr.doc_str() {
            for link in markdown_links(&doc.as_str()) {
                let path_str = if let Some(Ok(x)) = preprocess_link(&link) {
                    x.path_str
                } else {
                    continue;
                };
                self.resolver.borrow_mut().access(|resolver| {
                    let _ = resolver.resolve_str_path_error(
                        attr.span,
                        &path_str,
                        TypeNS,
                        self.current_mod,
                    );
                });
            }
        }
        ast::visit::walk_attribute(self, attr);
    }

    fn visit_item(&mut self, item: &ast::Item) {
        use rustc_ast_lowering::ResolverAstLowering;

        if let ast::ItemKind::Mod(..) = item.kind {
            let new_mod =
                self.resolver.borrow_mut().access(|resolver| resolver.local_def_id(item.id));
            let old_mod = mem::replace(&mut self.current_mod, new_mod.to_def_id());
            ast::visit::walk_item(self, item);
            self.current_mod = old_mod;
        } else {
            ast::visit::walk_item(self, item);
        }
    }
}
