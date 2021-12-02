use ast::visit;
use rustc_ast as ast;
use rustc_hir::def::Namespace::TypeNS;
use rustc_hir::def_id::{LocalDefId, CRATE_DEF_ID};
use rustc_interface::interface;
use rustc_span::Span;

use std::cell::RefCell;
use std::mem;
use std::rc::Rc;

type Resolver = Rc<RefCell<interface::BoxedResolver>>;
// Letting the resolver escape at the end of the function leads to inconsistencies between the
// crates the TyCtxt sees and the resolver sees (because the resolver could load more crates
// after escaping). Hopefully `IntraLinkCrateLoader` gets all the crates we need ...
crate fn load_intra_link_crates(resolver: Resolver, krate: &ast::Crate) -> Resolver {
    let mut loader = IntraLinkCrateLoader { current_mod: CRATE_DEF_ID, resolver };
    // `walk_crate` doesn't visit the crate itself for some reason.
    loader.load_links_in_attrs(&krate.attrs, krate.span);
    visit::walk_crate(&mut loader, krate);
    loader.resolver
}

struct IntraLinkCrateLoader {
    current_mod: LocalDefId,
    resolver: Rc<RefCell<interface::BoxedResolver>>,
}

impl IntraLinkCrateLoader {
    fn load_links_in_attrs(&mut self, attrs: &[ast::Attribute], span: Span) {
        use crate::html::markdown::markdown_links;
        use crate::passes::collect_intra_doc_links::preprocess_link;

        // FIXME: this probably needs to consider inlining
        let attrs = crate::clean::Attributes::from_ast(attrs, None);
        for (parent_module, doc) in attrs.collapsed_doc_value_by_module_level() {
            debug!(?doc);
            for link in markdown_links(doc.as_str()) {
                debug!(?link.link);
                let path_str = if let Some(Ok(x)) = preprocess_link(&link) {
                    x.path_str
                } else {
                    continue;
                };
                self.resolver.borrow_mut().access(|resolver| {
                    let _ = resolver.resolve_str_path_error(
                        span,
                        &path_str,
                        TypeNS,
                        parent_module.unwrap_or_else(|| self.current_mod.to_def_id()),
                    );
                });
            }
        }
    }
}

impl visit::Visitor<'_> for IntraLinkCrateLoader {
    fn visit_foreign_item(&mut self, item: &ast::ForeignItem) {
        self.load_links_in_attrs(&item.attrs, item.span);
        visit::walk_foreign_item(self, item)
    }

    fn visit_item(&mut self, item: &ast::Item) {
        use rustc_ast_lowering::ResolverAstLowering;

        if let ast::ItemKind::Mod(..) = item.kind {
            let new_mod =
                self.resolver.borrow_mut().access(|resolver| resolver.local_def_id(item.id));
            let old_mod = mem::replace(&mut self.current_mod, new_mod);

            self.load_links_in_attrs(&item.attrs, item.span);
            visit::walk_item(self, item);

            self.current_mod = old_mod;
        } else {
            self.load_links_in_attrs(&item.attrs, item.span);
            visit::walk_item(self, item);
        }
    }

    // NOTE: if doc-comments are ever allowed on function parameters, this will have to implement `visit_param` too.

    fn visit_assoc_item(&mut self, item: &ast::AssocItem, ctxt: visit::AssocCtxt) {
        self.load_links_in_attrs(&item.attrs, item.span);
        visit::walk_assoc_item(self, item, ctxt)
    }

    fn visit_field_def(&mut self, field: &ast::FieldDef) {
        self.load_links_in_attrs(&field.attrs, field.span);
        visit::walk_field_def(self, field)
    }

    fn visit_variant(&mut self, v: &ast::Variant) {
        self.load_links_in_attrs(&v.attrs, v.span);
        visit::walk_variant(self, v)
    }
}
