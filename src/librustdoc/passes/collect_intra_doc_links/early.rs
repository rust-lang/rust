use crate::core::ResolverCaches;
use crate::visit_lib::early_lib_embargo_visit_item;

use rustc_ast::visit::{self, Visitor};
use rustc_ast::{self as ast, ItemKind};
use rustc_data_structures::fx::FxHashMap;
use rustc_hir::def::Res;
use rustc_hir::def_id::{DefId, DefIdSet};
use rustc_resolve::Resolver;
use rustc_span::Symbol;

pub(crate) fn early_resolve_intra_doc_links(
    resolver: &mut Resolver<'_>,
    krate: &ast::Crate,
) -> ResolverCaches {
    let mut link_resolver = EarlyDocLinkResolver {
        resolver,
        all_trait_impls: Default::default(),
        all_macro_rules: Default::default(),
        extern_doc_reachable: Default::default(),
    };

    visit::walk_crate(&mut link_resolver, krate);
    link_resolver.process_extern_impls();

    ResolverCaches {
        all_trait_impls: Some(link_resolver.all_trait_impls),
        all_macro_rules: link_resolver.all_macro_rules,
        extern_doc_reachable: link_resolver.extern_doc_reachable,
    }
}

struct EarlyDocLinkResolver<'r, 'ra> {
    resolver: &'r mut Resolver<'ra>,
    all_trait_impls: Vec<DefId>,
    all_macro_rules: FxHashMap<Symbol, Res<ast::NodeId>>,
    /// This set is used as a seed for `effective_visibilities`, which are then extended by some
    /// more items using `lib_embargo_visit_item` during doc inlining.
    extern_doc_reachable: DefIdSet,
}

impl<'ra> EarlyDocLinkResolver<'_, 'ra> {
    fn process_extern_impls(&mut self) {
        for cnum in self.resolver.cstore().crates_untracked() {
            early_lib_embargo_visit_item(
                self.resolver,
                &mut self.extern_doc_reachable,
                cnum.as_def_id(),
                true,
            );
            for (_, impl_def_id, _) in self.resolver.cstore().trait_impls_in_crate_untracked(cnum) {
                self.all_trait_impls.push(impl_def_id);
            }
        }
    }
}

impl Visitor<'_> for EarlyDocLinkResolver<'_, '_> {
    fn visit_item(&mut self, item: &ast::Item) {
        match &item.kind {
            ItemKind::Impl(impl_) if impl_.of_trait.is_some() => {
                self.all_trait_impls.push(self.resolver.local_def_id(item.id).to_def_id());
            }
            ItemKind::MacroDef(macro_def) if macro_def.macro_rules => {
                let (_, res) = self.resolver.macro_rules_scope(self.resolver.local_def_id(item.id));
                self.all_macro_rules.insert(item.ident.name, res);
            }
            _ => {}
        }
        visit::walk_item(self, item);
    }
}
