use crate::clean;
use crate::core::ResolverCaches;
use crate::html::markdown::markdown_links;
use crate::passes::collect_intra_doc_links::preprocess_link;

use rustc_ast::visit::{self, AssocCtxt, Visitor};
use rustc_ast::{self as ast, ItemKind};
use rustc_ast_lowering::ResolverAstLowering;
use rustc_hir::def::Namespace::TypeNS;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::{DefId, DefIdMap, DefIdSet, LocalDefId, CRATE_DEF_ID};
use rustc_hir::TraitCandidate;
use rustc_middle::ty::{DefIdTree, Visibility};
use rustc_resolve::{ParentScope, Resolver};
use rustc_session::config::Externs;
use rustc_span::{Span, SyntaxContext, DUMMY_SP};

use std::collections::hash_map::Entry;
use std::mem;

crate fn early_resolve_intra_doc_links(
    resolver: &mut Resolver<'_>,
    krate: &ast::Crate,
    externs: Externs,
) -> ResolverCaches {
    let mut loader = IntraLinkCrateLoader {
        resolver,
        current_mod: CRATE_DEF_ID,
        visited_mods: Default::default(),
        traits_in_scope: Default::default(),
        all_traits: Default::default(),
        all_trait_impls: Default::default(),
    };

    // Because of the `crate::` prefix, any doc comment can reference
    // the crate root's set of in-scope traits. This line makes sure
    // it's available.
    loader.add_traits_in_scope(CRATE_DEF_ID.to_def_id());

    // Overridden `visit_item` below doesn't apply to the crate root,
    // so we have to visit its attributes and reexports separately.
    loader.load_links_in_attrs(&krate.attrs, krate.span);
    loader.process_module_children_or_reexports(CRATE_DEF_ID.to_def_id());
    visit::walk_crate(&mut loader, krate);
    loader.add_foreign_traits_in_scope();

    // FIXME: somehow rustdoc is still missing crates even though we loaded all
    // the known necessary crates. Load them all unconditionally until we find a way to fix this.
    // DO NOT REMOVE THIS without first testing on the reproducer in
    // https://github.com/jyn514/objr/commit/edcee7b8124abf0e4c63873e8422ff81beb11ebb
    for (extern_name, _) in externs.iter().filter(|(_, entry)| entry.add_prelude) {
        let _ = loader.resolver.resolve_str_path_error(
            DUMMY_SP,
            extern_name,
            TypeNS,
            CRATE_DEF_ID.to_def_id(),
        );
    }

    ResolverCaches {
        traits_in_scope: loader.traits_in_scope,
        all_traits: Some(loader.all_traits),
        all_trait_impls: Some(loader.all_trait_impls),
    }
}

struct IntraLinkCrateLoader<'r, 'ra> {
    resolver: &'r mut Resolver<'ra>,
    current_mod: LocalDefId,
    visited_mods: DefIdSet,
    traits_in_scope: DefIdMap<Vec<TraitCandidate>>,
    all_traits: Vec<DefId>,
    all_trait_impls: Vec<DefId>,
}

impl IntraLinkCrateLoader<'_, '_> {
    fn add_traits_in_scope(&mut self, def_id: DefId) {
        // Calls to `traits_in_scope` are expensive, so try to avoid them if only possible.
        // Keys in the `traits_in_scope` cache are always module IDs.
        if let Entry::Vacant(entry) = self.traits_in_scope.entry(def_id) {
            let module = self.resolver.get_nearest_non_block_module(def_id);
            let module_id = module.def_id();
            let entry = if module_id == def_id {
                Some(entry)
            } else if let Entry::Vacant(entry) = self.traits_in_scope.entry(module_id) {
                Some(entry)
            } else {
                None
            };
            if let Some(entry) = entry {
                entry.insert(self.resolver.traits_in_scope(
                    None,
                    &ParentScope::module(module, self.resolver),
                    SyntaxContext::root(),
                    None,
                ));
            }
        }
    }

    fn add_traits_in_parent_scope(&mut self, def_id: DefId) {
        if let Some(module_id) = self.resolver.parent(def_id) {
            self.add_traits_in_scope(module_id);
        }
    }

    /// Add traits in scope for links in impls collected by the `collect-intra-doc-links` pass.
    /// That pass filters impls using type-based information, but we don't yet have such
    /// information here, so we just conservatively calculate traits in scope for *all* modules
    /// having impls in them.
    fn add_foreign_traits_in_scope(&mut self) {
        for cnum in Vec::from_iter(self.resolver.cstore().crates_untracked()) {
            // FIXME: Due to #78696 rustdoc can query traits in scope for any crate root.
            self.add_traits_in_scope(cnum.as_def_id());

            let all_traits = Vec::from_iter(self.resolver.cstore().traits_in_crate_untracked(cnum));
            let all_trait_impls =
                Vec::from_iter(self.resolver.cstore().trait_impls_in_crate_untracked(cnum));
            let all_inherent_impls =
                Vec::from_iter(self.resolver.cstore().inherent_impls_in_crate_untracked(cnum));
            let all_lang_items = Vec::from_iter(self.resolver.cstore().lang_items_untracked(cnum));

            // Querying traits in scope is expensive so we try to prune the impl and traits lists
            // using privacy, private traits and impls from other crates are never documented in
            // the current crate, and links in their doc comments are not resolved.
            for &def_id in &all_traits {
                if self.resolver.cstore().visibility_untracked(def_id) == Visibility::Public {
                    self.add_traits_in_parent_scope(def_id);
                }
            }
            for &(trait_def_id, impl_def_id, simplified_self_ty) in &all_trait_impls {
                if self.resolver.cstore().visibility_untracked(trait_def_id) == Visibility::Public
                    && simplified_self_ty.and_then(|ty| ty.def()).map_or(true, |ty_def_id| {
                        self.resolver.cstore().visibility_untracked(ty_def_id) == Visibility::Public
                    })
                {
                    self.add_traits_in_parent_scope(impl_def_id);
                }
            }
            for (ty_def_id, impl_def_id) in all_inherent_impls {
                if self.resolver.cstore().visibility_untracked(ty_def_id) == Visibility::Public {
                    self.add_traits_in_parent_scope(impl_def_id);
                }
            }
            for def_id in all_lang_items {
                self.add_traits_in_parent_scope(def_id);
            }

            self.all_traits.extend(all_traits);
            self.all_trait_impls.extend(all_trait_impls.into_iter().map(|(_, def_id, _)| def_id));
        }
    }

    fn load_links_in_attrs(&mut self, attrs: &[ast::Attribute], span: Span) {
        // FIXME: this needs to consider reexport inlining.
        let attrs = clean::Attributes::from_ast(attrs, None);
        for (parent_module, doc) in attrs.collapsed_doc_value_by_module_level() {
            let module_id = parent_module.unwrap_or(self.current_mod.to_def_id());

            self.add_traits_in_scope(module_id);

            for link in markdown_links(&doc.as_str()) {
                let path_str = if let Some(Ok(x)) = preprocess_link(&link) {
                    x.path_str
                } else {
                    continue;
                };
                let _ = self.resolver.resolve_str_path_error(span, &path_str, TypeNS, module_id);
            }
        }
    }

    /// When reexports are inlined, they are replaced with item which they refer to, those items
    /// may have links in their doc comments, those links are resolved at the item definition site,
    /// so we need to know traits in scope at that definition site.
    fn process_module_children_or_reexports(&mut self, module_id: DefId) {
        if !self.visited_mods.insert(module_id) {
            return; // avoid infinite recursion
        }

        for child in self.resolver.module_children_or_reexports(module_id) {
            if child.vis == Visibility::Public {
                if let Some(def_id) = child.res.opt_def_id() {
                    self.add_traits_in_parent_scope(def_id);
                }
                if let Res::Def(DefKind::Mod, module_id) = child.res {
                    self.process_module_children_or_reexports(module_id);
                }
            }
        }
    }
}

impl Visitor<'_> for IntraLinkCrateLoader<'_, '_> {
    fn visit_item(&mut self, item: &ast::Item) {
        if let ItemKind::Mod(..) = item.kind {
            let old_mod = mem::replace(&mut self.current_mod, self.resolver.local_def_id(item.id));

            // A module written with a outline doc comments will resolve traits relative
            // to the parent module. Make sure the parent module's traits-in-scope are
            // loaded, even if the module itself has no doc comments.
            self.add_traits_in_parent_scope(self.current_mod.to_def_id());

            self.load_links_in_attrs(&item.attrs, item.span);
            self.process_module_children_or_reexports(self.current_mod.to_def_id());
            visit::walk_item(self, item);

            self.current_mod = old_mod;
        } else {
            match item.kind {
                ItemKind::Trait(..) => {
                    self.all_traits.push(self.resolver.local_def_id(item.id).to_def_id());
                }
                ItemKind::Impl(box ast::Impl { of_trait: Some(..), .. }) => {
                    self.all_trait_impls.push(self.resolver.local_def_id(item.id).to_def_id());
                }
                _ => {}
            }
            self.load_links_in_attrs(&item.attrs, item.span);
            visit::walk_item(self, item);
        }
    }

    fn visit_assoc_item(&mut self, item: &ast::AssocItem, ctxt: AssocCtxt) {
        self.load_links_in_attrs(&item.attrs, item.span);
        visit::walk_assoc_item(self, item, ctxt)
    }

    fn visit_foreign_item(&mut self, item: &ast::ForeignItem) {
        self.load_links_in_attrs(&item.attrs, item.span);
        visit::walk_foreign_item(self, item)
    }

    fn visit_variant(&mut self, v: &ast::Variant) {
        self.load_links_in_attrs(&v.attrs, v.span);
        visit::walk_variant(self, v)
    }

    fn visit_field_def(&mut self, field: &ast::FieldDef) {
        self.load_links_in_attrs(&field.attrs, field.span);
        visit::walk_field_def(self, field)
    }

    // NOTE: if doc-comments are ever allowed on other nodes (e.g. function parameters),
    // then this will have to implement other visitor methods too.
}
