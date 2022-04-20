use crate::clean::Attributes;
use crate::core::ResolverCaches;
use crate::passes::collect_intra_doc_links::preprocessed_markdown_links;
use crate::passes::collect_intra_doc_links::PreprocessedMarkdownLink;

use rustc_ast::visit::{self, AssocCtxt, Visitor};
use rustc_ast::{self as ast, ItemKind};
use rustc_ast_lowering::ResolverAstLowering;
use rustc_data_structures::fx::FxHashMap;
use rustc_hir::def::Namespace::*;
use rustc_hir::def::{DefKind, Namespace, Res};
use rustc_hir::def_id::{DefId, DefIdMap, DefIdSet, LocalDefId, CRATE_DEF_ID};
use rustc_hir::TraitCandidate;
use rustc_middle::ty::{DefIdTree, Visibility};
use rustc_resolve::{ParentScope, Resolver};
use rustc_session::config::Externs;
use rustc_span::{Symbol, SyntaxContext};

use std::collections::hash_map::Entry;
use std::mem;

crate fn early_resolve_intra_doc_links(
    resolver: &mut Resolver<'_>,
    krate: &ast::Crate,
    externs: Externs,
    document_private_items: bool,
) -> ResolverCaches {
    let mut link_resolver = EarlyDocLinkResolver {
        resolver,
        current_mod: CRATE_DEF_ID,
        visited_mods: Default::default(),
        markdown_links: Default::default(),
        doc_link_resolutions: Default::default(),
        traits_in_scope: Default::default(),
        all_traits: Default::default(),
        all_trait_impls: Default::default(),
        document_private_items,
    };

    // Overridden `visit_item` below doesn't apply to the crate root,
    // so we have to visit its attributes and reexports separately.
    link_resolver.resolve_doc_links_local(&krate.attrs);
    link_resolver.process_module_children_or_reexports(CRATE_DEF_ID.to_def_id());
    visit::walk_crate(&mut link_resolver, krate);
    link_resolver.process_extern_impls();

    // FIXME: somehow rustdoc is still missing crates even though we loaded all
    // the known necessary crates. Load them all unconditionally until we find a way to fix this.
    // DO NOT REMOVE THIS without first testing on the reproducer in
    // https://github.com/jyn514/objr/commit/edcee7b8124abf0e4c63873e8422ff81beb11ebb
    for (extern_name, _) in externs.iter().filter(|(_, entry)| entry.add_prelude) {
        link_resolver.resolver.resolve_rustdoc_path(extern_name, TypeNS, CRATE_DEF_ID.to_def_id());
    }

    ResolverCaches {
        markdown_links: Some(link_resolver.markdown_links),
        doc_link_resolutions: link_resolver.doc_link_resolutions,
        traits_in_scope: link_resolver.traits_in_scope,
        all_traits: Some(link_resolver.all_traits),
        all_trait_impls: Some(link_resolver.all_trait_impls),
        all_macro_rules: link_resolver.resolver.take_all_macro_rules(),
    }
}

fn doc_attrs<'a>(attrs: impl Iterator<Item = &'a ast::Attribute>) -> Attributes {
    let mut attrs = Attributes::from_ast_iter(attrs.map(|attr| (attr, None)), true);
    attrs.unindent_doc_comments();
    attrs
}

struct EarlyDocLinkResolver<'r, 'ra> {
    resolver: &'r mut Resolver<'ra>,
    current_mod: LocalDefId,
    visited_mods: DefIdSet,
    markdown_links: FxHashMap<String, Vec<PreprocessedMarkdownLink>>,
    doc_link_resolutions: FxHashMap<(Symbol, Namespace, DefId), Option<Res<ast::NodeId>>>,
    traits_in_scope: DefIdMap<Vec<TraitCandidate>>,
    all_traits: Vec<DefId>,
    all_trait_impls: Vec<DefId>,
    document_private_items: bool,
}

impl EarlyDocLinkResolver<'_, '_> {
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

    /// Add traits in scope for links in impls collected by the `collect-intra-doc-links` pass.
    /// That pass filters impls using type-based information, but we don't yet have such
    /// information here, so we just conservatively calculate traits in scope for *all* modules
    /// having impls in them.
    fn process_extern_impls(&mut self) {
        // Resolving links in already existing crates may trigger loading of new crates.
        let mut start_cnum = 0;
        loop {
            let crates = Vec::from_iter(self.resolver.cstore().crates_untracked());
            for &cnum in &crates[start_cnum..] {
                let all_traits =
                    Vec::from_iter(self.resolver.cstore().traits_in_crate_untracked(cnum));
                let all_trait_impls =
                    Vec::from_iter(self.resolver.cstore().trait_impls_in_crate_untracked(cnum));
                let all_inherent_impls =
                    Vec::from_iter(self.resolver.cstore().inherent_impls_in_crate_untracked(cnum));
                let all_incoherent_impls = Vec::from_iter(
                    self.resolver.cstore().incoherent_impls_in_crate_untracked(cnum),
                );

                // Querying traits in scope is expensive so we try to prune the impl and traits lists
                // using privacy, private traits and impls from other crates are never documented in
                // the current crate, and links in their doc comments are not resolved.
                for &def_id in &all_traits {
                    if self.resolver.cstore().visibility_untracked(def_id) == Visibility::Public {
                        self.resolve_doc_links_extern_impl(def_id, false);
                    }
                }
                for &(trait_def_id, impl_def_id, simplified_self_ty) in &all_trait_impls {
                    if self.resolver.cstore().visibility_untracked(trait_def_id)
                        == Visibility::Public
                        && simplified_self_ty.and_then(|ty| ty.def()).map_or(true, |ty_def_id| {
                            self.resolver.cstore().visibility_untracked(ty_def_id)
                                == Visibility::Public
                        })
                    {
                        self.resolve_doc_links_extern_impl(impl_def_id, false);
                    }
                }
                for (ty_def_id, impl_def_id) in all_inherent_impls {
                    if self.resolver.cstore().visibility_untracked(ty_def_id) == Visibility::Public
                    {
                        self.resolve_doc_links_extern_impl(impl_def_id, true);
                    }
                }
                for impl_def_id in all_incoherent_impls {
                    self.resolve_doc_links_extern_impl(impl_def_id, true);
                }

                self.all_traits.extend(all_traits);
                self.all_trait_impls
                    .extend(all_trait_impls.into_iter().map(|(_, def_id, _)| def_id));
            }

            if crates.len() > start_cnum {
                start_cnum = crates.len();
            } else {
                break;
            }
        }
    }

    fn resolve_doc_links_extern_impl(&mut self, def_id: DefId, _is_inherent: bool) {
        // FIXME: Resolve links in associated items in addition to traits themselves,
        // `force` is used to provide traits in scope for the associated items.
        self.resolve_doc_links_extern_outer(def_id, def_id, true);
    }

    fn resolve_doc_links_extern_outer(&mut self, def_id: DefId, scope_id: DefId, force: bool) {
        if !force && !self.resolver.cstore().may_have_doc_links_untracked(def_id) {
            return;
        }
        // FIXME: actually resolve links, not just add traits in scope.
        if let Some(parent_id) = self.resolver.parent(scope_id) {
            self.add_traits_in_scope(parent_id);
        }
    }

    fn resolve_doc_links_extern_inner(&mut self, def_id: DefId) {
        if !self.resolver.cstore().may_have_doc_links_untracked(def_id) {
            return;
        }
        // FIXME: actually resolve links, not just add traits in scope.
        self.add_traits_in_scope(def_id);
    }

    fn resolve_doc_links_local(&mut self, attrs: &[ast::Attribute]) {
        if !attrs.iter().any(|attr| attr.may_have_doc_links()) {
            return;
        }
        let module_id = self.current_mod.to_def_id();
        self.resolve_doc_links(doc_attrs(attrs.iter()), module_id);
    }

    fn resolve_doc_links(&mut self, attrs: Attributes, module_id: DefId) {
        let mut need_traits_in_scope = false;
        for (doc_module, doc) in attrs.prepare_to_doc_link_resolution() {
            assert_eq!(doc_module, None);
            let links = self
                .markdown_links
                .entry(doc)
                .or_insert_with_key(|doc| preprocessed_markdown_links(doc));
            for PreprocessedMarkdownLink(pp_link, _) in links {
                if let Ok(pinfo) = pp_link {
                    // FIXME: Resolve the path in all namespaces and resolve its prefixes too.
                    let ns = TypeNS;
                    self.doc_link_resolutions
                        .entry((Symbol::intern(&pinfo.path_str), ns, module_id))
                        .or_insert_with_key(|(path, ns, module_id)| {
                            self.resolver.resolve_rustdoc_path(path.as_str(), *ns, *module_id)
                        });
                    need_traits_in_scope = true;
                }
            }
        }

        if need_traits_in_scope {
            self.add_traits_in_scope(module_id);
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
            // This condition should give a superset of `denied` from `fn clean_use_statement`.
            if child.vis == Visibility::Public
                || self.document_private_items
                    && child.vis != Visibility::Restricted(module_id)
                    && module_id.is_local()
            {
                if let Some(def_id) = child.res.opt_def_id() && !def_id.is_local() {
                    let scope_id = match child.res {
                        Res::Def(DefKind::Variant, ..) => self.resolver.parent(def_id).unwrap(),
                        _ => def_id,
                    };
                    self.resolve_doc_links_extern_outer(def_id, scope_id, false); // Outer attribute scope
                    if let Res::Def(DefKind::Mod, ..) = child.res {
                        self.resolve_doc_links_extern_inner(def_id); // Inner attribute scope
                    }
                    // Traits are processed in `add_extern_traits_in_scope`.
                    if let Res::Def(DefKind::Mod | DefKind::Enum, ..) = child.res {
                        self.process_module_children_or_reexports(def_id);
                    }
                }
            }
        }
    }
}

impl Visitor<'_> for EarlyDocLinkResolver<'_, '_> {
    fn visit_item(&mut self, item: &ast::Item) {
        self.resolve_doc_links_local(&item.attrs); // Outer attribute scope
        if let ItemKind::Mod(..) = item.kind {
            let old_mod = mem::replace(&mut self.current_mod, self.resolver.local_def_id(item.id));
            self.resolve_doc_links_local(&item.attrs); // Inner attribute scope
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
            visit::walk_item(self, item);
        }
    }

    fn visit_assoc_item(&mut self, item: &ast::AssocItem, ctxt: AssocCtxt) {
        self.resolve_doc_links_local(&item.attrs);
        visit::walk_assoc_item(self, item, ctxt)
    }

    fn visit_foreign_item(&mut self, item: &ast::ForeignItem) {
        self.resolve_doc_links_local(&item.attrs);
        visit::walk_foreign_item(self, item)
    }

    fn visit_variant(&mut self, v: &ast::Variant) {
        self.resolve_doc_links_local(&v.attrs);
        visit::walk_variant(self, v)
    }

    fn visit_field_def(&mut self, field: &ast::FieldDef) {
        self.resolve_doc_links_local(&field.attrs);
        visit::walk_field_def(self, field)
    }

    // NOTE: if doc-comments are ever allowed on other nodes (e.g. function parameters),
    // then this will have to implement other visitor methods too.
}
