//! The Rust AST Visitor. Extracts useful information and massages it into a form
//! usable for `clean`.

use rustc_ast as ast;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_hir as hir;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::DefId;
use rustc_hir::Node;
use rustc_middle::middle::privacy::AccessLevel;
use rustc_middle::ty::TyCtxt;
use rustc_span::source_map::Spanned;
use rustc_span::symbol::{kw, sym, Ident, Symbol};
use rustc_span::{self, Span};

use std::mem;

use crate::clean::{self, AttributesExt, NestedAttributesExt};
use crate::core;
use crate::doctree::*;

// FIXME: Should this be replaced with tcx.def_path_str?
fn def_id_to_path(tcx: TyCtxt<'_>, did: DefId) -> Vec<String> {
    let crate_name = tcx.crate_name(did.krate).to_string();
    let relative = tcx.def_path(did).data.into_iter().filter_map(|elem| {
        // extern blocks have an empty name
        let s = elem.data.to_string();
        if !s.is_empty() { Some(s) } else { None }
    });
    std::iter::once(crate_name).chain(relative).collect()
}

// Also, is there some reason that this doesn't use the 'visit'
// framework from syntax?.

crate struct RustdocVisitor<'a, 'tcx> {
    cx: &'a mut core::DocContext<'tcx>,
    view_item_stack: FxHashSet<hir::HirId>,
    inlining: bool,
    /// Are the current module and all of its parents public?
    inside_public_path: bool,
    exact_paths: FxHashMap<DefId, Vec<String>>,
}

impl<'a, 'tcx> RustdocVisitor<'a, 'tcx> {
    crate fn new(cx: &'a mut core::DocContext<'tcx>) -> RustdocVisitor<'a, 'tcx> {
        // If the root is re-exported, terminate all recursion.
        let mut stack = FxHashSet::default();
        stack.insert(hir::CRATE_HIR_ID);
        RustdocVisitor {
            cx,
            view_item_stack: stack,
            inlining: false,
            inside_public_path: true,
            exact_paths: FxHashMap::default(),
        }
    }

    fn store_path(&mut self, did: DefId) {
        let tcx = self.cx.tcx;
        self.exact_paths.entry(did).or_insert_with(|| def_id_to_path(tcx, did));
    }

    crate fn visit(mut self, krate: &'tcx hir::Crate<'_>) -> Module<'tcx> {
        let mut module = self.visit_mod_contents(
            krate.item.span,
            krate.item.attrs,
            &Spanned { span: rustc_span::DUMMY_SP, node: hir::VisibilityKind::Public },
            hir::CRATE_HIR_ID,
            &krate.item.module,
            None,
        );
        // Attach the crate's exported macros to the top-level module:
        module
            .macros
            .extend(krate.exported_macros.iter().map(|def| self.visit_local_macro(def, None)));
        module.is_crate = true;

        self.cx.renderinfo.get_mut().exact_paths = self.exact_paths;

        module
    }

    fn visit_mod_contents(
        &mut self,
        span: Span,
        attrs: &'tcx [ast::Attribute],
        vis: &'tcx hir::Visibility<'_>,
        id: hir::HirId,
        m: &'tcx hir::Mod<'tcx>,
        name: Option<Symbol>,
    ) -> Module<'tcx> {
        let mut om = Module::new(name, attrs);
        om.where_outer = span;
        om.where_inner = m.inner;
        om.id = id;
        // Keep track of if there were any private modules in the path.
        let orig_inside_public_path = self.inside_public_path;
        self.inside_public_path &= vis.node.is_pub();
        for i in m.item_ids {
            let item = self.cx.tcx.hir().expect_item(i.id);
            self.visit_item(item, None, &mut om);
        }
        self.inside_public_path = orig_inside_public_path;
        om
    }

    /// Tries to resolve the target of a `crate use` statement and inlines the
    /// target if it is defined locally and would not be documented otherwise,
    /// or when it is specifically requested with `please_inline`.
    /// (the latter is the case when the import is marked `doc(inline)`)
    ///
    /// Cross-crate inlining occurs later on during crate cleaning
    /// and follows different rules.
    ///
    /// Returns `true` if the target has been inlined.
    fn maybe_inline_local(
        &mut self,
        id: hir::HirId,
        res: Res,
        renamed: Option<Ident>,
        glob: bool,
        om: &mut Module<'tcx>,
        please_inline: bool,
    ) -> bool {
        fn inherits_doc_hidden(cx: &core::DocContext<'_>, mut node: hir::HirId) -> bool {
            while let Some(id) = cx.tcx.hir().get_enclosing_scope(node) {
                node = id;
                if cx.tcx.hir().attrs(node).lists(sym::doc).has_word(sym::hidden) {
                    return true;
                }
                if node == hir::CRATE_HIR_ID {
                    break;
                }
            }
            false
        }

        debug!("maybe_inline_local res: {:?}", res);

        let tcx = self.cx.tcx;
        let res_did = if let Some(did) = res.opt_def_id() {
            did
        } else {
            return false;
        };

        let use_attrs = tcx.hir().attrs(id);
        // Don't inline `doc(hidden)` imports so they can be stripped at a later stage.
        let is_no_inline = use_attrs.lists(sym::doc).has_word(sym::no_inline)
            || use_attrs.lists(sym::doc).has_word(sym::hidden);

        // For cross-crate impl inlining we need to know whether items are
        // reachable in documentation -- a previously nonreachable item can be
        // made reachable by cross-crate inlining which we're checking here.
        // (this is done here because we need to know this upfront).
        if !res_did.is_local() && !is_no_inline {
            let attrs = clean::inline::load_attrs(self.cx, res_did);
            let self_is_hidden = attrs.lists(sym::doc).has_word(sym::hidden);
            if !self_is_hidden {
                if let Res::Def(kind, did) = res {
                    if kind == DefKind::Mod {
                        crate::visit_lib::LibEmbargoVisitor::new(self.cx).visit_mod(did)
                    } else {
                        // All items need to be handled here in case someone wishes to link
                        // to them with intra-doc links
                        self.cx
                            .renderinfo
                            .get_mut()
                            .access_levels
                            .map
                            .insert(did, AccessLevel::Public);
                    }
                }
            }
            return false;
        }

        let res_hir_id = match res_did.as_local() {
            Some(n) => tcx.hir().local_def_id_to_hir_id(n),
            None => return false,
        };

        let is_private = !self.cx.renderinfo.borrow().access_levels.is_public(res_did);
        let is_hidden = inherits_doc_hidden(self.cx, res_hir_id);

        // Only inline if requested or if the item would otherwise be stripped.
        if (!please_inline && !is_private && !is_hidden) || is_no_inline {
            return false;
        }

        if !self.view_item_stack.insert(res_hir_id) {
            return false;
        }

        let ret = match tcx.hir().get(res_hir_id) {
            Node::Item(&hir::Item { kind: hir::ItemKind::Mod(ref m), .. }) if glob => {
                let prev = mem::replace(&mut self.inlining, true);
                for i in m.item_ids {
                    let i = self.cx.tcx.hir().expect_item(i.id);
                    self.visit_item(i, None, om);
                }
                self.inlining = prev;
                true
            }
            Node::Item(it) if !glob => {
                let prev = mem::replace(&mut self.inlining, true);
                self.visit_item(it, renamed, om);
                self.inlining = prev;
                true
            }
            Node::ForeignItem(it) if !glob => {
                let prev = mem::replace(&mut self.inlining, true);
                self.visit_foreign_item(it, renamed, om);
                self.inlining = prev;
                true
            }
            Node::MacroDef(def) if !glob => {
                om.macros.push(self.visit_local_macro(def, renamed.map(|i| i.name)));
                true
            }
            _ => false,
        };
        self.view_item_stack.remove(&res_hir_id);
        ret
    }

    fn visit_item(
        &mut self,
        item: &'tcx hir::Item<'_>,
        renamed: Option<Ident>,
        om: &mut Module<'tcx>,
    ) {
        debug!("visiting item {:?}", item);
        let ident = renamed.unwrap_or(item.ident);

        if item.vis.node.is_pub() {
            let def_id = self.cx.tcx.hir().local_def_id(item.hir_id);
            self.store_path(def_id.to_def_id());
        }

        match item.kind {
            hir::ItemKind::ForeignMod(ref fm) => {
                for item in fm.items {
                    self.visit_foreign_item(item, None, om);
                }
            }
            // If we're inlining, skip private items.
            _ if self.inlining && !item.vis.node.is_pub() => {}
            hir::ItemKind::GlobalAsm(..) => {}
            hir::ItemKind::Use(_, hir::UseKind::ListStem) => {}
            hir::ItemKind::Use(ref path, kind) => {
                let is_glob = kind == hir::UseKind::Glob;

                // Struct and variant constructors and proc macro stubs always show up alongside
                // their definitions, we've already processed them so just discard these.
                if let Res::Def(DefKind::Ctor(..), _) | Res::SelfCtor(..) = path.res {
                    return;
                }

                // If there was a private module in the current path then don't bother inlining
                // anything as it will probably be stripped anyway.
                if item.vis.node.is_pub() && self.inside_public_path {
                    let please_inline = item.attrs.iter().any(|item| match item.meta_item_list() {
                        Some(ref list) if item.has_name(sym::doc) => {
                            list.iter().any(|i| i.has_name(sym::inline))
                        }
                        _ => false,
                    });
                    let ident = if is_glob { None } else { Some(ident) };
                    if self.maybe_inline_local(
                        item.hir_id,
                        path.res,
                        ident,
                        is_glob,
                        om,
                        please_inline,
                    ) {
                        return;
                    }
                }

                om.imports.push(Import {
                    name: ident.name,
                    id: item.hir_id,
                    vis: &item.vis,
                    attrs: &item.attrs,
                    path,
                    glob: is_glob,
                    span: item.span,
                });
            }
            hir::ItemKind::Mod(ref m) => {
                om.mods.push(self.visit_mod_contents(
                    item.span,
                    &item.attrs,
                    &item.vis,
                    item.hir_id,
                    m,
                    Some(ident.name),
                ));
            }
            hir::ItemKind::Fn(..)
            | hir::ItemKind::ExternCrate(..)
            | hir::ItemKind::Enum(..)
            | hir::ItemKind::Struct(..)
            | hir::ItemKind::Union(..)
            | hir::ItemKind::TyAlias(..)
            | hir::ItemKind::OpaqueTy(..)
            | hir::ItemKind::Static(..)
            | hir::ItemKind::Trait(..)
            | hir::ItemKind::TraitAlias(..) => om.items.push((item, renamed)),
            hir::ItemKind::Const(..) => {
                // Underscore constants do not correspond to a nameable item and
                // so are never useful in documentation.
                if ident.name != kw::Underscore {
                    om.items.push((item, renamed));
                }
            }
            hir::ItemKind::Impl { ref of_trait, .. } => {
                // Don't duplicate impls when inlining or if it's implementing a trait, we'll pick
                // them up regardless of where they're located.
                if !self.inlining && of_trait.is_none() {
                    om.items.push((item, None));
                }
            }
        }
    }

    fn visit_foreign_item(
        &mut self,
        item: &'tcx hir::ForeignItem<'_>,
        renamed: Option<Ident>,
        om: &mut Module<'tcx>,
    ) {
        // If inlining we only want to include public functions.
        if !self.inlining || item.vis.node.is_pub() {
            om.foreigns.push((item, renamed));
        }
    }

    // Convert each `exported_macro` into a doc item.
    fn visit_local_macro(&self, def: &'tcx hir::MacroDef<'_>, renamed: Option<Symbol>) -> Macro {
        debug!("visit_local_macro: {}", def.ident);
        let tts = def.ast.body.inner_tokens().trees().collect::<Vec<_>>();
        // Extract the spans of all matchers. They represent the "interface" of the macro.
        let matchers = tts.chunks(4).map(|arm| arm[0].span()).collect();

        Macro {
            def_id: self.cx.tcx.hir().local_def_id(def.hir_id).to_def_id(),
            name: renamed.unwrap_or(def.ident.name),
            matchers,
            imported_from: None,
        }
    }
}
