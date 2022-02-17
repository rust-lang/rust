//! The Rust AST Visitor. Extracts useful information and massages it into a form
//! usable for `clean`.

use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_hir as hir;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::DefId;
use rustc_hir::Node;
use rustc_hir::CRATE_HIR_ID;
use rustc_middle::middle::privacy::AccessLevel;
use rustc_middle::ty::TyCtxt;
use rustc_span::def_id::{CRATE_DEF_ID, LOCAL_CRATE};
use rustc_span::symbol::{kw, sym, Symbol};
use rustc_span::Span;

use std::mem;

use crate::clean::{self, cfg::Cfg, AttributesExt, NestedAttributesExt};
use crate::core;

/// This module is used to store stuff from Rust's AST in a more convenient
/// manner (and with prettier names) before cleaning.
#[derive(Debug)]
crate struct Module<'hir> {
    crate name: Symbol,
    crate where_inner: Span,
    crate mods: Vec<Module<'hir>>,
    crate id: hir::HirId,
    // (item, renamed)
    crate items: Vec<(&'hir hir::Item<'hir>, Option<Symbol>)>,
    crate foreigns: Vec<(&'hir hir::ForeignItem<'hir>, Option<Symbol>)>,
}

impl Module<'_> {
    crate fn new(name: Symbol, id: hir::HirId, where_inner: Span) -> Self {
        Module { name, id, where_inner, mods: Vec::new(), items: Vec::new(), foreigns: Vec::new() }
    }

    crate fn where_outer(&self, tcx: TyCtxt<'_>) -> Span {
        tcx.hir().span(self.id)
    }
}

// FIXME: Should this be replaced with tcx.def_path_str?
fn def_id_to_path(tcx: TyCtxt<'_>, did: DefId) -> Vec<Symbol> {
    let crate_name = tcx.crate_name(did.krate);
    let relative = tcx.def_path(did).data.into_iter().filter_map(|elem| elem.data.get_opt_name());
    std::iter::once(crate_name).chain(relative).collect()
}

crate fn inherits_doc_hidden(tcx: TyCtxt<'_>, mut node: hir::HirId) -> bool {
    while let Some(id) = tcx.hir().get_enclosing_scope(node) {
        node = id;
        if tcx.hir().attrs(node).lists(sym::doc).has_word(sym::hidden) {
            return true;
        }
    }
    false
}

// Also, is there some reason that this doesn't use the 'visit'
// framework from syntax?.

crate struct RustdocVisitor<'a, 'tcx> {
    cx: &'a mut core::DocContext<'tcx>,
    view_item_stack: FxHashSet<hir::HirId>,
    inlining: bool,
    /// Are the current module and all of its parents public?
    inside_public_path: bool,
    exact_paths: FxHashMap<DefId, Vec<Symbol>>,
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

    crate fn visit(mut self) -> Module<'tcx> {
        let mut top_level_module = self.visit_mod_contents(
            hir::CRATE_HIR_ID,
            self.cx.tcx.hir().root_module(),
            self.cx.tcx.crate_name(LOCAL_CRATE),
        );

        // `#[macro_export] macro_rules!` items are reexported at the top level of the
        // crate, regardless of where they're defined. We want to document the
        // top level rexport of the macro, not its original definition, since
        // the rexport defines the path that a user will actually see. Accordingly,
        // we add the rexport as an item here, and then skip over the original
        // definition in `visit_item()` below.
        //
        // We also skip `#[macro_export] macro_rules!` that have already been inserted,
        // it can happen if within the same module a `#[macro_export] macro_rules!`
        // is declared but also a reexport of itself producing two exports of the same
        // macro in the same module.
        let mut inserted = FxHashSet::default();
        for export in self.cx.tcx.module_reexports(CRATE_DEF_ID).unwrap_or(&[]) {
            if let Res::Def(DefKind::Macro(_), def_id) = export.res {
                if let Some(local_def_id) = def_id.as_local() {
                    if self.cx.tcx.has_attr(def_id, sym::macro_export) {
                        if inserted.insert(def_id) {
                            let item = self.cx.tcx.hir().expect_item(local_def_id);
                            top_level_module.items.push((item, None));
                        }
                    }
                }
            }
        }

        self.cx.cache.hidden_cfg = self
            .cx
            .tcx
            .hir()
            .attrs(CRATE_HIR_ID)
            .iter()
            .filter(|attr| attr.has_name(sym::doc))
            .flat_map(|attr| attr.meta_item_list().into_iter().flatten())
            .filter(|attr| attr.has_name(sym::cfg_hide))
            .flat_map(|attr| {
                attr.meta_item_list()
                    .unwrap_or(&[])
                    .iter()
                    .filter_map(|attr| {
                        Cfg::parse(attr.meta_item()?)
                            .map_err(|e| self.cx.sess().diagnostic().span_err(e.span, e.msg))
                            .ok()
                    })
                    .collect::<Vec<_>>()
            })
            .chain([Cfg::Cfg(sym::test, None)].into_iter())
            .collect();

        self.cx.cache.exact_paths = self.exact_paths;
        top_level_module
    }

    fn visit_mod_contents(
        &mut self,
        id: hir::HirId,
        m: &'tcx hir::Mod<'tcx>,
        name: Symbol,
    ) -> Module<'tcx> {
        let mut om = Module::new(name, id, m.inner);
        let def_id = self.cx.tcx.hir().local_def_id(id).to_def_id();
        // Keep track of if there were any private modules in the path.
        let orig_inside_public_path = self.inside_public_path;
        self.inside_public_path &= self.cx.tcx.visibility(def_id).is_public();
        for &i in m.item_ids {
            let item = self.cx.tcx.hir().item(i);
            self.visit_item(item, None, &mut om);
        }
        self.inside_public_path = orig_inside_public_path;
        om
    }

    /// Tries to resolve the target of a `pub use` statement and inlines the
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
        renamed: Option<Symbol>,
        glob: bool,
        om: &mut Module<'tcx>,
        please_inline: bool,
    ) -> bool {
        debug!("maybe_inline_local res: {:?}", res);

        let tcx = self.cx.tcx;
        let Some(res_did) = res.opt_def_id() else {
            return false;
        };

        let use_attrs = tcx.hir().attrs(id);
        // Don't inline `doc(hidden)` imports so they can be stripped at a later stage.
        let is_no_inline = use_attrs.lists(sym::doc).has_word(sym::no_inline)
            || use_attrs.lists(sym::doc).has_word(sym::hidden);

        // For cross-crate impl inlining we need to know whether items are
        // reachable in documentation -- a previously unreachable item can be
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
                        self.cx.cache.access_levels.map.insert(did, AccessLevel::Public);
                    }
                }
            }
            return false;
        }

        let res_hir_id = match res_did.as_local() {
            Some(n) => tcx.hir().local_def_id_to_hir_id(n),
            None => return false,
        };

        let is_private = !self.cx.cache.access_levels.is_public(res_did);
        let is_hidden = inherits_doc_hidden(self.cx.tcx, res_hir_id);

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
                for &i in m.item_ids {
                    let i = self.cx.tcx.hir().item(i);
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
            _ => false,
        };
        self.view_item_stack.remove(&res_hir_id);
        ret
    }

    fn visit_item(
        &mut self,
        item: &'tcx hir::Item<'_>,
        renamed: Option<Symbol>,
        om: &mut Module<'tcx>,
    ) {
        debug!("visiting item {:?}", item);
        let name = renamed.unwrap_or(item.ident.name);

        let def_id = item.def_id.to_def_id();
        let is_pub = self.cx.tcx.visibility(def_id).is_public();

        if is_pub {
            self.store_path(item.def_id.to_def_id());
        }

        match item.kind {
            hir::ItemKind::ForeignMod { items, .. } => {
                for item in items {
                    let item = self.cx.tcx.hir().foreign_item(item.id);
                    self.visit_foreign_item(item, None, om);
                }
            }
            // If we're inlining, skip private items.
            _ if self.inlining && !is_pub => {}
            hir::ItemKind::GlobalAsm(..) => {}
            hir::ItemKind::Use(_, hir::UseKind::ListStem) => {}
            hir::ItemKind::Use(path, kind) => {
                let is_glob = kind == hir::UseKind::Glob;

                // Struct and variant constructors and proc macro stubs always show up alongside
                // their definitions, we've already processed them so just discard these.
                if let Res::Def(DefKind::Ctor(..), _) | Res::SelfCtor(..) = path.res {
                    return;
                }

                let attrs = self.cx.tcx.hir().attrs(item.hir_id());

                // If there was a private module in the current path then don't bother inlining
                // anything as it will probably be stripped anyway.
                if is_pub && self.inside_public_path {
                    let please_inline = attrs.iter().any(|item| match item.meta_item_list() {
                        Some(ref list) if item.has_name(sym::doc) => {
                            list.iter().any(|i| i.has_name(sym::inline))
                        }
                        _ => false,
                    });
                    let ident = if is_glob { None } else { Some(name) };
                    if self.maybe_inline_local(
                        item.hir_id(),
                        path.res,
                        ident,
                        is_glob,
                        om,
                        please_inline,
                    ) {
                        return;
                    }
                }

                om.items.push((item, renamed))
            }
            hir::ItemKind::Macro(ref macro_def) => {
                // `#[macro_export] macro_rules!` items are handled seperately in `visit()`,
                // above, since they need to be documented at the module top level. Accordingly,
                // we only want to handle macros if one of three conditions holds:
                //
                // 1. This macro was defined by `macro`, and thus isn't covered by the case
                //    above.
                // 2. This macro isn't marked with `#[macro_export]`, and thus isn't covered
                //    by the case above.
                // 3. We're inlining, since a reexport where inlining has been requested
                //    should be inlined even if it is also documented at the top level.

                let def_id = item.def_id.to_def_id();
                let is_macro_2_0 = !macro_def.macro_rules;
                let nonexported = !self.cx.tcx.has_attr(def_id, sym::macro_export);

                if is_macro_2_0 || nonexported || self.inlining {
                    om.items.push((item, renamed));
                }
            }
            hir::ItemKind::Mod(ref m) => {
                om.mods.push(self.visit_mod_contents(item.hir_id(), m, name));
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
                if name != kw::Underscore {
                    om.items.push((item, renamed));
                }
            }
            hir::ItemKind::Impl(ref impl_) => {
                // Don't duplicate impls when inlining or if it's implementing a trait, we'll pick
                // them up regardless of where they're located.
                if !self.inlining && impl_.of_trait.is_none() {
                    om.items.push((item, None));
                }
            }
        }
    }

    fn visit_foreign_item(
        &mut self,
        item: &'tcx hir::ForeignItem<'_>,
        renamed: Option<Symbol>,
        om: &mut Module<'tcx>,
    ) {
        // If inlining we only want to include public functions.
        if !self.inlining || self.cx.tcx.visibility(item.def_id).is_public() {
            om.foreigns.push((item, renamed));
        }
    }
}
