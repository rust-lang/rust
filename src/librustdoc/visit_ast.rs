//! The Rust AST Visitor. Extracts useful information and massages it into a form
//! usable for `clean`.

use std::mem;

use rustc_data_structures::fx::{FxHashSet, FxIndexMap};
use rustc_hir as hir;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::{DefId, DefIdMap, LocalDefId, LocalDefIdSet};
use rustc_hir::intravisit::{Visitor, walk_body, walk_item};
use rustc_hir::{CRATE_HIR_ID, Node};
use rustc_middle::hir::nested_filter;
use rustc_middle::ty::TyCtxt;
use rustc_span::Span;
use rustc_span::def_id::{CRATE_DEF_ID, LOCAL_CRATE};
use rustc_span::hygiene::MacroKind;
use rustc_span::symbol::{Symbol, kw, sym};
use tracing::debug;

use crate::clean::cfg::Cfg;
use crate::clean::utils::{inherits_doc_hidden, should_ignore_res};
use crate::clean::{NestedAttributesExt, hir_attr_lists, reexport_chain};
use crate::core;

/// This module is used to store stuff from Rust's AST in a more convenient
/// manner (and with prettier names) before cleaning.
#[derive(Debug)]
pub(crate) struct Module<'hir> {
    pub(crate) name: Symbol,
    pub(crate) where_inner: Span,
    pub(crate) mods: Vec<Module<'hir>>,
    pub(crate) def_id: LocalDefId,
    pub(crate) renamed: Option<Symbol>,
    pub(crate) import_id: Option<LocalDefId>,
    /// The key is the item `ItemId` and the value is: (item, renamed, import_id).
    /// We use `FxIndexMap` to keep the insert order.
    pub(crate) items: FxIndexMap<
        (LocalDefId, Option<Symbol>),
        (&'hir hir::Item<'hir>, Option<Symbol>, Option<LocalDefId>),
    >,
    /// Same as for `items`.
    pub(crate) inlined_foreigns: FxIndexMap<(DefId, Option<Symbol>), (Res, LocalDefId)>,
    pub(crate) foreigns: Vec<(&'hir hir::ForeignItem<'hir>, Option<Symbol>)>,
}

impl Module<'_> {
    pub(crate) fn new(
        name: Symbol,
        def_id: LocalDefId,
        where_inner: Span,
        renamed: Option<Symbol>,
        import_id: Option<LocalDefId>,
    ) -> Self {
        Module {
            name,
            def_id,
            where_inner,
            renamed,
            import_id,
            mods: Vec::new(),
            items: FxIndexMap::default(),
            inlined_foreigns: FxIndexMap::default(),
            foreigns: Vec::new(),
        }
    }

    pub(crate) fn where_outer(&self, tcx: TyCtxt<'_>) -> Span {
        tcx.def_span(self.def_id)
    }
}

// FIXME: Should this be replaced with tcx.def_path_str?
fn def_id_to_path(tcx: TyCtxt<'_>, did: DefId) -> Vec<Symbol> {
    let crate_name = tcx.crate_name(did.krate);
    let relative = tcx.def_path(did).data.into_iter().filter_map(|elem| elem.data.get_opt_name());
    std::iter::once(crate_name).chain(relative).collect()
}

pub(crate) struct RustdocVisitor<'a, 'tcx> {
    cx: &'a mut core::DocContext<'tcx>,
    view_item_stack: LocalDefIdSet,
    inlining: bool,
    /// Are the current module and all of its parents public?
    inside_public_path: bool,
    exact_paths: DefIdMap<Vec<Symbol>>,
    modules: Vec<Module<'tcx>>,
    is_importable_from_parent: bool,
    inside_body: bool,
}

impl<'a, 'tcx> RustdocVisitor<'a, 'tcx> {
    pub(crate) fn new(cx: &'a mut core::DocContext<'tcx>) -> RustdocVisitor<'a, 'tcx> {
        // If the root is re-exported, terminate all recursion.
        let mut stack = LocalDefIdSet::default();
        stack.insert(CRATE_DEF_ID);
        let om = Module::new(
            cx.tcx.crate_name(LOCAL_CRATE),
            CRATE_DEF_ID,
            cx.tcx.hir_root_module().spans.inner_span,
            None,
            None,
        );

        RustdocVisitor {
            cx,
            view_item_stack: stack,
            inlining: false,
            inside_public_path: true,
            exact_paths: Default::default(),
            modules: vec![om],
            is_importable_from_parent: true,
            inside_body: false,
        }
    }

    fn store_path(&mut self, did: DefId) {
        let tcx = self.cx.tcx;
        self.exact_paths.entry(did).or_insert_with(|| def_id_to_path(tcx, did));
    }

    pub(crate) fn visit(mut self) -> Module<'tcx> {
        let root_module = self.cx.tcx.hir_root_module();
        self.visit_mod_contents(CRATE_DEF_ID, root_module);

        let mut top_level_module = self.modules.pop().unwrap();

        // `#[macro_export] macro_rules!` items are reexported at the top level of the
        // crate, regardless of where they're defined. We want to document the
        // top level re-export of the macro, not its original definition, since
        // the re-export defines the path that a user will actually see. Accordingly,
        // we add the re-export as an item here, and then skip over the original
        // definition in `visit_item()` below.
        //
        // We also skip `#[macro_export] macro_rules!` that have already been inserted,
        // it can happen if within the same module a `#[macro_export] macro_rules!`
        // is declared but also a reexport of itself producing two exports of the same
        // macro in the same module.
        let mut inserted = FxHashSet::default();
        for child in self.cx.tcx.module_children_local(CRATE_DEF_ID) {
            if !child.reexport_chain.is_empty()
                && let Res::Def(DefKind::Macro(_), def_id) = child.res
                && let Some(local_def_id) = def_id.as_local()
                && self.cx.tcx.has_attr(def_id, sym::macro_export)
                && inserted.insert(def_id)
            {
                let item = self.cx.tcx.hir_expect_item(local_def_id);
                let (ident, _, _) = item.expect_macro();
                top_level_module.items.insert((local_def_id, Some(ident.name)), (item, None, None));
            }
        }

        self.cx.cache.hidden_cfg = self
            .cx
            .tcx
            .hir_attrs(CRATE_HIR_ID)
            .iter()
            .filter(|attr| attr.has_name(sym::doc))
            .flat_map(|attr| attr.meta_item_list().into_iter().flatten())
            .filter(|attr| attr.has_name(sym::cfg_hide))
            .flat_map(|attr| {
                attr.meta_item_list()
                    .unwrap_or(&[])
                    .iter()
                    .filter_map(|attr| {
                        Cfg::parse(attr)
                            .map_err(|e| self.cx.sess().dcx().span_err(e.span, e.msg))
                            .ok()
                    })
                    .collect::<Vec<_>>()
            })
            .chain([
                Cfg::Cfg(sym::test, None),
                Cfg::Cfg(sym::doc, None),
                Cfg::Cfg(sym::doctest, None),
            ])
            .collect();

        self.cx.cache.exact_paths = self.exact_paths;
        top_level_module
    }

    /// This method will go through the given module items in two passes:
    /// 1. The items which are not glob imports/reexports.
    /// 2. The glob imports/reexports.
    fn visit_mod_contents(&mut self, def_id: LocalDefId, m: &'tcx hir::Mod<'tcx>) {
        debug!("Going through module {m:?}");
        // Keep track of if there were any private modules in the path.
        let orig_inside_public_path = self.inside_public_path;
        self.inside_public_path &= self.cx.tcx.local_visibility(def_id).is_public();

        // Reimplementation of `walk_mod` because we need to do it in two passes (explanations in
        // the second loop):
        for &i in m.item_ids {
            let item = self.cx.tcx.hir_item(i);
            if !matches!(item.kind, hir::ItemKind::Use(_, hir::UseKind::Glob)) {
                self.visit_item(item);
            }
        }
        for &i in m.item_ids {
            let item = self.cx.tcx.hir_item(i);
            // To match the way import precedence works, visit glob imports last.
            // Later passes in rustdoc will de-duplicate by name and kind, so if glob-
            // imported items appear last, then they'll be the ones that get discarded.
            if matches!(item.kind, hir::ItemKind::Use(_, hir::UseKind::Glob)) {
                self.visit_item(item);
            }
        }
        self.inside_public_path = orig_inside_public_path;
        debug!("Leaving module {m:?}");
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
        def_id: LocalDefId,
        res: Res,
        renamed: Option<Symbol>,
        please_inline: bool,
    ) -> bool {
        debug!("maybe_inline_local (renamed: {renamed:?}) res: {res:?}");

        let glob = renamed.is_none();
        if renamed == Some(kw::Underscore) {
            // We never inline `_` reexports.
            return false;
        }

        if self.cx.is_json_output() {
            return false;
        }

        let tcx = self.cx.tcx;
        let Some(ori_res_did) = res.opt_def_id() else {
            return false;
        };

        let document_hidden = self.cx.render_options.document_hidden;
        let use_attrs = tcx.hir_attrs(tcx.local_def_id_to_hir_id(def_id));
        // Don't inline `doc(hidden)` imports so they can be stripped at a later stage.
        let is_no_inline = hir_attr_lists(use_attrs, sym::doc).has_word(sym::no_inline)
            || (document_hidden && hir_attr_lists(use_attrs, sym::doc).has_word(sym::hidden));

        if is_no_inline {
            return false;
        }

        let is_hidden = !document_hidden && tcx.is_doc_hidden(ori_res_did);
        let Some(res_did) = ori_res_did.as_local() else {
            // For cross-crate impl inlining we need to know whether items are
            // reachable in documentation -- a previously unreachable item can be
            // made reachable by cross-crate inlining which we're checking here.
            // (this is done here because we need to know this upfront).
            crate::visit_lib::lib_embargo_visit_item(self.cx, ori_res_did);
            if is_hidden || glob {
                return false;
            }
            // We store inlined foreign items otherwise, it'd mean that the `use` item would be kept
            // around. It's not a problem unless this `use` imports both a local AND a foreign item.
            // If a local item is inlined, its `use` is not supposed to still be around in `clean`,
            // which would make appear the `use` in the generated documentation like the local item
            // was not inlined even though it actually was.
            self.modules
                .last_mut()
                .unwrap()
                .inlined_foreigns
                .insert((ori_res_did, renamed), (res, def_id));
            return true;
        };

        let is_private = !self.cx.cache.effective_visibilities.is_directly_public(tcx, ori_res_did);
        let item = tcx.hir_node_by_def_id(res_did);

        if !please_inline {
            let inherits_hidden = !document_hidden && inherits_doc_hidden(tcx, res_did, None);
            // Only inline if requested or if the item would otherwise be stripped.
            if (!is_private && !inherits_hidden) || (
                is_hidden &&
                // If it's a doc hidden module, we need to keep it in case some of its inner items
                // are re-exported.
                !matches!(item, Node::Item(&hir::Item { kind: hir::ItemKind::Mod(..), .. }))
            ) ||
                // The imported item is public and not `doc(hidden)` so no need to inline it.
                self.reexport_public_and_not_hidden(def_id, res_did)
            {
                return false;
            }
        }

        let is_bang_macro = matches!(
            item,
            Node::Item(&hir::Item { kind: hir::ItemKind::Macro(_, _, MacroKind::Bang), .. })
        );

        if !self.view_item_stack.insert(res_did) && !is_bang_macro {
            return false;
        }

        let inlined = match item {
            // Bang macros are handled a bit on their because of how they are handled by the
            // compiler. If they have `#[doc(hidden)]` and the re-export doesn't have
            // `#[doc(inline)]`, then we don't inline it.
            Node::Item(_) if is_bang_macro && !please_inline && renamed.is_some() && is_hidden => {
                return false;
            }
            Node::Item(&hir::Item { kind: hir::ItemKind::Mod(_, m), .. }) if glob => {
                let prev = mem::replace(&mut self.inlining, true);
                for &i in m.item_ids {
                    let i = tcx.hir_item(i);
                    self.visit_item_inner(i, None, Some(def_id));
                }
                self.inlining = prev;
                true
            }
            Node::Item(it) if !glob => {
                let prev = mem::replace(&mut self.inlining, true);
                self.visit_item_inner(it, renamed, Some(def_id));
                self.inlining = prev;
                true
            }
            Node::ForeignItem(it) if !glob => {
                let prev = mem::replace(&mut self.inlining, true);
                self.visit_foreign_item_inner(it, renamed);
                self.inlining = prev;
                true
            }
            _ => false,
        };
        self.view_item_stack.remove(&res_did);
        if inlined {
            self.cx.cache.inlined_items.insert(ori_res_did);
        }
        inlined
    }

    /// Returns `true` if the item is visible, meaning it's not `#[doc(hidden)]` or private.
    ///
    /// This function takes into account the entire re-export `use` chain, so it needs the
    /// ID of the "leaf" `use` and the ID of the "root" item.
    fn reexport_public_and_not_hidden(
        &self,
        import_def_id: LocalDefId,
        target_def_id: LocalDefId,
    ) -> bool {
        if self.cx.render_options.document_hidden {
            return true;
        }
        let tcx = self.cx.tcx;
        let item_def_id = reexport_chain(tcx, import_def_id, target_def_id.to_def_id())
            .iter()
            .flat_map(|reexport| reexport.id())
            .map(|id| id.expect_local())
            .nth(1)
            .unwrap_or(target_def_id);
        item_def_id != import_def_id
            && self.cx.cache.effective_visibilities.is_directly_public(tcx, item_def_id.to_def_id())
            && !tcx.is_doc_hidden(item_def_id)
            && !inherits_doc_hidden(tcx, item_def_id, None)
    }

    #[inline]
    fn add_to_current_mod(
        &mut self,
        item: &'tcx hir::Item<'_>,
        renamed: Option<Symbol>,
        parent_id: Option<LocalDefId>,
    ) {
        if self.is_importable_from_parent
            // If we're inside an item, only impl blocks and `macro_rules!` with the `macro_export`
            // attribute can still be visible.
            || match item.kind {
                hir::ItemKind::Impl(..) => true,
                hir::ItemKind::Macro(_, _, MacroKind::Bang) => {
                    self.cx.tcx.has_attr(item.owner_id.def_id, sym::macro_export)
                }
                _ => false,
            }
        {
            self.modules
                .last_mut()
                .unwrap()
                .items
                .insert((item.owner_id.def_id, renamed), (item, renamed, parent_id));
        }
    }

    fn visit_item_inner(
        &mut self,
        item: &'tcx hir::Item<'_>,
        renamed: Option<Symbol>,
        import_id: Option<LocalDefId>,
    ) {
        debug!("visiting item {item:?}");
        if self.inside_body {
            // Only impls can be "seen" outside a body. For example:
            //
            // ```
            // struct Bar;
            //
            // fn foo() {
            //     impl Bar { fn bar() {} }
            // }
            // Bar::bar();
            // ```
            if let hir::ItemKind::Impl(impl_) = item.kind &&
                // Don't duplicate impls when inlining or if it's implementing a trait, we'll pick
                // them up regardless of where they're located.
                impl_.of_trait.is_none()
            {
                self.add_to_current_mod(item, None, None);
            }
            return;
        }
        let get_name = || renamed.unwrap_or(item.kind.ident().unwrap().name);
        let tcx = self.cx.tcx;

        let def_id = item.owner_id.to_def_id();
        let is_pub = tcx.visibility(def_id).is_public();

        if is_pub {
            self.store_path(item.owner_id.to_def_id());
        }

        match item.kind {
            hir::ItemKind::ForeignMod { items, .. } => {
                for item in items {
                    let item = tcx.hir_foreign_item(item.id);
                    self.visit_foreign_item_inner(item, None);
                }
            }
            // If we're inlining, skip private items.
            _ if self.inlining && !is_pub => {}
            hir::ItemKind::GlobalAsm { .. } => {}
            hir::ItemKind::Use(_, hir::UseKind::ListStem) => {}
            hir::ItemKind::Use(path, kind) => {
                for res in path.res.present_items() {
                    // Struct and variant constructors and proc macro stubs always show up alongside
                    // their definitions, we've already processed them so just discard these.
                    if should_ignore_res(res) {
                        continue;
                    }

                    let attrs = tcx.hir_attrs(tcx.local_def_id_to_hir_id(item.owner_id.def_id));

                    // If there was a private module in the current path then don't bother inlining
                    // anything as it will probably be stripped anyway.
                    if is_pub && self.inside_public_path {
                        let please_inline = attrs.iter().any(|item| match item.meta_item_list() {
                            Some(ref list) if item.has_name(sym::doc) => {
                                list.iter().any(|i| i.has_name(sym::inline))
                            }
                            _ => false,
                        });
                        let ident = match kind {
                            hir::UseKind::Single(ident) => Some(renamed.unwrap_or(ident.name)),
                            hir::UseKind::Glob => None,
                            hir::UseKind::ListStem => unreachable!(),
                        };
                        if self.maybe_inline_local(item.owner_id.def_id, res, ident, please_inline)
                        {
                            debug!("Inlining {:?}", item.owner_id.def_id);
                            continue;
                        }
                    }
                    self.add_to_current_mod(item, renamed, import_id);
                }
            }
            hir::ItemKind::Macro(_, macro_def, _) => {
                // `#[macro_export] macro_rules!` items are handled separately in `visit()`,
                // above, since they need to be documented at the module top level. Accordingly,
                // we only want to handle macros if one of three conditions holds:
                //
                // 1. This macro was defined by `macro`, and thus isn't covered by the case
                //    above.
                // 2. This macro isn't marked with `#[macro_export]`, and thus isn't covered
                //    by the case above.
                // 3. We're inlining, since a reexport where inlining has been requested
                //    should be inlined even if it is also documented at the top level.

                let def_id = item.owner_id.to_def_id();
                let is_macro_2_0 = !macro_def.macro_rules;
                let nonexported = !tcx.has_attr(def_id, sym::macro_export);

                if is_macro_2_0 || nonexported || self.inlining {
                    self.add_to_current_mod(item, renamed, import_id);
                }
            }
            hir::ItemKind::Mod(_, m) => {
                self.enter_mod(item.owner_id.def_id, m, get_name(), renamed, import_id);
            }
            hir::ItemKind::Fn { .. }
            | hir::ItemKind::ExternCrate(..)
            | hir::ItemKind::Enum(..)
            | hir::ItemKind::Struct(..)
            | hir::ItemKind::Union(..)
            | hir::ItemKind::TyAlias(..)
            | hir::ItemKind::Static(..)
            | hir::ItemKind::Trait(..)
            | hir::ItemKind::TraitAlias(..) => {
                self.add_to_current_mod(item, renamed, import_id);
            }
            hir::ItemKind::Const(..) => {
                // Underscore constants do not correspond to a nameable item and
                // so are never useful in documentation.
                if get_name() != kw::Underscore {
                    self.add_to_current_mod(item, renamed, import_id);
                }
            }
            hir::ItemKind::Impl(impl_) => {
                // Don't duplicate impls when inlining or if it's implementing a trait, we'll pick
                // them up regardless of where they're located.
                if !self.inlining && impl_.of_trait.is_none() {
                    self.add_to_current_mod(item, None, None);
                }
            }
        }
    }

    fn visit_foreign_item_inner(
        &mut self,
        item: &'tcx hir::ForeignItem<'_>,
        renamed: Option<Symbol>,
    ) {
        // If inlining we only want to include public functions.
        if !self.inlining || self.cx.tcx.visibility(item.owner_id).is_public() {
            self.modules.last_mut().unwrap().foreigns.push((item, renamed));
        }
    }

    /// This method will create a new module and push it onto the "modules stack" then call
    /// `visit_mod_contents`. Once done, it'll remove it from the "modules stack" and instead
    /// add into the list of modules of the current module.
    fn enter_mod(
        &mut self,
        id: LocalDefId,
        m: &'tcx hir::Mod<'tcx>,
        name: Symbol,
        renamed: Option<Symbol>,
        import_id: Option<LocalDefId>,
    ) {
        self.modules.push(Module::new(name, id, m.spans.inner_span, renamed, import_id));

        self.visit_mod_contents(id, m);

        let last = self.modules.pop().unwrap();
        self.modules.last_mut().unwrap().mods.push(last);
    }
}

// We need to implement this visitor so it'll go everywhere and retrieve items we're interested in
// such as impl blocks in const blocks.
impl<'tcx> Visitor<'tcx> for RustdocVisitor<'_, 'tcx> {
    type NestedFilter = nested_filter::All;

    fn maybe_tcx(&mut self) -> Self::MaybeTyCtxt {
        self.cx.tcx
    }

    fn visit_item(&mut self, i: &'tcx hir::Item<'tcx>) {
        self.visit_item_inner(i, None, None);
        let new_value = self.is_importable_from_parent
            && matches!(
                i.kind,
                hir::ItemKind::Mod(..)
                    | hir::ItemKind::ForeignMod { .. }
                    | hir::ItemKind::Impl(..)
                    | hir::ItemKind::Trait(..)
            );
        let prev = mem::replace(&mut self.is_importable_from_parent, new_value);
        walk_item(self, i);
        self.is_importable_from_parent = prev;
    }

    fn visit_mod(&mut self, _: &hir::Mod<'tcx>, _: Span, _: hir::HirId) {
        // Handled in `visit_item_inner`
    }

    fn visit_use(&mut self, _: &hir::UsePath<'tcx>, _: hir::HirId) {
        // Handled in `visit_item_inner`
    }

    fn visit_path(&mut self, _: &hir::Path<'tcx>, _: hir::HirId) {
        // Handled in `visit_item_inner`
    }

    fn visit_label(&mut self, _: &rustc_ast::Label) {
        // Unneeded.
    }

    fn visit_infer(
        &mut self,
        _inf_id: hir::HirId,
        _inf_span: Span,
        _kind: hir::intravisit::InferKind<'tcx>,
    ) -> Self::Result {
        // Unneeded
    }

    fn visit_lifetime(&mut self, _: &hir::Lifetime) {
        // Unneeded.
    }

    fn visit_body(&mut self, b: &hir::Body<'tcx>) {
        let prev = mem::replace(&mut self.inside_body, true);
        walk_body(self, b);
        self.inside_body = prev;
    }
}
