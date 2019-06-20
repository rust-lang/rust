//! The Rust AST Visitor. Extracts useful information and massages it into a form
//! usable for `clean`.

use rustc::hir::{self, Node};
use rustc::hir::def::{Res, DefKind};
use rustc::hir::def_id::{DefId, LOCAL_CRATE};
use rustc::middle::privacy::AccessLevel;
use rustc::util::nodemap::{FxHashSet, FxHashMap};
use syntax::ast;
use syntax::attr;
use syntax::ext::base::MacroKind;
use syntax::source_map::Spanned;
use syntax::symbol::sym;
use syntax_pos::{self, Span};

use std::mem;

use crate::core;
use crate::clean::{self, AttributesExt, NestedAttributesExt, def_id_to_path};
use crate::doctree::*;


// Looks to me like the first two of these are actually
// output parameters, maybe only mutated once; perhaps
// better simply to have the visit method return a tuple
// containing them?

// Also, is there some reason that this doesn't use the 'visit'
// framework from syntax?.

pub struct RustdocVisitor<'a, 'tcx> {
    pub module: Option<Module<'tcx>>,
    pub cx: &'a core::DocContext<'tcx>,
    view_item_stack: FxHashSet<hir::HirId>,
    inlining: bool,
    /// Are the current module and all of its parents public?
    inside_public_path: bool,
    exact_paths: Option<FxHashMap<DefId, Vec<String>>>,
}

impl<'a, 'tcx> RustdocVisitor<'a, 'tcx> {
    pub fn new(
        cx: &'a core::DocContext<'tcx>
    ) -> RustdocVisitor<'a, 'tcx> {
        // If the root is re-exported, terminate all recursion.
        let mut stack = FxHashSet::default();
        stack.insert(hir::CRATE_HIR_ID);
        RustdocVisitor {
            module: None,
            cx,
            view_item_stack: stack,
            inlining: false,
            inside_public_path: true,
            exact_paths: Some(FxHashMap::default()),
        }
    }

    fn store_path(&mut self, did: DefId) {
        // We can't use the entry API, as that keeps the mutable borrow of `self` active
        // when we try to use `cx`.
        let exact_paths = self.exact_paths.as_mut().unwrap();
        if exact_paths.get(&did).is_none() {
            let path = def_id_to_path(self.cx, did, self.cx.crate_name.clone());
            exact_paths.insert(did, path);
        }
    }

    fn stability(&self, id: hir::HirId) -> Option<attr::Stability> {
        self.cx.tcx.hir().opt_local_def_id_from_hir_id(id)
            .and_then(|def_id| self.cx.tcx.lookup_stability(def_id)).cloned()
    }

    fn deprecation(&self, id: hir::HirId) -> Option<attr::Deprecation> {
        self.cx.tcx.hir().opt_local_def_id_from_hir_id(id)
            .and_then(|def_id| self.cx.tcx.lookup_deprecation(def_id))
    }

    pub fn visit(&mut self, krate: &'tcx hir::Crate) {
        let mut module = self.visit_mod_contents(krate.span,
                                              &krate.attrs,
                                              &Spanned { span: syntax_pos::DUMMY_SP,
                                                        node: hir::VisibilityKind::Public },
                                              hir::CRATE_HIR_ID,
                                              &krate.module,
                                              None);
        // Attach the crate's exported macros to the top-level module:
        module.macros.extend(
            krate.exported_macros.iter().map(|def| self.visit_local_macro(def, None)),
        );
        module.is_crate = true;
        self.module = Some(module);

        self.cx.renderinfo.borrow_mut().exact_paths = self.exact_paths.take().unwrap();
    }

    pub fn visit_variant_data(&mut self, item: &'tcx hir::Item,
                              name: ast::Name, sd: &'tcx hir::VariantData,
                              generics: &'tcx hir::Generics) -> Struct<'tcx> {
        debug!("Visiting struct");
        let struct_type = struct_type_from_def(&*sd);
        Struct {
            id: item.hir_id,
            struct_type,
            name,
            vis: &item.vis,
            stab: self.stability(item.hir_id),
            depr: self.deprecation(item.hir_id),
            attrs: &item.attrs,
            generics,
            fields: sd.fields(),
            whence: item.span
        }
    }

    pub fn visit_union_data(&mut self, item: &'tcx hir::Item,
                            name: ast::Name, sd: &'tcx hir::VariantData,
                            generics: &'tcx hir::Generics) -> Union<'tcx> {
        debug!("Visiting union");
        let struct_type = struct_type_from_def(&*sd);
        Union {
            id: item.hir_id,
            struct_type,
            name,
            vis: &item.vis,
            stab: self.stability(item.hir_id),
            depr: self.deprecation(item.hir_id),
            attrs: &item.attrs,
            generics,
            fields: sd.fields(),
            whence: item.span
        }
    }

    pub fn visit_enum_def(&mut self, it: &'tcx hir::Item,
                          name: ast::Name, def: &'tcx hir::EnumDef,
                          generics: &'tcx hir::Generics) -> Enum<'tcx> {
        debug!("Visiting enum");
        Enum {
            name,
            variants: def.variants.iter().map(|v| Variant {
                name: v.node.ident.name,
                id: v.node.id,
                attrs: &v.node.attrs,
                stab: self.stability(v.node.id),
                depr: self.deprecation(v.node.id),
                def: &v.node.data,
                whence: v.span,
            }).collect(),
            vis: &it.vis,
            stab: self.stability(it.hir_id),
            depr: self.deprecation(it.hir_id),
            generics,
            attrs: &it.attrs,
            id: it.hir_id,
            whence: it.span,
        }
    }

    pub fn visit_fn(&mut self, om: &mut Module<'tcx>, item: &'tcx hir::Item,
                    name: ast::Name, decl: &'tcx hir::FnDecl,
                    header: hir::FnHeader,
                    generics: &'tcx hir::Generics,
                    body: hir::BodyId) {
        debug!("Visiting fn");
        let macro_kind = item.attrs.iter().filter_map(|a| {
            if a.check_name(sym::proc_macro) {
                Some(MacroKind::Bang)
            } else if a.check_name(sym::proc_macro_derive) {
                Some(MacroKind::Derive)
            } else if a.check_name(sym::proc_macro_attribute) {
                Some(MacroKind::Attr)
            } else {
                None
            }
        }).next();
        match macro_kind {
            Some(kind) => {
                let name = if kind == MacroKind::Derive {
                    item.attrs.lists(sym::proc_macro_derive)
                              .filter_map(|mi| mi.ident())
                              .next()
                              .expect("proc-macro derives require a name")
                              .name
                } else {
                    name
                };

                let mut helpers = Vec::new();
                for mi in item.attrs.lists(sym::proc_macro_derive) {
                    if !mi.check_name(sym::attributes) {
                        continue;
                    }

                    if let Some(list) = mi.meta_item_list() {
                        for inner_mi in list {
                            if let Some(ident) = inner_mi.ident() {
                                helpers.push(ident.name);
                            }
                        }
                    }
                }

                om.proc_macros.push(ProcMacro {
                    name,
                    id: item.hir_id,
                    kind,
                    helpers,
                    attrs: &item.attrs,
                    whence: item.span,
                    stab: self.stability(item.hir_id),
                    depr: self.deprecation(item.hir_id),
                });
            }
            None => {
                om.fns.push(Function {
                    id: item.hir_id,
                    vis: &item.vis,
                    stab: self.stability(item.hir_id),
                    depr: self.deprecation(item.hir_id),
                    attrs: &item.attrs,
                    decl,
                    name,
                    whence: item.span,
                    generics,
                    header,
                    body,
                });
            }
        }
    }

    pub fn visit_mod_contents(&mut self, span: Span, attrs: &'tcx hir::HirVec<ast::Attribute>,
                              vis: &'tcx hir::Visibility, id: hir::HirId,
                              m: &'tcx hir::Mod,
                              name: Option<ast::Name>) -> Module<'tcx> {
        let mut om = Module::new(name, attrs, vis);
        om.where_outer = span;
        om.where_inner = m.inner;
        om.stab = self.stability(id);
        om.depr = self.deprecation(id);
        om.id = self.cx.tcx.hir().hir_to_node_id(id);
        // Keep track of if there were any private modules in the path.
        let orig_inside_public_path = self.inside_public_path;
        self.inside_public_path &= vis.node.is_pub();
        for i in &m.item_ids {
            let item = self.cx.tcx.hir().expect_item(i.id);
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
    fn maybe_inline_local(&mut self,
                          id: hir::HirId,
                          res: Res,
                          renamed: Option<ast::Ident>,
                          glob: bool,
                          om: &mut Module<'tcx>,
                          please_inline: bool) -> bool {

        fn inherits_doc_hidden(cx: &core::DocContext<'_>, mut node: hir::HirId) -> bool {
            while let Some(id) = cx.tcx.hir().get_enclosing_scope(node) {
                node = id;
                if cx.tcx.hir().attrs(node)
                    .lists(sym::doc).has_word(sym::hidden) {
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
        let is_no_inline = use_attrs.lists(sym::doc).has_word(sym::no_inline) ||
                           use_attrs.lists(sym::doc).has_word(sym::hidden);

        // For cross-crate impl inlining we need to know whether items are
        // reachable in documentation -- a previously nonreachable item can be
        // made reachable by cross-crate inlining which we're checking here.
        // (this is done here because we need to know this upfront).
        if !res_did.is_local() && !is_no_inline {
            let attrs = clean::inline::load_attrs(self.cx, res_did);
            let self_is_hidden = attrs.lists(sym::doc).has_word(sym::hidden);
            match res {
                Res::Def(DefKind::Trait, did) |
                Res::Def(DefKind::Struct, did) |
                Res::Def(DefKind::Union, did) |
                Res::Def(DefKind::Enum, did) |
                Res::Def(DefKind::ForeignTy, did) |
                Res::Def(DefKind::TyAlias, did) if !self_is_hidden => {
                    self.cx.renderinfo
                        .borrow_mut()
                        .access_levels.map
                        .insert(did, AccessLevel::Public);
                },
                Res::Def(DefKind::Mod, did) => if !self_is_hidden {
                    crate::visit_lib::LibEmbargoVisitor::new(self.cx).visit_mod(did);
                },
                _ => {},
            }

            return false
        }

        let res_hir_id = match tcx.hir().as_local_hir_id(res_did) {
            Some(n) => n, None => return false
        };

        let is_private = !self.cx.renderinfo.borrow().access_levels.is_public(res_did);
        let is_hidden = inherits_doc_hidden(self.cx, res_hir_id);

        // Only inline if requested or if the item would otherwise be stripped.
        if (!please_inline && !is_private && !is_hidden) || is_no_inline {
            return false
        }

        if !self.view_item_stack.insert(res_hir_id) { return false }

        let ret = match tcx.hir().get(res_hir_id) {
            Node::Item(&hir::Item { node: hir::ItemKind::Mod(ref m), .. }) if glob => {
                let prev = mem::replace(&mut self.inlining, true);
                for i in &m.item_ids {
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

    pub fn visit_item(&mut self, item: &'tcx hir::Item,
                      renamed: Option<ast::Ident>, om: &mut Module<'tcx>) {
        debug!("Visiting item {:?}", item);
        let ident = renamed.unwrap_or(item.ident);

        if item.vis.node.is_pub() {
            let def_id = self.cx.tcx.hir().local_def_id_from_hir_id(item.hir_id);
            self.store_path(def_id);
        }

        match item.node {
            hir::ItemKind::ForeignMod(ref fm) => {
                for item in &fm.items {
                    self.visit_foreign_item(item, None, om);
                }
            }
            // If we're inlining, skip private items.
            _ if self.inlining && !item.vis.node.is_pub() => {}
            hir::ItemKind::GlobalAsm(..) => {}
            hir::ItemKind::ExternCrate(orig_name) => {
                let def_id = self.cx.tcx.hir().local_def_id_from_hir_id(item.hir_id);
                om.extern_crates.push(ExternCrate {
                    cnum: self.cx.tcx.extern_mod_stmt_cnum(def_id)
                                .unwrap_or(LOCAL_CRATE),
                    name: ident.name,
                    path: orig_name.map(|x|x.to_string()),
                    vis: &item.vis,
                    attrs: &item.attrs,
                    whence: item.span,
                })
            }
            hir::ItemKind::Use(_, hir::UseKind::ListStem) => {}
            hir::ItemKind::Use(ref path, kind) => {
                let is_glob = kind == hir::UseKind::Glob;

                // Struct and variant constructors and proc macro stubs always show up alongside
                // their definitions, we've already processed them so just discard these.
                match path.res {
                    Res::Def(DefKind::Ctor(..), _)
                    | Res::SelfCtor(..)
                    | Res::Def(DefKind::Macro(MacroKind::ProcMacroStub), _) => return,
                    _ => {}
                }

                // If there was a private module in the current path then don't bother inlining
                // anything as it will probably be stripped anyway.
                if item.vis.node.is_pub() && self.inside_public_path {
                    let please_inline = item.attrs.iter().any(|item| {
                        match item.meta_item_list() {
                            Some(ref list) if item.check_name(sym::doc) => {
                                list.iter().any(|i| i.check_name(sym::inline))
                            }
                            _ => false,
                        }
                    });
                    let ident = if is_glob { None } else { Some(ident) };
                    if self.maybe_inline_local(item.hir_id,
                                               path.res,
                                               ident,
                                               is_glob,
                                               om,
                                               please_inline) {
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
                    whence: item.span,
                });
            }
            hir::ItemKind::Mod(ref m) => {
                om.mods.push(self.visit_mod_contents(item.span,
                                                     &item.attrs,
                                                     &item.vis,
                                                     item.hir_id,
                                                     m,
                                                     Some(ident.name)));
            },
            hir::ItemKind::Enum(ref ed, ref gen) =>
                om.enums.push(self.visit_enum_def(item, ident.name, ed, gen)),
            hir::ItemKind::Struct(ref sd, ref gen) =>
                om.structs.push(self.visit_variant_data(item, ident.name, sd, gen)),
            hir::ItemKind::Union(ref sd, ref gen) =>
                om.unions.push(self.visit_union_data(item, ident.name, sd, gen)),
            hir::ItemKind::Fn(ref fd, header, ref gen, body) =>
                self.visit_fn(om, item, ident.name, &**fd, header, gen, body),
            hir::ItemKind::Ty(ref ty, ref gen) => {
                let t = Typedef {
                    ty,
                    gen,
                    name: ident.name,
                    id: item.hir_id,
                    attrs: &item.attrs,
                    whence: item.span,
                    vis: &item.vis,
                    stab: self.stability(item.hir_id),
                    depr: self.deprecation(item.hir_id),
                };
                om.typedefs.push(t);
            },
            hir::ItemKind::Existential(ref exist_ty) => {
                let t = Existential {
                    exist_ty,
                    name: ident.name,
                    id: item.hir_id,
                    attrs: &item.attrs,
                    whence: item.span,
                    vis: &item.vis,
                    stab: self.stability(item.hir_id),
                    depr: self.deprecation(item.hir_id),
                };
                om.existentials.push(t);
            },
            hir::ItemKind::Static(ref type_, mutability, expr) => {
                let s = Static {
                    type_,
                    mutability,
                    expr,
                    id: item.hir_id,
                    name: ident.name,
                    attrs: &item.attrs,
                    whence: item.span,
                    vis: &item.vis,
                    stab: self.stability(item.hir_id),
                    depr: self.deprecation(item.hir_id),
                };
                om.statics.push(s);
            },
            hir::ItemKind::Const(ref type_, expr) => {
                let s = Constant {
                    type_,
                    expr,
                    id: item.hir_id,
                    name: ident.name,
                    attrs: &item.attrs,
                    whence: item.span,
                    vis: &item.vis,
                    stab: self.stability(item.hir_id),
                    depr: self.deprecation(item.hir_id),
                };
                om.constants.push(s);
            },
            hir::ItemKind::Trait(is_auto, unsafety, ref generics, ref bounds, ref item_ids) => {
                let items = item_ids.iter()
                                    .map(|ti| self.cx.tcx.hir().trait_item(ti.id))
                                    .collect();
                let t = Trait {
                    is_auto,
                    unsafety,
                    name: ident.name,
                    items,
                    generics,
                    bounds,
                    id: item.hir_id,
                    attrs: &item.attrs,
                    whence: item.span,
                    vis: &item.vis,
                    stab: self.stability(item.hir_id),
                    depr: self.deprecation(item.hir_id),
                };
                om.traits.push(t);
            },
            hir::ItemKind::TraitAlias(ref generics, ref bounds) => {
                let t = TraitAlias {
                    name: ident.name,
                    generics,
                    bounds,
                    id: item.hir_id,
                    attrs: &item.attrs,
                    whence: item.span,
                    vis: &item.vis,
                    stab: self.stability(item.hir_id),
                    depr: self.deprecation(item.hir_id),
                };
                om.trait_aliases.push(t);
            },

            hir::ItemKind::Impl(unsafety,
                          polarity,
                          defaultness,
                          ref generics,
                          ref trait_,
                          ref for_,
                          ref item_ids) => {
                // Don't duplicate impls when inlining or if it's implementing a trait, we'll pick
                // them up regardless of where they're located.
                if !self.inlining && trait_.is_none() {
                    let items = item_ids.iter()
                                        .map(|ii| self.cx.tcx.hir().impl_item(ii.id))
                                        .collect();
                    let i = Impl {
                        unsafety,
                        polarity,
                        defaultness,
                        generics,
                        trait_,
                        for_,
                        items,
                        attrs: &item.attrs,
                        id: item.hir_id,
                        whence: item.span,
                        vis: &item.vis,
                        stab: self.stability(item.hir_id),
                        depr: self.deprecation(item.hir_id),
                    };
                    om.impls.push(i);
                }
            },
        }
    }

    fn visit_foreign_item(&mut self, item: &'tcx hir::ForeignItem,
                      renamed: Option<ast::Ident>, om: &mut Module<'tcx>) {
        // If inlining we only want to include public functions.
        if self.inlining && !item.vis.node.is_pub() {
            return;
        }

        om.foreigns.push(ForeignItem {
            id: item.hir_id,
            name: renamed.unwrap_or(item.ident).name,
            kind: &item.node,
            vis: &item.vis,
            stab: self.stability(item.hir_id),
            depr: self.deprecation(item.hir_id),
            attrs: &item.attrs,
            whence: item.span
        });
    }

    // Convert each `exported_macro` into a doc item.
    fn visit_local_macro(
        &self,
        def: &'tcx hir::MacroDef,
        renamed: Option<ast::Name>
    ) -> Macro<'tcx> {
        debug!("visit_local_macro: {}", def.name);
        let tts = def.body.trees().collect::<Vec<_>>();
        // Extract the spans of all matchers. They represent the "interface" of the macro.
        let matchers = tts.chunks(4).map(|arm| arm[0].span()).collect();

        Macro {

            def_id: self.cx.tcx.hir().local_def_id_from_hir_id(def.hir_id),
            attrs: &def.attrs,
            name: renamed.unwrap_or(def.name),
            whence: def.span,
            matchers,
            stab: self.stability(def.hir_id),
            depr: self.deprecation(def.hir_id),
            imported_from: None,
        }
    }
}
