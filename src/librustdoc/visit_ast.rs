// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Rust AST Visitor. Extracts useful information and massages it into a form
//! usable for clean

use std::mem;

use rustc_target::spec::abi;
use syntax::ast;
use syntax::attr;
use syntax_pos::Span;

use rustc::hir::map as hir_map;
use rustc::hir::def::Def;
use rustc::hir::def_id::{DefId, LOCAL_CRATE};
use rustc::middle::cstore::{LoadedMacro, CrateStore};
use rustc::middle::privacy::AccessLevel;
use rustc::ty::Visibility;
use rustc::util::nodemap::{FxHashSet, FxHashMap};

use rustc::hir;

use core;
use clean::{self, AttributesExt, NestedAttributesExt, def_id_to_path};
use doctree::*;

// looks to me like the first two of these are actually
// output parameters, maybe only mutated once; perhaps
// better simply to have the visit method return a tuple
// containing them?

// also, is there some reason that this doesn't use the 'visit'
// framework from syntax?

pub struct RustdocVisitor<'a, 'tcx: 'a, 'rcx: 'a> {
    pub cstore: &'a CrateStore,
    pub module: Module,
    pub attrs: hir::HirVec<ast::Attribute>,
    pub cx: &'a core::DocContext<'a, 'tcx, 'rcx>,
    view_item_stack: FxHashSet<ast::NodeId>,
    inlining: bool,
    /// Is the current module and all of its parents public?
    inside_public_path: bool,
    exact_paths: Option<FxHashMap<DefId, Vec<String>>>,
}

impl<'a, 'tcx, 'rcx> RustdocVisitor<'a, 'tcx, 'rcx> {
    pub fn new(cstore: &'a CrateStore,
               cx: &'a core::DocContext<'a, 'tcx, 'rcx>) -> RustdocVisitor<'a, 'tcx, 'rcx> {
        // If the root is re-exported, terminate all recursion.
        let mut stack = FxHashSet();
        stack.insert(ast::CRATE_NODE_ID);
        RustdocVisitor {
            module: Module::new(None),
            attrs: hir::HirVec::new(),
            cx,
            view_item_stack: stack,
            inlining: false,
            inside_public_path: true,
            exact_paths: Some(FxHashMap()),
            cstore,
        }
    }

    fn store_path(&mut self, did: DefId) {
        // We can't use the entry api, as that keeps the mutable borrow of self active
        // when we try to use cx
        let exact_paths = self.exact_paths.as_mut().unwrap();
        if exact_paths.get(&did).is_none() {
            let path = def_id_to_path(self.cx, did, self.cx.crate_name.clone());
            exact_paths.insert(did, path);
        }
    }

    fn stability(&self, id: ast::NodeId) -> Option<attr::Stability> {
        self.cx.tcx.hir.opt_local_def_id(id)
            .and_then(|def_id| self.cx.tcx.lookup_stability(def_id)).cloned()
    }

    fn deprecation(&self, id: ast::NodeId) -> Option<attr::Deprecation> {
        self.cx.tcx.hir.opt_local_def_id(id)
            .and_then(|def_id| self.cx.tcx.lookup_deprecation(def_id))
    }

    pub fn visit(&mut self, krate: &hir::Crate) {
        self.attrs = krate.attrs.clone();

        self.module = self.visit_mod_contents(krate.span,
                                              krate.attrs.clone(),
                                              hir::Public,
                                              ast::CRATE_NODE_ID,
                                              &krate.module,
                                              None);
        // attach the crate's exported macros to the top-level module:
        let macro_exports: Vec<_> =
            krate.exported_macros.iter().map(|def| self.visit_local_macro(def)).collect();
        self.module.macros.extend(macro_exports);
        self.module.is_crate = true;

        self.cx.renderinfo.borrow_mut().exact_paths = self.exact_paths.take().unwrap();
    }

    pub fn visit_variant_data(&mut self, item: &hir::Item,
                            name: ast::Name, sd: &hir::VariantData,
                            generics: &hir::Generics) -> Struct {
        debug!("Visiting struct");
        let struct_type = struct_type_from_def(&*sd);
        Struct {
            id: item.id,
            struct_type,
            name,
            vis: item.vis.clone(),
            stab: self.stability(item.id),
            depr: self.deprecation(item.id),
            attrs: item.attrs.clone(),
            generics: generics.clone(),
            fields: sd.fields().iter().cloned().collect(),
            whence: item.span
        }
    }

    pub fn visit_union_data(&mut self, item: &hir::Item,
                            name: ast::Name, sd: &hir::VariantData,
                            generics: &hir::Generics) -> Union {
        debug!("Visiting union");
        let struct_type = struct_type_from_def(&*sd);
        Union {
            id: item.id,
            struct_type,
            name,
            vis: item.vis.clone(),
            stab: self.stability(item.id),
            depr: self.deprecation(item.id),
            attrs: item.attrs.clone(),
            generics: generics.clone(),
            fields: sd.fields().iter().cloned().collect(),
            whence: item.span
        }
    }

    pub fn visit_enum_def(&mut self, it: &hir::Item,
                          name: ast::Name, def: &hir::EnumDef,
                          params: &hir::Generics) -> Enum {
        debug!("Visiting enum");
        Enum {
            name,
            variants: def.variants.iter().map(|v| Variant {
                name: v.node.name,
                attrs: v.node.attrs.clone(),
                stab: self.stability(v.node.data.id()),
                depr: self.deprecation(v.node.data.id()),
                def: v.node.data.clone(),
                whence: v.span,
            }).collect(),
            vis: it.vis.clone(),
            stab: self.stability(it.id),
            depr: self.deprecation(it.id),
            generics: params.clone(),
            attrs: it.attrs.clone(),
            id: it.id,
            whence: it.span,
        }
    }

    pub fn visit_fn(&mut self, item: &hir::Item,
                    name: ast::Name, fd: &hir::FnDecl,
                    unsafety: &hir::Unsafety,
                    constness: hir::Constness,
                    abi: &abi::Abi,
                    gen: &hir::Generics,
                    body: hir::BodyId) -> Function {
        debug!("Visiting fn");
        Function {
            id: item.id,
            vis: item.vis.clone(),
            stab: self.stability(item.id),
            depr: self.deprecation(item.id),
            attrs: item.attrs.clone(),
            decl: fd.clone(),
            name,
            whence: item.span,
            generics: gen.clone(),
            unsafety: *unsafety,
            constness,
            abi: *abi,
            body,
        }
    }

    pub fn visit_mod_contents(&mut self, span: Span, attrs: hir::HirVec<ast::Attribute>,
                              vis: hir::Visibility, id: ast::NodeId,
                              m: &hir::Mod,
                              name: Option<ast::Name>) -> Module {
        let mut om = Module::new(name);
        om.where_outer = span;
        om.where_inner = m.inner;
        om.attrs = attrs;
        om.vis = vis.clone();
        om.stab = self.stability(id);
        om.depr = self.deprecation(id);
        om.id = id;
        // Keep track of if there were any private modules in the path.
        let orig_inside_public_path = self.inside_public_path;
        self.inside_public_path &= vis == hir::Public;
        for i in &m.item_ids {
            let item = self.cx.tcx.hir.expect_item(i.id);
            self.visit_item(item, None, &mut om);
        }
        self.inside_public_path = orig_inside_public_path;
        let def_id = self.cx.tcx.hir.local_def_id(id);
        if let Some(exports) = self.cx.tcx.module_exports(def_id) {
            for export in exports.iter().filter(|e| e.vis == Visibility::Public) {
                if let Def::Macro(def_id, ..) = export.def {
                    // FIXME(50647): this eager macro inlining does not take
                    // doc(hidden)/doc(no_inline) into account
                    if def_id.krate == LOCAL_CRATE {
                        continue // These are `krate.exported_macros`, handled in `self.visit()`.
                    }

                    let imported_from = self.cx.tcx.original_crate_name(def_id.krate);
                    let def = match self.cstore.load_macro_untracked(def_id, self.cx.sess()) {
                        LoadedMacro::MacroDef(macro_def) => macro_def,
                        // FIXME(jseyfried): document proc macro re-exports
                        LoadedMacro::ProcMacro(..) => continue,
                    };

                    let matchers = if let ast::ItemKind::MacroDef(ref def) = def.node {
                        let tts: Vec<_> = def.stream().into_trees().collect();
                        tts.chunks(4).map(|arm| arm[0].span()).collect()
                    } else {
                        unreachable!()
                    };

                    debug!("inlining macro {}", def.ident.name);
                    om.macros.push(Macro {
                        def_id,
                        attrs: def.attrs.clone().into(),
                        name: def.ident.name,
                        whence: def.span,
                        matchers,
                        stab: self.stability(def.id),
                        depr: self.deprecation(def.id),
                        imported_from: Some(imported_from),
                    })
                }
            }
        }
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
    /// Returns true if the target has been inlined.
    fn maybe_inline_local(&mut self,
                          id: ast::NodeId,
                          def: Def,
                          renamed: Option<ast::Name>,
                          glob: bool,
                          om: &mut Module,
                          please_inline: bool) -> bool {

        fn inherits_doc_hidden(cx: &core::DocContext, mut node: ast::NodeId) -> bool {
            while let Some(id) = cx.tcx.hir.get_enclosing_scope(node) {
                node = id;
                if cx.tcx.hir.attrs(node).lists("doc").has_word("hidden") {
                    return true;
                }
                if node == ast::CRATE_NODE_ID {
                    break;
                }
            }
            false
        }

        debug!("maybe_inline_local def: {:?}", def);

        let tcx = self.cx.tcx;
        if def == Def::Err {
            return false;
        }
        let def_did = def.def_id();

        let use_attrs = tcx.hir.attrs(id);
        // Don't inline doc(hidden) imports so they can be stripped at a later stage.
        let is_no_inline = use_attrs.lists("doc").has_word("no_inline") ||
                           use_attrs.lists("doc").has_word("hidden");

        // For cross-crate impl inlining we need to know whether items are
        // reachable in documentation - a previously nonreachable item can be
        // made reachable by cross-crate inlining which we're checking here.
        // (this is done here because we need to know this upfront)
        if !def_did.is_local() && !is_no_inline {
            let attrs = clean::inline::load_attrs(self.cx, def_did);
            let self_is_hidden = attrs.lists("doc").has_word("hidden");
            match def {
                Def::Trait(did) |
                Def::Struct(did) |
                Def::Union(did) |
                Def::Enum(did) |
                Def::TyForeign(did) |
                Def::TyAlias(did) if !self_is_hidden => {
                    self.cx.access_levels.borrow_mut().map.insert(did, AccessLevel::Public);
                },
                Def::Mod(did) => if !self_is_hidden {
                    ::visit_lib::LibEmbargoVisitor::new(self.cx).visit_mod(did);
                },
                _ => {},
            }

            return false
        }

        let def_node_id = match tcx.hir.as_local_node_id(def_did) {
            Some(n) => n, None => return false
        };

        let is_private = !self.cx.access_levels.borrow().is_public(def_did);
        let is_hidden = inherits_doc_hidden(self.cx, def_node_id);

        // Only inline if requested or if the item would otherwise be stripped
        if (!please_inline && !is_private && !is_hidden) || is_no_inline {
            return false
        }

        if !self.view_item_stack.insert(def_node_id) { return false }

        let ret = match tcx.hir.get(def_node_id) {
            hir_map::NodeItem(&hir::Item { node: hir::ItemMod(ref m), .. }) if glob => {
                let prev = mem::replace(&mut self.inlining, true);
                for i in &m.item_ids {
                    let i = self.cx.tcx.hir.expect_item(i.id);
                    self.visit_item(i, None, om);
                }
                self.inlining = prev;
                true
            }
            hir_map::NodeItem(it) if !glob => {
                let prev = mem::replace(&mut self.inlining, true);
                self.visit_item(it, renamed, om);
                self.inlining = prev;
                true
            }
            hir_map::NodeForeignItem(it) if !glob => {
                // generate a fresh `extern {}` block if we want to inline a foreign item.
                om.foreigns.push(hir::ForeignMod {
                    abi: tcx.hir.get_foreign_abi(it.id),
                    items: vec![hir::ForeignItem {
                        name: renamed.unwrap_or(it.name),
                        .. it.clone()
                    }].into(),
                });
                true
            }
            _ => false,
        };
        self.view_item_stack.remove(&def_node_id);
        ret
    }

    pub fn visit_item(&mut self, item: &hir::Item,
                      renamed: Option<ast::Name>, om: &mut Module) {
        debug!("Visiting item {:?}", item);
        let name = renamed.unwrap_or(item.name);

        if item.vis == hir::Public {
            let def_id = self.cx.tcx.hir.local_def_id(item.id);
            self.store_path(def_id);
        }

        match item.node {
            hir::ItemForeignMod(ref fm) => {
                // If inlining we only want to include public functions.
                om.foreigns.push(if self.inlining {
                    hir::ForeignMod {
                        abi: fm.abi,
                        items: fm.items.iter().filter(|i| i.vis == hir::Public).cloned().collect(),
                    }
                } else {
                    fm.clone()
                });
            }
            // If we're inlining, skip private items.
            _ if self.inlining && item.vis != hir::Public => {}
            hir::ItemGlobalAsm(..) => {}
            hir::ItemExternCrate(orig_name) => {
                let def_id = self.cx.tcx.hir.local_def_id(item.id);
                om.extern_crates.push(ExternCrate {
                    cnum: self.cx.tcx.extern_mod_stmt_cnum(def_id)
                                .unwrap_or(LOCAL_CRATE),
                    name,
                    path: orig_name.map(|x|x.to_string()),
                    vis: item.vis.clone(),
                    attrs: item.attrs.clone(),
                    whence: item.span,
                })
            }
            hir::ItemUse(_, hir::UseKind::ListStem) => {}
            hir::ItemUse(ref path, kind) => {
                let is_glob = kind == hir::UseKind::Glob;

                // If there was a private module in the current path then don't bother inlining
                // anything as it will probably be stripped anyway.
                if item.vis == hir::Public && self.inside_public_path {
                    let please_inline = item.attrs.iter().any(|item| {
                        match item.meta_item_list() {
                            Some(ref list) if item.check_name("doc") => {
                                list.iter().any(|i| i.check_name("inline"))
                            }
                            _ => false,
                        }
                    });
                    let name = if is_glob { None } else { Some(name) };
                    if self.maybe_inline_local(item.id,
                                               path.def,
                                               name,
                                               is_glob,
                                               om,
                                               please_inline) {
                        return;
                    }
                }

                om.imports.push(Import {
                    name,
                    id: item.id,
                    vis: item.vis.clone(),
                    attrs: item.attrs.clone(),
                    path: (**path).clone(),
                    glob: is_glob,
                    whence: item.span,
                });
            }
            hir::ItemMod(ref m) => {
                om.mods.push(self.visit_mod_contents(item.span,
                                                     item.attrs.clone(),
                                                     item.vis.clone(),
                                                     item.id,
                                                     m,
                                                     Some(name)));
            },
            hir::ItemEnum(ref ed, ref gen) =>
                om.enums.push(self.visit_enum_def(item, name, ed, gen)),
            hir::ItemStruct(ref sd, ref gen) =>
                om.structs.push(self.visit_variant_data(item, name, sd, gen)),
            hir::ItemUnion(ref sd, ref gen) =>
                om.unions.push(self.visit_union_data(item, name, sd, gen)),
            hir::ItemFn(ref fd, ref unsafety, constness, ref abi, ref gen, body) =>
                om.fns.push(self.visit_fn(item, name, &**fd, unsafety,
                                          constness, abi, gen, body)),
            hir::ItemTy(ref ty, ref gen) => {
                let t = Typedef {
                    ty: ty.clone(),
                    gen: gen.clone(),
                    name,
                    id: item.id,
                    attrs: item.attrs.clone(),
                    whence: item.span,
                    vis: item.vis.clone(),
                    stab: self.stability(item.id),
                    depr: self.deprecation(item.id),
                };
                om.typedefs.push(t);
            },
            hir::ItemStatic(ref ty, ref mut_, ref exp) => {
                let s = Static {
                    type_: ty.clone(),
                    mutability: mut_.clone(),
                    expr: exp.clone(),
                    id: item.id,
                    name,
                    attrs: item.attrs.clone(),
                    whence: item.span,
                    vis: item.vis.clone(),
                    stab: self.stability(item.id),
                    depr: self.deprecation(item.id),
                };
                om.statics.push(s);
            },
            hir::ItemConst(ref ty, ref exp) => {
                let s = Constant {
                    type_: ty.clone(),
                    expr: exp.clone(),
                    id: item.id,
                    name,
                    attrs: item.attrs.clone(),
                    whence: item.span,
                    vis: item.vis.clone(),
                    stab: self.stability(item.id),
                    depr: self.deprecation(item.id),
                };
                om.constants.push(s);
            },
            hir::ItemTrait(is_auto, unsafety, ref gen, ref b, ref item_ids) => {
                let items = item_ids.iter()
                                    .map(|ti| self.cx.tcx.hir.trait_item(ti.id).clone())
                                    .collect();
                let t = Trait {
                    is_auto,
                    unsafety,
                    name,
                    items,
                    generics: gen.clone(),
                    bounds: b.iter().cloned().collect(),
                    id: item.id,
                    attrs: item.attrs.clone(),
                    whence: item.span,
                    vis: item.vis.clone(),
                    stab: self.stability(item.id),
                    depr: self.deprecation(item.id),
                };
                om.traits.push(t);
            },
            hir::ItemTraitAlias(..) => {
                unimplemented!("trait objects are not yet implemented")
            },

            hir::ItemImpl(unsafety,
                          polarity,
                          defaultness,
                          ref gen,
                          ref tr,
                          ref ty,
                          ref item_ids) => {
                // Don't duplicate impls when inlining, we'll pick them up
                // regardless of where they're located.
                if !self.inlining {
                    let items = item_ids.iter()
                                        .map(|ii| self.cx.tcx.hir.impl_item(ii.id).clone())
                                        .collect();
                    let i = Impl {
                        unsafety,
                        polarity,
                        defaultness,
                        generics: gen.clone(),
                        trait_: tr.clone(),
                        for_: ty.clone(),
                        items,
                        attrs: item.attrs.clone(),
                        id: item.id,
                        whence: item.span,
                        vis: item.vis.clone(),
                        stab: self.stability(item.id),
                        depr: self.deprecation(item.id),
                    };
                    om.impls.push(i);
                }
            },
        }
    }

    // convert each exported_macro into a doc item
    fn visit_local_macro(&self, def: &hir::MacroDef) -> Macro {
        debug!("visit_local_macro: {}", def.name);
        let tts = def.body.trees().collect::<Vec<_>>();
        // Extract the spans of all matchers. They represent the "interface" of the macro.
        let matchers = tts.chunks(4).map(|arm| arm[0].span()).collect();

        Macro {
            def_id: self.cx.tcx.hir.local_def_id(def.id),
            attrs: def.attrs.clone(),
            name: def.name,
            whence: def.span,
            matchers,
            stab: self.stability(def.id),
            depr: self.deprecation(def.id),
            imported_from: None,
        }
    }
}
