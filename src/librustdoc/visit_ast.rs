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

use std::vec_ng::Vec;
use syntax::abi::AbiSet;
use syntax::ast;
use syntax::ast_util;
use syntax::ast_map;
use syntax::codemap::Span;

use core;
use doctree::*;

pub struct RustdocVisitor<'a> {
    module: Module,
    attrs: ~[ast::Attribute],
    cx: &'a core::DocContext,
    analysis: Option<&'a core::CrateAnalysis>,
}

impl<'a> RustdocVisitor<'a> {
    pub fn new<'b>(cx: &'b core::DocContext,
                   analysis: Option<&'b core::CrateAnalysis>) -> RustdocVisitor<'b> {
        RustdocVisitor {
            module: Module::new(None),
            attrs: ~[],
            cx: cx,
            analysis: analysis,
        }
    }

    pub fn visit(&mut self, krate: &ast::Crate) {
        self.attrs = krate.attrs.iter().map(|x| (*x).clone()).collect();

        self.module = self.visit_mod_contents(krate.span,
                                              krate.attrs
                                                   .iter()
                                                   .map(|x| *x)
                                                   .collect(),
                                              ast::Public,
                                              ast::CRATE_NODE_ID,
                                              &krate.module,
                                              None);
        self.module.is_crate = true;
    }

    pub fn visit_struct_def(&mut self, item: &ast::Item, sd: @ast::StructDef,
                            generics: &ast::Generics) -> Struct {
        debug!("Visiting struct");
        let struct_type = struct_type_from_def(sd);
        Struct {
            id: item.id,
            struct_type: struct_type,
            name: item.ident,
            vis: item.vis,
            attrs: item.attrs.iter().map(|x| *x).collect(),
            generics: generics.clone(),
            fields: sd.fields.iter().map(|x| (*x).clone()).collect(),
            where: item.span
        }
    }

    pub fn visit_enum_def(&mut self, it: &ast::Item, def: &ast::EnumDef,
                          params: &ast::Generics) -> Enum {
        debug!("Visiting enum");
        let mut vars: ~[Variant] = ~[];
        for x in def.variants.iter() {
            vars.push(Variant {
                name: x.node.name,
                attrs: x.node.attrs.iter().map(|x| *x).collect(),
                vis: x.node.vis,
                id: x.node.id,
                kind: x.node.kind.clone(),
                where: x.span,
            });
        }
        Enum {
            name: it.ident,
            variants: vars,
            vis: it.vis,
            generics: params.clone(),
            attrs: it.attrs.iter().map(|x| *x).collect(),
            id: it.id,
            where: it.span,
        }
    }

    pub fn visit_fn(&mut self, item: &ast::Item, fd: &ast::FnDecl,
                    purity: &ast::Purity, _abi: &AbiSet,
                    gen: &ast::Generics) -> Function {
        debug!("Visiting fn");
        Function {
            id: item.id,
            vis: item.vis,
            attrs: item.attrs.iter().map(|x| *x).collect(),
            decl: fd.clone(),
            name: item.ident,
            where: item.span,
            generics: gen.clone(),
            purity: *purity,
        }
    }

    pub fn visit_mod_contents(&mut self, span: Span, attrs: ~[ast::Attribute],
                              vis: ast::Visibility, id: ast::NodeId,
                              m: &ast::Mod,
                              name: Option<ast::Ident>) -> Module {
        let mut om = Module::new(name);
        for item in m.view_items.iter() {
            self.visit_view_item(item, &mut om);
        }
        om.where = span;
        om.attrs = attrs;
        om.vis = vis;
        om.id = id;
        for i in m.items.iter() {
            self.visit_item(*i, &mut om);
        }
        om
    }

    pub fn visit_view_item(&mut self, item: &ast::ViewItem, om: &mut Module) {
        if item.vis != ast::Public {
            return om.view_items.push(item.clone());
        }
        let item = match item.node {
            ast::ViewItemUse(ref paths) => {
                // rustc no longer supports "use foo, bar;"
                assert_eq!(paths.len(), 1);
                match self.visit_view_path(*paths.get(0), om) {
                    None => return,
                    Some(path) => {
                        ast::ViewItem {
                            node: ast::ViewItemUse(vec!(path)),
                            .. item.clone()
                        }
                    }
                }
            }
            ast::ViewItemExternCrate(..) => item.clone()
        };
        om.view_items.push(item);
    }

    fn visit_view_path(&mut self, path: @ast::ViewPath,
                       om: &mut Module) -> Option<@ast::ViewPath> {
        match path.node {
            ast::ViewPathSimple(_, _, id) => {
                if self.resolve_id(id, false, om) { return None }
            }
            ast::ViewPathList(ref p, ref paths, ref b) => {
                let mut mine = Vec::new();
                for path in paths.iter() {
                    if !self.resolve_id(path.node.id, false, om) {
                        mine.push(path.clone());
                    }
                }

                if mine.len() == 0 { return None }
                return Some(@::syntax::codemap::Spanned {
                    node: ast::ViewPathList(p.clone(), mine, b.clone()),
                    span: path.span,
                })
            }

            // these are feature gated anyway
            ast::ViewPathGlob(_, id) => {
                if self.resolve_id(id, true, om) { return None }
            }
        }
        return Some(path);
    }

    fn resolve_id(&mut self, id: ast::NodeId, glob: bool,
                  om: &mut Module) -> bool {
        let def = {
            let dm = match self.cx.tycx {
                Some(tcx) => tcx.def_map.borrow(),
                None => return false,
            };
            ast_util::def_id_of_def(*dm.get().get(&id))
        };
        if !ast_util::is_local(def) { return false }
        let analysis = match self.analysis {
            Some(analysis) => analysis, None => return false
        };
        if analysis.public_items.contains(&def.node) { return false }

        let item = self.cx.tycx.unwrap().map.get(def.node);
        match item {
            ast_map::NodeItem(it) => {
                if glob {
                    match it.node {
                        ast::ItemMod(ref m) => {
                            for vi in m.view_items.iter() {
                                self.visit_view_item(vi, om);
                            }
                            for i in m.items.iter() {
                                self.visit_item(*i, om);
                            }
                        }
                        _ => { fail!("glob not mapped to a module"); }
                    }
                } else {
                    self.visit_item(it, om);
                }
                true
            }
            _ => false,
        }
    }

    pub fn visit_item(&mut self, item: &ast::Item, om: &mut Module) {
        debug!("Visiting item {:?}", item);
        match item.node {
            ast::ItemMod(ref m) => {
                om.mods.push(self.visit_mod_contents(item.span,
                                                     item.attrs
                                                         .iter()
                                                         .map(|x| *x)
                                                         .collect(),
                                                     item.vis,
                                                     item.id,
                                                     m,
                                                     Some(item.ident)));
            },
            ast::ItemEnum(ref ed, ref gen) =>
                om.enums.push(self.visit_enum_def(item, ed, gen)),
            ast::ItemStruct(sd, ref gen) =>
                om.structs.push(self.visit_struct_def(item, sd, gen)),
            ast::ItemFn(fd, ref pur, ref abi, ref gen, _) =>
                om.fns.push(self.visit_fn(item, fd, pur, abi, gen)),
            ast::ItemTy(ty, ref gen) => {
                let t = Typedef {
                    ty: ty,
                    gen: gen.clone(),
                    name: item.ident,
                    id: item.id,
                    attrs: item.attrs.iter().map(|x| *x).collect(),
                    where: item.span,
                    vis: item.vis,
                };
                om.typedefs.push(t);
            },
            ast::ItemStatic(ty, ref mut_, ref exp) => {
                let s = Static {
                    type_: ty,
                    mutability: mut_.clone(),
                    expr: exp.clone(),
                    id: item.id,
                    name: item.ident,
                    attrs: item.attrs.iter().map(|x| *x).collect(),
                    where: item.span,
                    vis: item.vis,
                };
                om.statics.push(s);
            },
            ast::ItemTrait(ref gen, ref tr, ref met) => {
                let t = Trait {
                    name: item.ident,
                    methods: met.iter().map(|x| (*x).clone()).collect(),
                    generics: gen.clone(),
                    parents: tr.iter().map(|x| (*x).clone()).collect(),
                    id: item.id,
                    attrs: item.attrs.iter().map(|x| *x).collect(),
                    where: item.span,
                    vis: item.vis,
                };
                om.traits.push(t);
            },
            ast::ItemImpl(ref gen, ref tr, ty, ref meths) => {
                let i = Impl {
                    generics: gen.clone(),
                    trait_: tr.clone(),
                    for_: ty,
                    methods: meths.iter().map(|x| *x).collect(),
                    attrs: item.attrs.iter().map(|x| *x).collect(),
                    id: item.id,
                    where: item.span,
                    vis: item.vis,
                };
                om.impls.push(i);
            },
            ast::ItemForeignMod(ref fm) => {
                om.foreigns.push(fm.clone());
            }
            ast::ItemMac(ref _m) => {
                om.macros.push(Macro {
                    id: item.id,
                    attrs: item.attrs.iter().map(|x| *x).collect(),
                    name: item.ident,
                    where: item.span,
                })
            }
        }
    }
}
