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

use syntax::abi::AbiSet;
use syntax::ast;
use syntax::codemap::Span;

use doctree::*;

pub struct RustdocVisitor {
    module: Module,
    attrs: ~[ast::Attribute],
}

impl RustdocVisitor {
    pub fn new() -> RustdocVisitor {
        RustdocVisitor {
            module: Module::new(None),
            attrs: ~[],
        }
    }
}

impl RustdocVisitor {
    pub fn visit(@mut self, crate: &ast::Crate) {
        self.attrs = crate.attrs.clone();
        fn visit_struct_def(item: &ast::item, sd: @ast::struct_def, generics:
                            &ast::Generics) -> Struct {
            debug!("Visiting struct");
            let struct_type = struct_type_from_def(sd);
            Struct {
                id: item.id,
                struct_type: struct_type,
                name: item.ident,
                vis: item.vis,
                attrs: item.attrs.clone(),
                generics: generics.clone(),
                fields: sd.fields.clone(),
                where: item.span
            }
        }

        fn visit_enum_def(it: &ast::item, def: &ast::enum_def, params: &ast::Generics) -> Enum {
            debug!("Visiting enum");
            let mut vars: ~[Variant] = ~[];
            for x in def.variants.iter() {
                vars.push(Variant {
                    name: x.node.name,
                    attrs: x.node.attrs.clone(),
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
                attrs: it.attrs.clone(),
                id: it.id,
                where: it.span,
            }
        }

        fn visit_fn(item: &ast::item, fd: &ast::fn_decl, purity: &ast::purity,
                     _abi: &AbiSet, gen: &ast::Generics) -> Function {
            debug!("Visiting fn");
            Function {
                id: item.id,
                vis: item.vis,
                attrs: item.attrs.clone(),
                decl: fd.clone(),
                name: item.ident,
                where: item.span,
                generics: gen.clone(),
                purity: *purity,
            }
        }

        fn visit_mod_contents(span: Span, attrs: ~[ast::Attribute], vis:
                              ast::visibility, id: ast::NodeId, m: &ast::_mod,
                              name: Option<ast::Ident>) -> Module {
            let mut om = Module::new(name);
            om.view_items = m.view_items.clone();
            om.where = span;
            om.attrs = attrs;
            om.vis = vis;
            om.id = id;
            for i in m.items.iter() {
                visit_item(*i, &mut om);
            }
            om
        }

        fn visit_item(item: &ast::item, om: &mut Module) {
            debug!("Visiting item {:?}", item);
            match item.node {
                ast::item_mod(ref m) => {
                    om.mods.push(visit_mod_contents(item.span, item.attrs.clone(),
                                                    item.vis, item.id, m,
                                                    Some(item.ident)));
                },
                ast::item_enum(ref ed, ref gen) => om.enums.push(visit_enum_def(item, ed, gen)),
                ast::item_struct(sd, ref gen) => om.structs.push(visit_struct_def(item, sd, gen)),
                ast::item_fn(fd, ref pur, ref abi, ref gen, _) =>
                    om.fns.push(visit_fn(item, fd, pur, abi, gen)),
                ast::item_ty(ty, ref gen) => {
                    let t = Typedef {
                        ty: ty,
                        gen: gen.clone(),
                        name: item.ident,
                        id: item.id,
                        attrs: item.attrs.clone(),
                        where: item.span,
                        vis: item.vis,
                    };
                    om.typedefs.push(t);
                },
                ast::item_static(ty, ref mut_, ref exp) => {
                    let s = Static {
                        type_: ty,
                        mutability: mut_.clone(),
                        expr: exp.clone(),
                        id: item.id,
                        name: item.ident,
                        attrs: item.attrs.clone(),
                        where: item.span,
                        vis: item.vis,
                    };
                    om.statics.push(s);
                },
                ast::item_trait(ref gen, ref tr, ref met) => {
                    let t = Trait {
                        name: item.ident,
                        methods: met.clone(),
                        generics: gen.clone(),
                        parents: tr.clone(),
                        id: item.id,
                        attrs: item.attrs.clone(),
                        where: item.span,
                        vis: item.vis,
                    };
                    om.traits.push(t);
                },
                ast::item_impl(ref gen, ref tr, ty, ref meths) => {
                    let i = Impl {
                        generics: gen.clone(),
                        trait_: tr.clone(),
                        for_: ty,
                        methods: meths.clone(),
                        attrs: item.attrs.clone(),
                        id: item.id,
                        where: item.span,
                        vis: item.vis,
                    };
                    om.impls.push(i);
                },
                ast::item_foreign_mod(ref fm) => {
                    om.foreigns.push(fm.clone());
                }
                _ => (),
            }
        }

        self.module = visit_mod_contents(crate.span, crate.attrs.clone(),
                                         ast::public, ast::CRATE_NODE_ID,
                                         &crate.module, None);
    }
}
