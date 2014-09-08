// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// force-host
// no-prefer-dynamic

#![crate_type = "dylib"]
#![feature(plugin_registrar, quote)]

extern crate syntax;
extern crate rustc;

use std::gc::Gc;

use syntax::ast::{
    Expr,
    Ident,
    Item,
    MetaItem,
};
use syntax::ast;
use syntax::attr;
use syntax::codemap::Span;
use syntax::ext::base::{ExtCtxt, ItemDecorator};
use syntax::ext::build::AstBuilder;
use syntax::ext::deriving::generic::{
    MethodDef,
    Named,
    StaticStruct,
    Struct,
    Substructure,
    TraitDef,
    Unnamed,
    combine_substructure,
};
use syntax::ext::deriving::generic::ty::{
    Borrowed,
    LifetimeBounds,
    Literal,
    Self,
    Path,
    Ptr,
    borrowed_explicit_self,
};
use syntax::parse::token;

use rustc::plugin::Registry;

#[plugin_registrar]
pub fn registrar(reg: &mut Registry) {
    reg.register_syntax_extension(
        token::intern("deriving_field_names"),
        ItemDecorator(expand_deriving_field_names));
}

fn expand_deriving_field_names(cx: &mut ExtCtxt,
                               sp: Span,
                               mitem: Gc<MetaItem>,
                               item: Gc<Item>,
                               push: |Gc<ast::Item>|) {
    let trait_def = TraitDef {
        span: sp,
        attributes: vec!(),
        path: Path::new_local("FieldNames"),
        additional_bounds: Vec::new(),
        generics: LifetimeBounds::empty(),
        methods: vec!(
            MethodDef {
                name: "field_names",
                generics: LifetimeBounds::empty(),
                explicit_self: borrowed_explicit_self(),
                args: Vec::new(),
                ret_ty: Literal(
                    Path::new_(
                        vec!("Vec"),
                        None,
                        vec!(
                            box Ptr(
                                box Literal(Path::new_local("str")),
                                Borrowed(Some("'static"), ast::MutImmutable))),
                        false)),
                attributes: vec!(),
                combine_substructure: combine_substructure(|a, b, c| {
                    field_names_substructure(a, b, c)
                }),
            },
            MethodDef {
                name: "static_field_names",
                generics: LifetimeBounds::empty(),
                explicit_self: None,
                args: vec!(
                    Literal(
                        Path::new_(
                            vec!("std", "option", "Option"),
                            None,
                            vec!(box Self),
                            true))),
                ret_ty: Literal(
                    Path::new_(
                        vec!("Vec"),
                        None,
                        vec!(
                            box Ptr(
                                box Literal(Path::new_local("str")),
                                Borrowed(Some("'static"), ast::MutImmutable))),
                        false)),
                attributes: vec!(),
                combine_substructure: combine_substructure(|a, b, c| {
                    field_names_substructure(a, b, c)
                }),
            },
        )
    };

    trait_def.expand(cx, mitem, item, push)
}

fn field_names_substructure(cx: &ExtCtxt,
                            _span: Span,
                            substr: &Substructure) -> Gc<Expr> {

    let ident_attrs: Vec<(Span, Option<Ident>, &[ast::Attribute])> = match substr.fields {
        Struct(ref fields) => {
            fields.iter()
                .map(|field_info| {
                    (field_info.span, field_info.name, field_info.attrs)
                })
                .collect()
        }
        StaticStruct(_, ref summary) => {
            match *summary {
                Unnamed(_) => {
                    fail!()
                }
                Named(ref fields) => {
                    fields.iter()
                        .map(|field_info| {
                            (field_info.span, Some(field_info.name), field_info.attrs)
                        })
                        .collect()
                }
            }
        }

        _ => cx.bug("expected Struct in deriving_test")
    };

    let stmts: Vec<Gc<ast::Stmt>> = ident_attrs.iter()
        .enumerate()
        .map(|(i, &(span, ident, attrs))| {
            let name = find_name(attrs);

            let name = match (name, ident) {
                (Some(serial), _) => serial.clone(),
                (None, Some(id)) => token::get_ident(id),
                (None, None) => token::intern_and_get_ident(format!("_field{}", i).as_slice()),
            };

            let name = cx.expr_str(span, name);

            quote_stmt!(
                cx,
                parts.push($name);
            )
        })
        .collect();

    quote_expr!(cx, {
        let mut parts = Vec::new();
        $stmts
        parts
    })
}

fn find_name(attrs: &[ast::Attribute]) -> Option<token::InternedString> {
    for attr in attrs.iter() {
        match attr.node.value.node {
            ast::MetaNameValue(ref at_name, ref value) => {
                match (at_name.get(), &value.node) {
                    ("name", &ast::LitStr(ref string, _)) => {
                        attr::mark_used(attr);
                        return Some(string.clone());
                    },
                    _ => ()
                }
            },
            _ => ()
        }
    }
    None
}
