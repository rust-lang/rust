// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// force-host

#![feature(plugin_registrar, rustc_private)]

extern crate syntax;
extern crate syntax_ext;
extern crate rustc_plugin;

use syntax_ext::deriving;
use deriving::generic::*;
use deriving::generic::ty::*;

use rustc_plugin::Registry;
use syntax::ast::*;
use syntax::codemap::Span;
use syntax::ext::base::*;
use syntax::ext::build::AstBuilder;
use syntax::symbol::Symbol;
use syntax::ptr::P;

#[plugin_registrar]
pub fn plugin_registrar(reg: &mut Registry) {
    reg.register_syntax_extension(Symbol::intern("derive_CustomPartialEq"),
                                  MultiDecorator(Box::new(expand_deriving_partial_eq)));
}

fn expand_deriving_partial_eq(cx: &mut ExtCtxt, span: Span, mitem: &MetaItem, item: &Annotatable,
                              push: &mut FnMut(Annotatable)) {
    // structures are equal if all fields are equal, and non equal, if
    // any fields are not equal or if the enum variants are different
    fn cs_eq(cx: &mut ExtCtxt, span: Span, substr: &Substructure) -> P<Expr> {
        cs_fold(true,
                |cx, span, subexpr, self_f, other_fs| {
                    let other_f = (other_fs.len(), other_fs.get(0)).1.unwrap();
                    let eq = cx.expr_binary(span, BinOpKind::Eq, self_f, other_f.clone());
                    cx.expr_binary(span, BinOpKind::And, subexpr, eq)
                },
                cx.expr_bool(span, true),
                Box::new(|cx, span, _, _| cx.expr_bool(span, false)),
                cx,
                span,
                substr)
    }

    let inline = cx.meta_word(span, Symbol::intern("inline"));
    let attrs = vec![cx.attribute(span, inline)];
    let methods = vec![MethodDef {
        name: "eq",
        generics: LifetimeBounds::empty(),
        explicit_self: borrowed_explicit_self(),
        args: vec![borrowed_self()],
        ret_ty: Literal(deriving::generic::ty::Path::new_local("bool")),
        attributes: attrs,
        is_unsafe: false,
        unify_fieldless_variants: true,
        combine_substructure: combine_substructure(Box::new(cs_eq)),
    }];

    let trait_def = TraitDef {
        span: span,
        attributes: Vec::new(),
        path: deriving::generic::ty::Path::new(vec!["std", "cmp", "PartialEq"]),
        additional_bounds: Vec::new(),
        generics: LifetimeBounds::empty(),
        is_unsafe: false,
        supports_unions: false,
        methods: methods,
        associated_types: Vec::new(),
    };
    trait_def.expand(cx, mitem, item, push)
}
