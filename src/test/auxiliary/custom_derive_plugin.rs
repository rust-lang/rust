// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// force-host

#![feature(plugin_registrar)]
#![feature(rustc_private)]

extern crate syntax;
extern crate rustc;

use syntax::ast;
use syntax::codemap::Span;
use syntax::ext::base::{Decorator, ExtCtxt};
use syntax::ext::build::AstBuilder;
use syntax::ext::deriving::generic::{cs_fold, TraitDef, MethodDef, combine_substructure};
use syntax::ext::deriving::generic::ty::{Literal, LifetimeBounds, Path, borrowed_explicit_self};
use syntax::parse::token;
use syntax::ptr::P;
use rustc::plugin::Registry;

#[plugin_registrar]
pub fn plugin_registrar(reg: &mut Registry) {
    reg.register_syntax_extension(
        token::intern("derive_TotalSum"),
        Decorator(Box::new(expand)));
}

fn expand(cx: &mut ExtCtxt,
          span: Span,
          mitem: &ast::MetaItem,
          item: &ast::Item,
          push: &mut FnMut(P<ast::Item>)) {
    let trait_def = TraitDef {
        span: span,
        attributes: vec![],
        path: Path::new(vec!["TotalSum"]),
        additional_bounds: vec![],
        generics: LifetimeBounds::empty(),
        associated_types: vec![],
        methods: vec![
            MethodDef {
                name: "total_sum",
                generics: LifetimeBounds::empty(),
                explicit_self: borrowed_explicit_self(),
                args: vec![],
                ret_ty: Literal(Path::new_local("isize")),
                attributes: vec![],
                combine_substructure: combine_substructure(Box::new(|cx, span, substr| {
                    let zero = cx.expr_int(span, 0);
                    cs_fold(false,
                            |cx, span, subexpr, field, _| {
                                cx.expr_binary(span, ast::BiAdd, subexpr,
                                    cx.expr_method_call(span, field,
                                        token::str_to_ident("total_sum"), vec![]))
                            },
                            zero,
                            Box::new(|cx, span, _, _| { cx.span_bug(span, "wtf??"); }),
                            cx, span, substr)
                })),
            },
        ],
    };

    trait_def.expand(cx, mitem, item, |i| push(i))
}
