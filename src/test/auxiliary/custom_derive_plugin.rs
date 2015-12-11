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
#![feature(box_syntax)]
#![feature(rustc_private)]

extern crate syntax;
extern crate syntax_ext;
extern crate rustc;
extern crate rustc_plugin;

use syntax::ast;
use syntax::codemap::Span;
use syntax::ext::base::{MultiDecorator, ExtCtxt, Annotatable};
use syntax::ext::build::AstBuilder;
use syntax::parse::token;
use syntax::ptr::P;
use syntax_ext::deriving::generic::{cs_fold, TraitDef, MethodDef, combine_substructure};
use syntax_ext::deriving::generic::ty::{Literal, LifetimeBounds, Path, borrowed_explicit_self};
use rustc_plugin::Registry;

#[plugin_registrar]
pub fn plugin_registrar(reg: &mut Registry) {
    reg.register_syntax_extension(
        token::intern("derive_TotalSum"),
        MultiDecorator(box expand));
}

fn expand(cx: &mut ExtCtxt,
          span: Span,
          mitem: &ast::MetaItem,
          item: &Annotatable,
          push: &mut FnMut(Annotatable)) {
    let trait_def = TraitDef {
        span: span,
        attributes: vec![],
        path: Path::new(vec!["TotalSum"]),
        additional_bounds: vec![],
        generics: LifetimeBounds::empty(),
        associated_types: vec![],
        is_unsafe: false,
        methods: vec![
            MethodDef {
                name: "total_sum",
                generics: LifetimeBounds::empty(),
                explicit_self: borrowed_explicit_self(),
                args: vec![],
                ret_ty: Literal(Path::new_local("isize")),
                attributes: vec![],
                is_unsafe: false,
                combine_substructure: combine_substructure(box |cx, span, substr| {
                    let zero = cx.expr_isize(span, 0);
                    cs_fold(false,
                            |cx, span, subexpr, field, _| {
                                cx.expr_binary(span, ast::BiAdd, subexpr,
                                    cx.expr_method_call(span, field,
                                        token::str_to_ident("total_sum"), vec![]))
                            },
                            zero,
                            box |cx, span, _, _| { cx.span_bug(span, "wtf??"); },
                            cx, span, substr)
                }),
            },
        ],
    };

    trait_def.expand(cx, mitem, item, push)
}
