// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use deriving::generic::*;
use deriving::generic::ty::*;

use syntax::ast::{MetaItem, Expr};
use syntax::codemap::Span;
use syntax::ext::base::{ExtCtxt, Annotatable};
use syntax::ext::build::AstBuilder;
use syntax::parse::token::InternedString;
use syntax::ptr::P;

pub fn expand_deriving_eq(cx: &mut ExtCtxt,
                          span: Span,
                          mitem: &MetaItem,
                          item: &Annotatable,
                          push: &mut FnMut(Annotatable))
{
    fn cs_total_eq_assert(cx: &mut ExtCtxt, span: Span, substr: &Substructure) -> P<Expr> {
        cs_same_method(
            |cx, span, exprs| {
                // create `a.<method>(); b.<method>(); c.<method>(); ...`
                // (where method is `assert_receiver_is_total_eq`)
                let stmts = exprs.into_iter().map(|e| cx.stmt_expr(e)).collect();
                let block = cx.block(span, stmts, None);
                cx.expr_block(block)
            },
            Box::new(|cx, sp, _, _| {
                cx.span_bug(sp, "non matching enums in derive(Eq)?") }),
            cx,
            span,
            substr
        )
    }

    let inline = cx.meta_word(span, InternedString::new("inline"));
    let hidden = cx.meta_word(span, InternedString::new("hidden"));
    let doc = cx.meta_list(span, InternedString::new("doc"), vec!(hidden));
    let attrs = vec!(cx.attribute(span, inline),
                     cx.attribute(span, doc));
    let trait_def = TraitDef {
        span: span,
        attributes: Vec::new(),
        path: path_std!(cx, core::cmp::Eq),
        additional_bounds: Vec::new(),
        generics: LifetimeBounds::empty(),
        is_unsafe: false,
        methods: vec!(
            MethodDef {
                name: "assert_receiver_is_total_eq",
                generics: LifetimeBounds::empty(),
                explicit_self: borrowed_explicit_self(),
                args: vec!(),
                ret_ty: nil_ty(),
                attributes: attrs,
                is_unsafe: false,
                combine_substructure: combine_substructure(Box::new(|a, b, c| {
                    cs_total_eq_assert(a, b, c)
                }))
            }
        ),
        associated_types: Vec::new(),
    };
    trait_def.expand(cx, mitem, item, push)
}
