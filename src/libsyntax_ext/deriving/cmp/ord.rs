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

use syntax::ast;
use syntax::ast::{MetaItem, Expr};
use syntax::codemap::Span;
use syntax::ext::base::{ExtCtxt, Annotatable};
use syntax::ext::build::AstBuilder;
use syntax::parse::token::InternedString;
use syntax::ptr::P;

pub fn expand_deriving_ord(cx: &mut ExtCtxt,
                           span: Span,
                           mitem: &MetaItem,
                           item: &Annotatable,
                           push: &mut FnMut(Annotatable))
{
    let inline = cx.meta_word(span, InternedString::new("inline"));
    let attrs = vec!(cx.attribute(span, inline));
    let trait_def = TraitDef {
        span: span,
        attributes: Vec::new(),
        path: path_std!(cx, core::cmp::Ord),
        additional_bounds: Vec::new(),
        generics: LifetimeBounds::empty(),
        is_unsafe: false,
        methods: vec!(
            MethodDef {
                name: "cmp",
                generics: LifetimeBounds::empty(),
                explicit_self: borrowed_explicit_self(),
                args: vec!(borrowed_self()),
                ret_ty: Literal(path_std!(cx, core::cmp::Ordering)),
                attributes: attrs,
                is_unsafe: false,
                combine_substructure: combine_substructure(Box::new(|a, b, c| {
                    cs_cmp(a, b, c)
                })),
            }
        ),
        associated_types: Vec::new(),
    };

    trait_def.expand(cx, mitem, item, push)
}


pub fn ordering_collapsed(cx: &mut ExtCtxt,
                          span: Span,
                          self_arg_tags: &[ast::Ident]) -> P<ast::Expr> {
    let lft = cx.expr_ident(span, self_arg_tags[0]);
    let rgt = cx.expr_addr_of(span, cx.expr_ident(span, self_arg_tags[1]));
    cx.expr_method_call(span, lft, cx.ident_of("cmp"), vec![rgt])
}

pub fn cs_cmp(cx: &mut ExtCtxt, span: Span,
              substr: &Substructure) -> P<Expr> {
    let test_id = cx.ident_of("__test");
    let equals_path = cx.path_global(span,
                                     cx.std_path(&["cmp", "Ordering", "Equal"]));

    let cmp_path = cx.std_path(&["cmp", "Ord", "cmp"]);

    /*
    Builds:

    let __test = ::std::cmp::Ord::cmp(&self_field1, &other_field1);
    if other == ::std::cmp::Ordering::Equal {
        let __test = ::std::cmp::Ord::cmp(&self_field2, &other_field2);
        if __test == ::std::cmp::Ordering::Equal {
            ...
        } else {
            __test
        }
    } else {
        __test
    }

    FIXME #6449: These `if`s could/should be `match`es.
    */
    cs_fold(
        // foldr nests the if-elses correctly, leaving the first field
        // as the outermost one, and the last as the innermost.
        false,
        |cx, span, old, self_f, other_fs| {
            // let __test = new;
            // if __test == ::std::cmp::Ordering::Equal {
            //    old
            // } else {
            //    __test
            // }

            let new = {
                let other_f = match (other_fs.len(), other_fs.get(0)) {
                    (1, Some(o_f)) => o_f,
                    _ => cx.span_bug(span, "not exactly 2 arguments in `derive(PartialOrd)`"),
                };

                let args = vec![
                    cx.expr_addr_of(span, self_f),
                    cx.expr_addr_of(span, other_f.clone()),
                ];

                cx.expr_call_global(span, cmp_path.clone(), args)
            };

            let assign = cx.stmt_let(span, false, test_id, new);

            let cond = cx.expr_binary(span, ast::BiEq,
                                      cx.expr_ident(span, test_id),
                                      cx.expr_path(equals_path.clone()));
            let if_ = cx.expr_if(span,
                                 cond,
                                 old, Some(cx.expr_ident(span, test_id)));
            cx.expr_block(cx.block(span, vec!(assign), Some(if_)))
        },
        cx.expr_path(equals_path.clone()),
        Box::new(|cx, span, (self_args, tag_tuple), _non_self_args| {
            if self_args.len() != 2 {
                cx.span_bug(span, "not exactly 2 arguments in `derives(Ord)`")
            } else {
                ordering_collapsed(cx, span, tag_tuple)
            }
        }),
        cx, span, substr)
}
