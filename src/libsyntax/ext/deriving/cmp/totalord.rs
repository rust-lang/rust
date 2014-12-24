// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ast;
use ast::{MetaItem, Item, Expr};
use codemap::Span;
use ext::base::ExtCtxt;
use ext::build::AstBuilder;
use ext::deriving::generic::*;
use ext::deriving::generic::ty::*;
use parse::token::InternedString;
use ptr::P;

pub fn expand_deriving_totalord<F>(cx: &mut ExtCtxt,
                                   span: Span,
                                   mitem: &MetaItem,
                                   item: &Item,
                                   push: F) where
    F: FnOnce(P<Item>),
{
    let inline = cx.meta_word(span, InternedString::new("inline"));
    let attrs = vec!(cx.attribute(span, inline));
    let trait_def = TraitDef {
        span: span,
        attributes: Vec::new(),
        path: Path::new(vec!("std", "cmp", "Ord")),
        additional_bounds: Vec::new(),
        generics: LifetimeBounds::empty(),
        methods: vec!(
            MethodDef {
                name: "cmp",
                generics: LifetimeBounds::empty(),
                explicit_self: borrowed_explicit_self(),
                args: vec!(borrowed_self()),
                ret_ty: Literal(Path::new(vec!("std", "cmp", "Ordering"))),
                attributes: attrs,
                combine_substructure: combine_substructure(|a, b, c| {
                    cs_cmp(a, b, c)
                }),
            }
        )
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
                                     vec!(cx.ident_of("std"),
                                          cx.ident_of("cmp"),
                                          cx.ident_of("Ordering"),
                                          cx.ident_of("Equal")));

    let cmp_path = vec![
        cx.ident_of("std"),
        cx.ident_of("cmp"),
        cx.ident_of("Ord"),
        cx.ident_of("cmp"),
    ];

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
                let other_f = match other_fs {
                    [ref o_f] => o_f,
                    _ => cx.span_bug(span, "not exactly 2 arguments in `deriving(PartialOrd)`"),
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
        |cx, span, (self_args, tag_tuple), _non_self_args| {
            if self_args.len() != 2 {
                cx.span_bug(span, "not exactly 2 arguments in `deriving(Ord)`")
            } else {
                ordering_collapsed(cx, span, tag_tuple)
            }
        },
        cx, span, substr)
}
