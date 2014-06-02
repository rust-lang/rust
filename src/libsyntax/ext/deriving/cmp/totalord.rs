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

use std::cmp::{Ordering, Equal, Less, Greater};

pub fn expand_deriving_totalord(cx: &mut ExtCtxt,
                                span: Span,
                                mitem: @MetaItem,
                                item: @Item,
                                push: |@Item|) {
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
                const_nonmatching: false,
                combine_substructure: combine_substructure(|a, b, c| {
                    cs_cmp(a, b, c)
                }),
            }
        )
    };

    trait_def.expand(cx, mitem, item, push)
}


pub fn ordering_const(cx: &mut ExtCtxt, span: Span, cnst: Ordering) -> ast::Path {
    let cnst = match cnst {
        Less => "Less",
        Equal => "Equal",
        Greater => "Greater"
    };
    cx.path_global(span,
                   vec!(cx.ident_of("std"),
                     cx.ident_of("cmp"),
                     cx.ident_of(cnst)))
}

pub fn cs_cmp(cx: &mut ExtCtxt, span: Span,
              substr: &Substructure) -> @Expr {
    let test_id = cx.ident_of("__test");
    let equals_path = ordering_const(cx, span, Equal);

    /*
    Builds:

    let __test = self_field1.cmp(&other_field2);
    if other == ::std::cmp::Equal {
        let __test = self_field2.cmp(&other_field2);
        if __test == ::std::cmp::Equal {
            ...
        } else {
            __test
        }
    } else {
        __test
    }

    FIXME #6449: These `if`s could/should be `match`es.
    */
    cs_same_method_fold(
        // foldr nests the if-elses correctly, leaving the first field
        // as the outermost one, and the last as the innermost.
        false,
        |cx, span, old, new| {
            // let __test = new;
            // if __test == ::std::cmp::Equal {
            //    old
            // } else {
            //    __test
            // }

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
        |cx, span, list, _| {
            match list {
                // an earlier nonmatching variant is Less than a
                // later one.
                [(self_var, _, _),
                 (other_var, _, _)] => {
                    let order = ordering_const(cx, span, self_var.cmp(&other_var));
                    cx.expr_path(order)
                }
                _ => cx.span_bug(span, "not exactly 2 arguments in `deriving(Ord)`")
            }
        },
        cx, span, substr)
}
