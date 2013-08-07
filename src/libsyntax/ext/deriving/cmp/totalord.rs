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
use ast::{MetaItem, item, expr};
use codemap::span;
use ext::base::ExtCtxt;
use ext::build::AstBuilder;
use ext::deriving::generic::*;
use std::cmp::{Ordering, Equal, Less, Greater};

pub fn expand_deriving_totalord(cx: @ExtCtxt,
                                span: span,
                                mitem: @MetaItem,
                                in_items: ~[@item]) -> ~[@item] {
    let trait_def = TraitDef {
        path: Path::new(~["std", "cmp", "TotalOrd"]),
        additional_bounds: ~[],
        generics: LifetimeBounds::empty(),
        methods: ~[
            MethodDef {
                name: "cmp",
                generics: LifetimeBounds::empty(),
                explicit_self: borrowed_explicit_self(),
                args: ~[borrowed_self()],
                ret_ty: Literal(Path::new(~["std", "cmp", "Ordering"])),
                const_nonmatching: false,
                combine_substructure: cs_cmp
            }
        ]
    };

    trait_def.expand(cx, span, mitem, in_items)
}


pub fn ordering_const(cx: @ExtCtxt, span: span, cnst: Ordering) -> ast::Path {
    let cnst = match cnst {
        Less => "Less",
        Equal => "Equal",
        Greater => "Greater"
    };
    cx.path_global(span,
                   ~[cx.ident_of("std"),
                     cx.ident_of("cmp"),
                     cx.ident_of(cnst)])
}

pub fn cs_cmp(cx: @ExtCtxt, span: span,
              substr: &Substructure) -> @expr {
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

            let cond = cx.expr_binary(span, ast::eq,
                                      cx.expr_ident(span, test_id),
                                      cx.expr_path(equals_path.clone()));
            let if_ = cx.expr_if(span,
                                 cond,
                                 old, Some(cx.expr_ident(span, test_id)));
            cx.expr_block(cx.block(span, ~[assign], Some(if_)))
        },
        cx.expr_path(equals_path.clone()),
        |cx, span, list, _| {
            match list {
                // an earlier nonmatching variant is Less than a
                // later one.
                [(self_var, _, _),
                 (other_var, _, _)] => cx.expr_path(ordering_const(cx, span,
                                                                   self_var.cmp(&other_var))),
                _ => cx.span_bug(span, "Not exactly 2 arguments in `deriving(TotalOrd)`")
            }
        },
        cx, span, substr)
}
