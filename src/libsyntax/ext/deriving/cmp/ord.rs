// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::prelude::*;

use ast;
use ast::{meta_item, item, expr};
use codemap::span;
use ext::base::ExtCtxt;
use ext::build::AstBuilder;
use ext::deriving::generic::*;

pub fn expand_deriving_ord(cx: @ExtCtxt,
                           span: span,
                           mitem: @meta_item,
                           in_items: ~[@item]) -> ~[@item] {
    macro_rules! md (
        ($name:expr, $func:expr, $op:expr) => {
            MethodDef {
                name: $name,
                generics: LifetimeBounds::empty(),
                explicit_self: borrowed_explicit_self(),
                args: ~[borrowed_self()],
                ret_ty: Literal(Path::new(~["bool"])),
                const_nonmatching: false,
                combine_substructure: |cx, span, substr| $func($op, cx, span, substr)
            }
        }
    );

    let trait_def = TraitDef {
        path: Path::new(~["std", "cmp", "Ord"]),
        additional_bounds: ~[],
        generics: LifetimeBounds::empty(),
        methods: ~[
            md!("lt", cs_strict, true),
            md!("le", cs_nonstrict, true), // inverse operation
            md!("gt", cs_strict, false),
            md!("ge", cs_nonstrict, false)
        ]
    };
    trait_def.expand(cx, span, mitem, in_items)
}

/// Strict inequality.
fn cs_strict(less: bool, cx: @ExtCtxt, span: span, substr: &Substructure) -> @expr {
    let op = if less {ast::lt} else {ast::gt};
    cs_fold(
        false, // need foldr,
        |cx, span, subexpr, self_f, other_fs| {
            /*
            build up a series of chain ||'s and &&'s from the inside
            out (hence foldr) to get lexical ordering, i.e. for op ==
            `ast::lt`

            ```
            self.f1 < other.f1 || (!(other.f1 < self.f1) &&
                (self.f2 < other.f2 || (!(other.f2 < self.f2) &&
                    (false)
                ))
            )
            ```

            The optimiser should remove the redundancy. We explicitly
            get use the binops to avoid auto-deref derefencing too many
            layers of pointers, if the type includes pointers.
            */
            let other_f = match other_fs {
                [o_f] => o_f,
                _ => cx.span_bug(span, "Not exactly 2 arguments in `deriving(Ord)`")
            };

            let cmp = cx.expr_binary(span, op,
                                     cx.expr_deref(span, self_f),
                                     cx.expr_deref(span, other_f));

            let not_cmp = cx.expr_binary(span, op,
                                         cx.expr_deref(span, other_f),
                                         cx.expr_deref(span, self_f));
            let not_cmp = cx.expr_unary(span, ast::not, not_cmp);

            let and = cx.expr_binary(span, ast::and,
                                     not_cmp, subexpr);
            cx.expr_binary(span, ast::or, cmp, and)
        },
        cx.expr_bool(span, false),
        |cx, span, args, _| {
            // nonmatching enums, order by the order the variants are
            // written
            match args {
                [(self_var, _, _),
                 (other_var, _, _)] =>
                    cx.expr_bool(span,
                                 if less {
                                     self_var < other_var
                                 } else {
                                     self_var > other_var
                                 }),
                _ => cx.span_bug(span, "Not exactly 2 arguments in `deriving(Ord)`")
            }
        },
        cx, span, substr)
}

fn cs_nonstrict(less: bool, cx: @ExtCtxt, span: span, substr: &Substructure) -> @expr {
    // Example: ge becomes !(*self < *other), le becomes !(*self > *other)

    let inverse_op = if less {ast::gt} else {ast::lt};
    match substr.self_args {
        [self_, other] => {
            let inverse_cmp = cx.expr_binary(span, inverse_op,
                                             cx.expr_deref(span, self_),
                                             cx.expr_deref(span, other));

            cx.expr_unary(span, ast::not, inverse_cmp)
        }
        _ => cx.span_bug(span, "Not exactly 2 arguments in `deriving(Ord)`")
    }
}
