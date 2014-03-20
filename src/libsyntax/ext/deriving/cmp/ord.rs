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

use std::vec::Vec;

pub fn expand_deriving_ord(cx: &mut ExtCtxt,
                           span: Span,
                           mitem: @MetaItem,
                           item: @Item,
                           push: |@Item|) {
    macro_rules! md (
        ($name:expr, $op:expr, $equal:expr) => {
            MethodDef {
                name: $name,
                generics: LifetimeBounds::empty(),
                explicit_self: borrowed_explicit_self(),
                args: vec!(borrowed_self()),
                ret_ty: Literal(Path::new(vec!("bool"))),
                inline: true,
                const_nonmatching: false,
                combine_substructure: |cx, span, substr| cs_op($op, $equal, cx, span, substr)
            }
        }
    );

    let trait_def = TraitDef {
        span: span,
        attributes: Vec::new(),
        path: Path::new(vec!("std", "cmp", "Ord")),
        additional_bounds: Vec::new(),
        generics: LifetimeBounds::empty(),
        methods: vec!(
            md!("lt", true, false),
            md!("le", true, true),
            md!("gt", false, false),
            md!("ge", false, true)
        )
    };
    trait_def.expand(cx, mitem, item, push)
}

/// Strict inequality.
fn cs_op(less: bool, equal: bool, cx: &mut ExtCtxt, span: Span, substr: &Substructure) -> @Expr {
    let op = if less {ast::BiLt} else {ast::BiGt};
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
                _ => cx.span_bug(span, "not exactly 2 arguments in `deriving(Ord)`")
            };

            let cmp = cx.expr_binary(span, op, self_f, other_f);

            let not_cmp = cx.expr_unary(span, ast::UnNot,
                                        cx.expr_binary(span, op, other_f, self_f));

            let and = cx.expr_binary(span, ast::BiAnd, not_cmp, subexpr);
            cx.expr_binary(span, ast::BiOr, cmp, and)
        },
        cx.expr_bool(span, equal),
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
                _ => cx.span_bug(span, "not exactly 2 arguments in `deriving(Ord)`")
            }
        },
        cx, span, substr)
}
