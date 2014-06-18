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

use std::gc::Gc;

pub fn expand_deriving_ord(cx: &mut ExtCtxt,
                           span: Span,
                           mitem: Gc<MetaItem>,
                           item: Gc<Item>,
                           push: |Gc<Item>|) {
    macro_rules! md (
        ($name:expr, $op:expr, $equal:expr) => { {
            let inline = cx.meta_word(span, InternedString::new("inline"));
            let attrs = vec!(cx.attribute(span, inline));
            MethodDef {
                name: $name,
                generics: LifetimeBounds::empty(),
                explicit_self: borrowed_explicit_self(),
                args: vec!(borrowed_self()),
                ret_ty: Literal(Path::new(vec!("bool"))),
                attributes: attrs,
                const_nonmatching: false,
                combine_substructure: combine_substructure(|cx, span, substr| {
                    cs_op($op, $equal, cx, span, substr)
                })
            }
        } }
    );

    let ordering_ty = Literal(Path::new(vec!["std", "cmp", "Ordering"]));
    let ret_ty = Literal(Path::new_(vec!["std", "option", "Option"],
                                    None,
                                    vec![box ordering_ty],
                                    true));

    let inline = cx.meta_word(span, InternedString::new("inline"));
    let attrs = vec!(cx.attribute(span, inline));

    let partial_cmp_def = MethodDef {
        name: "partial_cmp",
        generics: LifetimeBounds::empty(),
        explicit_self: borrowed_explicit_self(),
        args: vec![borrowed_self()],
        ret_ty: ret_ty,
        attributes: attrs,
        const_nonmatching: false,
        combine_substructure: combine_substructure(|cx, span, substr| {
            cs_partial_cmp(cx, span, substr)
        })
    };

    let trait_def = TraitDef {
        span: span,
        attributes: vec![],
        path: Path::new(vec!["std", "cmp", "PartialOrd"]),
        additional_bounds: vec![],
        generics: LifetimeBounds::empty(),
        methods: vec![
            partial_cmp_def,
            md!("lt", true, false),
            md!("le", true, true),
            md!("gt", false, false),
            md!("ge", false, true)
        ]
    };
    trait_def.expand(cx, mitem, item, push)
}

pub fn some_ordering_const(cx: &mut ExtCtxt, span: Span, cnst: Ordering) -> Gc<ast::Expr> {
    let cnst = match cnst {
        Less => "Less",
        Equal => "Equal",
        Greater => "Greater"
    };
    let ordering = cx.path_global(span,
                                  vec!(cx.ident_of("std"),
                                       cx.ident_of("cmp"),
                                       cx.ident_of(cnst)));
    let ordering = cx.expr_path(ordering);
    cx.expr_some(span, ordering)
}

pub fn cs_partial_cmp(cx: &mut ExtCtxt, span: Span,
              substr: &Substructure) -> Gc<Expr> {
    let test_id = cx.ident_of("__test");
    let equals_expr = some_ordering_const(cx, span, Equal);

    /*
    Builds:

    let __test = self_field1.partial_cmp(&other_field2);
    if __test == ::std::option::Some(::std::cmp::Equal) {
        let __test = self_field2.partial_cmp(&other_field2);
        if __test == ::std::option::Some(::std::cmp::Equal) {
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
            // if __test == Some(::std::cmp::Equal) {
            //    old
            // } else {
            //    __test
            // }

            let assign = cx.stmt_let(span, false, test_id, new);

            let cond = cx.expr_binary(span, ast::BiEq,
                                      cx.expr_ident(span, test_id),
                                      equals_expr.clone());
            let if_ = cx.expr_if(span,
                                 cond,
                                 old, Some(cx.expr_ident(span, test_id)));
            cx.expr_block(cx.block(span, vec!(assign), Some(if_)))
        },
        equals_expr.clone(),
        |cx, span, list, _| {
            match list {
                // an earlier nonmatching variant is Less than a
                // later one.
                [(self_var, _, _), (other_var, _, _)] =>
                     some_ordering_const(cx, span, self_var.cmp(&other_var)),
                _ => cx.span_bug(span, "not exactly 2 arguments in `deriving(Ord)`")
            }
        },
        cx, span, substr)
}

/// Strict inequality.
fn cs_op(less: bool, equal: bool, cx: &mut ExtCtxt, span: Span,
         substr: &Substructure) -> Gc<Expr> {
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
            get use the binops to avoid auto-deref dereferencing too many
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
