// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub use self::OrderingOp::*;

use deriving::generic::*;
use deriving::generic::ty::*;

use syntax::ast;
use syntax::ast::{MetaItem, Expr};
use syntax::codemap::Span;
use syntax::ext::base::{ExtCtxt, Annotatable};
use syntax::ext::build::AstBuilder;
use syntax::parse::token::InternedString;
use syntax::ptr::P;

pub fn expand_deriving_partial_ord(cx: &mut ExtCtxt,
                                   span: Span,
                                   mitem: &MetaItem,
                                   item: &Annotatable,
                                   push: &mut FnMut(Annotatable))
{
    macro_rules! md {
        ($name:expr, $op:expr, $equal:expr) => { {
            let inline = cx.meta_word(span, InternedString::new("inline"));
            let attrs = vec!(cx.attribute(span, inline));
            MethodDef {
                name: $name,
                generics: LifetimeBounds::empty(),
                explicit_self: borrowed_explicit_self(),
                args: vec!(borrowed_self()),
                ret_ty: Literal(path_local!(bool)),
                attributes: attrs,
                is_unsafe: false,
                combine_substructure: combine_substructure(Box::new(|cx, span, substr| {
                    cs_op($op, $equal, cx, span, substr)
                }))
            }
        } }
    }

    let ordering_ty = Literal(path_std!(cx, core::cmp::Ordering));
    let ret_ty = Literal(Path::new_(pathvec_std!(cx, core::option::Option),
                                    None,
                                    vec![Box::new(ordering_ty)],
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
        is_unsafe: false,
        combine_substructure: combine_substructure(Box::new(|cx, span, substr| {
            cs_partial_cmp(cx, span, substr)
        }))
    };

    let trait_def = TraitDef {
        span: span,
        attributes: vec![],
        path: path_std!(cx, core::cmp::PartialOrd),
        additional_bounds: vec![],
        generics: LifetimeBounds::empty(),
        is_unsafe: false,
        methods: vec![
            partial_cmp_def,
            md!("lt", true, false),
            md!("le", true, true),
            md!("gt", false, false),
            md!("ge", false, true)
        ],
        associated_types: Vec::new(),
    };
    trait_def.expand(cx, mitem, item, push)
}

#[derive(Copy, Clone)]
pub enum OrderingOp {
    PartialCmpOp, LtOp, LeOp, GtOp, GeOp,
}

pub fn some_ordering_collapsed(cx: &mut ExtCtxt,
                               span: Span,
                               op: OrderingOp,
                               self_arg_tags: &[ast::Ident]) -> P<ast::Expr> {
    let lft = cx.expr_ident(span, self_arg_tags[0]);
    let rgt = cx.expr_addr_of(span, cx.expr_ident(span, self_arg_tags[1]));
    let op_str = match op {
        PartialCmpOp => "partial_cmp",
        LtOp => "lt", LeOp => "le",
        GtOp => "gt", GeOp => "ge",
    };
    cx.expr_method_call(span, lft, cx.ident_of(op_str), vec![rgt])
}

pub fn cs_partial_cmp(cx: &mut ExtCtxt, span: Span,
              substr: &Substructure) -> P<Expr> {
    let test_id = cx.ident_of("__test");
    let ordering = cx.path_global(span,
                                  cx.std_path(&["cmp", "Ordering", "Equal"]));
    let ordering = cx.expr_path(ordering);
    let equals_expr = cx.expr_some(span, ordering);

    let partial_cmp_path = cx.std_path(&["cmp", "PartialOrd", "partial_cmp"]);

    /*
    Builds:

    let __test = ::std::cmp::PartialOrd::partial_cmp(&self_field1, &other_field1);
    if __test == ::std::option::Option::Some(::std::cmp::Ordering::Equal) {
        let __test = ::std::cmp::PartialOrd::partial_cmp(&self_field2, &other_field2);
        if __test == ::std::option::Option::Some(::std::cmp::Ordering::Equal) {
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
            // if __test == Some(::std::cmp::Ordering::Equal) {
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

                cx.expr_call_global(span, partial_cmp_path.clone(), args)
            };

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
        Box::new(|cx, span, (self_args, tag_tuple), _non_self_args| {
            if self_args.len() != 2 {
                cx.span_bug(span, "not exactly 2 arguments in `derive(PartialOrd)`")
            } else {
                some_ordering_collapsed(cx, span, PartialCmpOp, tag_tuple)
            }
        }),
        cx, span, substr)
}

/// Strict inequality.
fn cs_op(less: bool, equal: bool, cx: &mut ExtCtxt,
         span: Span, substr: &Substructure) -> P<Expr> {
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
            let other_f = match (other_fs.len(), other_fs.get(0)) {
                (1, Some(o_f)) => o_f,
                _ => cx.span_bug(span, "not exactly 2 arguments in `derive(PartialOrd)`")
            };

            let cmp = cx.expr_binary(span, op, self_f.clone(), other_f.clone());

            let not_cmp = cx.expr_unary(span, ast::UnNot,
                                        cx.expr_binary(span, op, other_f.clone(), self_f));

            let and = cx.expr_binary(span, ast::BiAnd, not_cmp, subexpr);
            cx.expr_binary(span, ast::BiOr, cmp, and)
        },
        cx.expr_bool(span, equal),
        Box::new(|cx, span, (self_args, tag_tuple), _non_self_args| {
            if self_args.len() != 2 {
                cx.span_bug(span, "not exactly 2 arguments in `derive(PartialOrd)`")
            } else {
                let op = match (less, equal) {
                    (true,  true) => LeOp, (true,  false) => LtOp,
                    (false, true) => GeOp, (false, false) => GtOp,
                };
                some_ordering_collapsed(cx, span, op, tag_tuple)
            }
        }),
        cx, span, substr)
}
