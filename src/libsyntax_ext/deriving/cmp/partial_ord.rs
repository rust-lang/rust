pub use OrderingOp::*;

use crate::deriving::{path_local, pathvec_std, path_std};
use crate::deriving::generic::*;
use crate::deriving::generic::ty::*;

use syntax::ast::{self, BinOpKind, Expr, MetaItem};
use syntax::ext::base::{Annotatable, ExtCtxt};
use syntax::ext::build::AstBuilder;
use syntax::ptr::P;
use syntax::symbol::{sym, Symbol};
use syntax_pos::Span;

pub fn expand_deriving_partial_ord(cx: &mut ExtCtxt<'_>,
                                   span: Span,
                                   mitem: &MetaItem,
                                   item: &Annotatable,
                                   push: &mut dyn FnMut(Annotatable)) {
    macro_rules! md {
        ($name:expr, $op:expr, $equal:expr) => { {
            let inline = cx.meta_word(span, sym::inline);
            let attrs = vec![cx.attribute(span, inline)];
            MethodDef {
                name: $name,
                generics: LifetimeBounds::empty(),
                explicit_self: borrowed_explicit_self(),
                args: vec![(borrowed_self(), "other")],
                ret_ty: Literal(path_local!(bool)),
                attributes: attrs,
                is_unsafe: false,
                unify_fieldless_variants: true,
                combine_substructure: combine_substructure(Box::new(|cx, span, substr| {
                    cs_op($op, $equal, cx, span, substr)
                }))
            }
        } }
    }

    let ordering_ty = Literal(path_std!(cx, cmp::Ordering));
    let ret_ty = Literal(Path::new_(pathvec_std!(cx, option::Option),
                                    None,
                                    vec![Box::new(ordering_ty)],
                                    PathKind::Std));

    let inline = cx.meta_word(span, sym::inline);
    let attrs = vec![cx.attribute(span, inline)];

    let partial_cmp_def = MethodDef {
        name: "partial_cmp",
        generics: LifetimeBounds::empty(),
        explicit_self: borrowed_explicit_self(),
        args: vec![(borrowed_self(), "other")],
        ret_ty,
        attributes: attrs,
        is_unsafe: false,
        unify_fieldless_variants: true,
        combine_substructure: combine_substructure(Box::new(|cx, span, substr| {
            cs_partial_cmp(cx, span, substr)
        })),
    };

    // avoid defining extra methods if we can
    // c-like enums, enums without any fields and structs without fields
    // can safely define only `partial_cmp`.
    let methods = if is_type_without_fields(item) {
        vec![partial_cmp_def]
    } else {
        vec![partial_cmp_def,
             md!("lt", true, false),
             md!("le", true, true),
             md!("gt", false, false),
             md!("ge", false, true)]
    };

    let trait_def = TraitDef {
        span,
        attributes: vec![],
        path: path_std!(cx, cmp::PartialOrd),
        additional_bounds: vec![],
        generics: LifetimeBounds::empty(),
        is_unsafe: false,
        supports_unions: false,
        methods,
        associated_types: Vec::new(),
    };
    trait_def.expand(cx, mitem, item, push)
}

#[derive(Copy, Clone)]
pub enum OrderingOp {
    PartialCmpOp,
    LtOp,
    LeOp,
    GtOp,
    GeOp,
}

pub fn some_ordering_collapsed(cx: &mut ExtCtxt<'_>,
                               span: Span,
                               op: OrderingOp,
                               self_arg_tags: &[ast::Ident])
                               -> P<ast::Expr> {
    let lft = cx.expr_ident(span, self_arg_tags[0]);
    let rgt = cx.expr_addr_of(span, cx.expr_ident(span, self_arg_tags[1]));
    let op_str = match op {
        PartialCmpOp => "partial_cmp",
        LtOp => "lt",
        LeOp => "le",
        GtOp => "gt",
        GeOp => "ge",
    };
    cx.expr_method_call(span, lft, cx.ident_of(op_str), vec![rgt])
}

pub fn cs_partial_cmp(cx: &mut ExtCtxt<'_>, span: Span, substr: &Substructure<'_>) -> P<Expr> {
    let test_id = cx.ident_of("cmp").gensym();
    let ordering = cx.path_global(span, cx.std_path(&[sym::cmp, sym::Ordering, sym::Equal]));
    let ordering_expr = cx.expr_path(ordering.clone());
    let equals_expr = cx.expr_some(span, ordering_expr);

    let partial_cmp_path = cx.std_path(&[sym::cmp, sym::PartialOrd, sym::partial_cmp]);

    // Builds:
    //
    // match ::std::cmp::PartialOrd::partial_cmp(&self_field1, &other_field1) {
    // ::std::option::Option::Some(::std::cmp::Ordering::Equal) =>
    // match ::std::cmp::PartialOrd::partial_cmp(&self_field2, &other_field2) {
    // ::std::option::Option::Some(::std::cmp::Ordering::Equal) => {
    // ...
    // }
    // cmp => cmp
    // },
    // cmp => cmp
    // }
    //
    cs_fold(// foldr nests the if-elses correctly, leaving the first field
            // as the outermost one, and the last as the innermost.
            false,
            |cx, span, old, self_f, other_fs| {
                // match new {
                //     Some(::std::cmp::Ordering::Equal) => old,
                //     cmp => cmp
                // }

                let new = {
                    let other_f = match other_fs {
                        [o_f] => o_f,
                                _ => {
                                    cx.span_bug(span,
                                        "not exactly 2 arguments in `derive(PartialOrd)`")
                                }
                    };

                    let args = vec![
                            cx.expr_addr_of(span, self_f),
                            cx.expr_addr_of(span, other_f.clone()),
                        ];

                    cx.expr_call_global(span, partial_cmp_path.clone(), args)
                };

                let eq_arm = cx.arm(span,
                                    vec![cx.pat_some(span, cx.pat_path(span, ordering.clone()))],
                                    old);
                let neq_arm = cx.arm(span,
                                    vec![cx.pat_ident(span, test_id)],
                                    cx.expr_ident(span, test_id));

                cx.expr_match(span, new, vec![eq_arm, neq_arm])
            },
            equals_expr,
            Box::new(|cx, span, (self_args, tag_tuple), _non_self_args| {
        if self_args.len() != 2 {
            cx.span_bug(span, "not exactly 2 arguments in `derive(PartialOrd)`")
        } else {
            some_ordering_collapsed(cx, span, PartialCmpOp, tag_tuple)
        }
    }),
            cx,
            span,
            substr)
}

/// Strict inequality.
fn cs_op(less: bool,
         inclusive: bool,
         cx: &mut ExtCtxt<'_>,
         span: Span,
         substr: &Substructure<'_>) -> P<Expr> {
    let ordering_path = |cx: &mut ExtCtxt<'_>, name: &str| {
        cx.expr_path(cx.path_global(
            span, cx.std_path(&[sym::cmp, sym::Ordering, Symbol::intern(name)])))
    };

    let par_cmp = |cx: &mut ExtCtxt<'_>, span, self_f: P<Expr>, other_fs: &[P<Expr>], default| {
        let other_f = match other_fs {
            [o_f] => o_f,
            _ => cx.span_bug(span, "not exactly 2 arguments in `derive(PartialOrd)`"),
        };

        // `PartialOrd::partial_cmp(self.fi, other.fi)`
        let cmp_path = cx.expr_path(cx.path_global(span, cx.std_path(&[sym::cmp,
                                                                       sym::PartialOrd,
                                                                       sym::partial_cmp])));
        let cmp = cx.expr_call(span,
                               cmp_path,
                               vec![cx.expr_addr_of(span, self_f),
                                    cx.expr_addr_of(span, other_f.clone())]);

        let default = ordering_path(cx, default);
        // `Option::unwrap_or(_, Ordering::Equal)`
        let unwrap_path = cx.expr_path(cx.path_global(span, cx.std_path(&[sym::option,
                                                                          sym::Option,
                                                                          sym::unwrap_or])));
        cx.expr_call(span, unwrap_path, vec![cmp, default])
    };

    let fold = cs_fold1(false, // need foldr
        |cx, span, subexpr, self_f, other_fs| {
            // build up a series of `partial_cmp`s from the inside
            // out (hence foldr) to get lexical ordering, i.e., for op ==
            // `ast::lt`
            //
            // ```
            // Ordering::then_with(
            //    Option::unwrap_or(
            //        PartialOrd::partial_cmp(self.f1, other.f1), Ordering::Equal)
            //    ),
            //    Option::unwrap_or(
            //        PartialOrd::partial_cmp(self.f2, other.f2), Ordering::Greater)
            //    )
            // )
            // == Ordering::Less
            // ```
            //
            // and for op ==
            // `ast::le`
            //
            // ```
            // Ordering::then_with(
            //    Option::unwrap_or(
            //        PartialOrd::partial_cmp(self.f1, other.f1), Ordering::Equal)
            //    ),
            //    Option::unwrap_or(
            //        PartialOrd::partial_cmp(self.f2, other.f2), Ordering::Greater)
            //    )
            // )
            // != Ordering::Greater
            // ```
            //
            // The optimiser should remove the redundancy. We explicitly
            // get use the binops to avoid auto-deref dereferencing too many
            // layers of pointers, if the type includes pointers.

            // `Option::unwrap_or(PartialOrd::partial_cmp(self.fi, other.fi), Ordering::Equal)`
            let par_cmp = par_cmp(cx, span, self_f, other_fs, "Equal");

            // `Ordering::then_with(Option::unwrap_or(..), ..)`
            let then_with_path = cx.expr_path(cx.path_global(span,
                                                             cx.std_path(&[sym::cmp,
                                                                           sym::Ordering,
                                                                           sym::then_with])));
            cx.expr_call(span, then_with_path, vec![par_cmp, cx.lambda0(span, subexpr)])
        },
        |cx, args| {
            match args {
                Some((span, self_f, other_fs)) => {
                    let opposite = if less { "Greater" } else { "Less" };
                    par_cmp(cx, span, self_f, other_fs, opposite)
                },
                None => cx.expr_bool(span, inclusive)
            }
        },
        Box::new(|cx, span, (self_args, tag_tuple), _non_self_args| {
            if self_args.len() != 2 {
                cx.span_bug(span, "not exactly 2 arguments in `derive(PartialOrd)`")
            } else {
                let op = match (less, inclusive) {
                    (false, false) => GtOp,
                    (false, true) => GeOp,
                    (true, false) => LtOp,
                    (true, true) => LeOp,
                };
                some_ordering_collapsed(cx, span, op, tag_tuple)
            }
        }),
        cx,
        span,
        substr);

    match *substr.fields {
        EnumMatching(.., ref all_fields) |
        Struct(.., ref all_fields) if !all_fields.is_empty() => {
            let ordering = ordering_path(cx, if less ^ inclusive { "Less" } else { "Greater" });
            let comp_op = if inclusive { BinOpKind::Ne } else { BinOpKind::Eq };

            cx.expr_binary(span, comp_op, fold, ordering)
        }
        _ => fold
    }
}
