use crate::deriving::{pathvec_std, path_std};
use crate::deriving::generic::*;
use crate::deriving::generic::ty::*;

use syntax::ast::{self, Expr, MetaItem};
use syntax::ext::base::{Annotatable, ExtCtxt};
use syntax::ptr::P;
use syntax::symbol::sym;
use syntax_pos::Span;

pub fn expand_deriving_partial_ord(cx: &mut ExtCtxt<'_>,
                                   span: Span,
                                   mitem: &MetaItem,
                                   item: &Annotatable,
                                   push: &mut dyn FnMut(Annotatable)) {
    let ordering_ty = Literal(path_std!(cx, cmp::Ordering));
    let ret_ty = Literal(Path::new_(pathvec_std!(cx, option::Option),
                                    None,
                                    vec![Box::new(ordering_ty)],
                                    PathKind::Std));

    let inline = cx.meta_word(span, sym::inline);
    let attrs = vec![cx.attribute(inline)];

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

    let methods = vec![partial_cmp_def];

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

pub fn some_ordering_collapsed(
    cx: &mut ExtCtxt<'_>,
    span: Span,
    self_arg_tags: &[ast::Ident],
) -> P<ast::Expr> {
    let lft = cx.expr_ident(span, self_arg_tags[0]);
    let rgt = cx.expr_addr_of(span, cx.expr_ident(span, self_arg_tags[1]));
    cx.expr_method_call(span, lft, ast::Ident::from_str_and_span("partial_cmp", span), vec![rgt])
}

pub fn cs_partial_cmp(cx: &mut ExtCtxt<'_>, span: Span, substr: &Substructure<'_>) -> P<Expr> {
    let test_id = ast::Ident::new(sym::cmp, span);
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
            some_ordering_collapsed(cx, span, tag_tuple)
        }
    }),
            cx,
            span,
            substr)
}
