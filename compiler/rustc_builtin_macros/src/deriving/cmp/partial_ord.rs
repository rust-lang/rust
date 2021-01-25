use crate::deriving::generic::ty::*;
use crate::deriving::generic::*;
use crate::deriving::{path_std, pathvec_std};

use rustc_ast::ptr::P;
use rustc_ast::{Expr, MetaItem};
use rustc_expand::base::{Annotatable, ExtCtxt};
use rustc_span::symbol::{sym, Ident};
use rustc_span::Span;

pub fn expand_deriving_partial_ord(
    cx: &mut ExtCtxt<'_>,
    span: Span,
    mitem: &MetaItem,
    item: &Annotatable,
    push: &mut dyn FnMut(Annotatable),
) {
    let ordering_ty = Literal(path_std!(cmp::Ordering));
    let ret_ty = Literal(Path::new_(
        pathvec_std!(option::Option),
        None,
        vec![Box::new(ordering_ty)],
        PathKind::Std,
    ));

    let inline = cx.meta_word(span, sym::inline);
    let attrs = vec![cx.attribute(inline)];

    let partial_cmp_def = MethodDef {
        name: sym::partial_cmp,
        generics: Bounds::empty(),
        explicit_self: borrowed_explicit_self(),
        args: vec![(borrowed_self(), sym::other)],
        ret_ty,
        attributes: attrs,
        is_unsafe: false,
        unify_fieldless_variants: true,
        combine_substructure: combine_substructure(Box::new(|cx, span, substr| {
            cs_partial_cmp(cx, span, substr)
        })),
    };

    let trait_def = TraitDef {
        span,
        attributes: vec![],
        path: path_std!(cmp::PartialOrd),
        additional_bounds: vec![],
        generics: Bounds::empty(),
        is_unsafe: false,
        supports_unions: false,
        methods: vec![partial_cmp_def],
        associated_types: Vec::new(),
    };
    trait_def.expand(cx, mitem, item, push)
}

pub fn cs_partial_cmp(cx: &mut ExtCtxt<'_>, span: Span, substr: &Substructure<'_>) -> P<Expr> {
    let test_id = Ident::new(sym::cmp, span);
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
    cs_fold(
        // foldr nests the if-elses correctly, leaving the first field
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
                    _ => cx.span_bug(span, "not exactly 2 arguments in `derive(PartialOrd)`"),
                };

                let args =
                    vec![cx.expr_addr_of(span, self_f), cx.expr_addr_of(span, other_f.clone())];

                cx.expr_call_global(span, partial_cmp_path.clone(), args)
            };

            let eq_arm = cx.arm(span, cx.pat_some(span, cx.pat_path(span, ordering.clone())), old);
            let neq_arm = cx.arm(span, cx.pat_ident(span, test_id), cx.expr_ident(span, test_id));

            cx.expr_match(span, new, vec![eq_arm, neq_arm])
        },
        equals_expr,
        Box::new(|cx, span, (self_args, tag_tuple), _non_self_args| {
            if self_args.len() != 2 {
                cx.span_bug(span, "not exactly 2 arguments in `derive(PartialOrd)`")
            } else {
                let lft = cx.expr_ident(span, tag_tuple[0]);
                let rgt = cx.expr_addr_of(span, cx.expr_ident(span, tag_tuple[1]));
                cx.expr_method_call(span, lft, Ident::new(sym::partial_cmp, span), vec![rgt])
            }
        }),
        cx,
        span,
        substr,
    )
}
