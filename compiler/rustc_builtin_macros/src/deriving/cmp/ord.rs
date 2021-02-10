use crate::deriving::generic::ty::*;
use crate::deriving::generic::*;
use crate::deriving::path_std;

use rustc_ast::ptr::P;
use rustc_ast::{self as ast, Expr, MetaItem};
use rustc_expand::base::{Annotatable, ExtCtxt};
use rustc_span::symbol::{sym, Ident};
use rustc_span::Span;

pub fn expand_deriving_ord(
    cx: &mut ExtCtxt<'_>,
    span: Span,
    mitem: &MetaItem,
    item: &Annotatable,
    push: &mut dyn FnMut(Annotatable),
) {
    let inline = cx.meta_word(span, sym::inline);
    let attrs = vec![cx.attribute(inline)];
    let trait_def = TraitDef {
        span,
        attributes: Vec::new(),
        path: path_std!(cmp::Ord),
        additional_bounds: Vec::new(),
        generics: Bounds::empty(),
        is_unsafe: false,
        supports_unions: false,
        methods: vec![MethodDef {
            name: sym::cmp,
            generics: Bounds::empty(),
            explicit_self: borrowed_explicit_self(),
            args: vec![(borrowed_self(), sym::other)],
            ret_ty: Literal(path_std!(cmp::Ordering)),
            attributes: attrs,
            is_unsafe: false,
            unify_fieldless_variants: true,
            combine_substructure: combine_substructure(Box::new(|a, b, c| cs_cmp(a, b, c))),
        }],
        associated_types: Vec::new(),
    };

    trait_def.expand(cx, mitem, item, push)
}

pub fn ordering_collapsed(
    cx: &mut ExtCtxt<'_>,
    span: Span,
    self_arg_tags: &[Ident],
) -> P<ast::Expr> {
    let lft = cx.expr_addr_of(span, cx.expr_ident(span, self_arg_tags[0]));
    let rgt = cx.expr_addr_of(span, cx.expr_ident(span, self_arg_tags[1]));
    let fn_cmp_path = cx.std_path(&[sym::cmp, sym::Ord, sym::cmp]);
    cx.expr_call_global(span, fn_cmp_path, vec![lft, rgt])
}

pub fn cs_cmp(cx: &mut ExtCtxt<'_>, span: Span, substr: &Substructure<'_>) -> P<Expr> {
    let test_id = Ident::new(sym::cmp, span);
    let equals_path = cx.path_global(span, cx.std_path(&[sym::cmp, sym::Ordering, sym::Equal]));

    let cmp_path = cx.std_path(&[sym::cmp, sym::Ord, sym::cmp]);

    // Builds:
    //
    // match ::std::cmp::Ord::cmp(&self_field1, &other_field1) {
    // ::std::cmp::Ordering::Equal =>
    // match ::std::cmp::Ord::cmp(&self_field2, &other_field2) {
    // ::std::cmp::Ordering::Equal => {
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
            //     ::std::cmp::Ordering::Equal => old,
            //     cmp => cmp
            // }

            let new = {
                let other_f = match other_fs {
                    [o_f] => o_f,
                    _ => cx.span_bug(span, "not exactly 2 arguments in `derive(Ord)`"),
                };

                let args =
                    vec![cx.expr_addr_of(span, self_f), cx.expr_addr_of(span, other_f.clone())];

                cx.expr_call_global(span, cmp_path.clone(), args)
            };

            let eq_arm = cx.arm(span, cx.pat_path(span, equals_path.clone()), old);
            let neq_arm = cx.arm(span, cx.pat_ident(span, test_id), cx.expr_ident(span, test_id));

            cx.expr_match(span, new, vec![eq_arm, neq_arm])
        },
        cx.expr_path(equals_path.clone()),
        Box::new(|cx, span, (self_args, tag_tuple), _non_self_args| {
            if self_args.len() != 2 {
                cx.span_bug(span, "not exactly 2 arguments in `derive(Ord)`")
            } else {
                ordering_collapsed(cx, span, tag_tuple)
            }
        }),
        cx,
        span,
        substr,
    )
}
