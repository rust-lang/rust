use crate::deriving::generic::ty::*;
use crate::deriving::generic::*;
use crate::deriving::{path_std, pathvec_std};
use rustc_ast::MetaItem;
use rustc_expand::base::{Annotatable, ExtCtxt};
use rustc_span::symbol::{sym, Ident};
use rustc_span::Span;
use thin_vec::thin_vec;

pub fn expand_deriving_partial_ord(
    cx: &mut ExtCtxt<'_>,
    span: Span,
    mitem: &MetaItem,
    item: &Annotatable,
    push: &mut dyn FnMut(Annotatable),
) {
    let ordering_ty = Path(path_std!(cmp::Ordering));
    let ret_ty =
        Path(Path::new_(pathvec_std!(option::Option), vec![Box::new(ordering_ty)], PathKind::Std));

    let inline = cx.meta_word(span, sym::inline);
    let attrs = thin_vec![cx.attribute(inline)];

    let partial_cmp_def = MethodDef {
        name: sym::partial_cmp,
        generics: Bounds::empty(),
        explicit_self: true,
        nonself_args: vec![(self_ref(), sym::other)],
        ret_ty,
        attributes: attrs,
        unify_fieldless_variants: true,
        combine_substructure: combine_substructure(Box::new(|cx, span, substr| {
            cs_partial_cmp(cx, span, substr)
        })),
    };

    let trait_def = TraitDef {
        span,
        path: path_std!(cmp::PartialOrd),
        additional_bounds: vec![],
        generics: Bounds::empty(),
        supports_unions: false,
        methods: vec![partial_cmp_def],
        associated_types: Vec::new(),
    };
    trait_def.expand(cx, mitem, item, push)
}

pub fn cs_partial_cmp(cx: &mut ExtCtxt<'_>, span: Span, substr: &Substructure<'_>) -> BlockOrExpr {
    let test_id = Ident::new(sym::cmp, span);
    let equal_path = cx.path_global(span, cx.std_path(&[sym::cmp, sym::Ordering, sym::Equal]));
    let partial_cmp_path = cx.std_path(&[sym::cmp, sym::PartialOrd, sym::partial_cmp]);

    // Builds:
    //
    // match ::core::cmp::PartialOrd::partial_cmp(&self.x, &other.x) {
    //     ::core::option::Option::Some(::core::cmp::Ordering::Equal) =>
    //         ::core::cmp::PartialOrd::partial_cmp(&self.y, &other.y),
    //     cmp => cmp,
    // }
    let expr = cs_fold(
        // foldr nests the if-elses correctly, leaving the first field
        // as the outermost one, and the last as the innermost.
        false,
        cx,
        span,
        substr,
        |cx, fold| match fold {
            CsFold::Single(field) => {
                let [other_expr] = &field.other_selflike_exprs[..] else {
                        cx.span_bug(field.span, "not exactly 2 arguments in `derive(Ord)`");
                    };
                let args = vec![field.self_expr.clone(), other_expr.clone()];
                cx.expr_call_global(field.span, partial_cmp_path.clone(), args)
            }
            CsFold::Combine(span, expr1, expr2) => {
                let eq_arm =
                    cx.arm(span, cx.pat_some(span, cx.pat_path(span, equal_path.clone())), expr1);
                let neq_arm =
                    cx.arm(span, cx.pat_ident(span, test_id), cx.expr_ident(span, test_id));
                cx.expr_match(span, expr2, vec![eq_arm, neq_arm])
            }
            CsFold::Fieldless => cx.expr_some(span, cx.expr_path(equal_path.clone())),
        },
    );
    BlockOrExpr::new_expr(expr)
}
