use crate::deriving::generic::ty::*;
use crate::deriving::generic::*;
use crate::deriving::path_std;
use rustc_ast::MetaItem;
use rustc_expand::base::{Annotatable, ExtCtxt};
use rustc_span::symbol::{sym, Ident};
use rustc_span::Span;
use thin_vec::thin_vec;

pub fn expand_deriving_ord(
    cx: &mut ExtCtxt<'_>,
    span: Span,
    mitem: &MetaItem,
    item: &Annotatable,
    push: &mut dyn FnMut(Annotatable),
    is_const: bool,
) {
    let attrs = thin_vec![cx.attr_word(sym::inline, span)];
    let trait_def = TraitDef {
        span,
        path: path_std!(cmp::Ord),
        skip_path_as_bound: false,
        needs_copy_as_bound_if_packed: true,
        additional_bounds: Vec::new(),
        supports_unions: false,
        methods: vec![MethodDef {
            name: sym::cmp,
            generics: Bounds::empty(),
            explicit_self: true,
            nonself_args: vec![(self_ref(), sym::other)],
            ret_ty: Path(path_std!(cmp::Ordering)),
            attributes: attrs,
            fieldless_variants_strategy: FieldlessVariantsStrategy::Unify,
            combine_substructure: combine_substructure(Box::new(|a, b, c| cs_cmp(a, b, c))),
        }],
        associated_types: Vec::new(),
        is_const,
    };

    trait_def.expand(cx, mitem, item, push)
}

pub fn cs_cmp(cx: &mut ExtCtxt<'_>, span: Span, substr: &Substructure<'_>) -> BlockOrExpr {
    let test_id = Ident::new(sym::cmp, span);
    let equal_path = cx.path_global(span, cx.std_path(&[sym::cmp, sym::Ordering, sym::Equal]));
    let cmp_path = cx.std_path(&[sym::cmp, sym::Ord, sym::cmp]);

    // Builds:
    //
    // match ::core::cmp::Ord::cmp(&self.x, &other.x) {
    //     ::std::cmp::Ordering::Equal =>
    //         ::core::cmp::Ord::cmp(&self.y, &other.y),
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
                let args = thin_vec![field.self_expr.clone(), other_expr.clone()];
                cx.expr_call_global(field.span, cmp_path.clone(), args)
            }
            CsFold::Combine(span, expr1, expr2) => {
                let eq_arm = cx.arm(span, cx.pat_path(span, equal_path.clone()), expr1);
                let neq_arm =
                    cx.arm(span, cx.pat_ident(span, test_id), cx.expr_ident(span, test_id));
                cx.expr_match(span, expr2, thin_vec![eq_arm, neq_arm])
            }
            CsFold::Fieldless => cx.expr_path(equal_path.clone()),
        },
    );
    BlockOrExpr::new_expr(expr)
}
