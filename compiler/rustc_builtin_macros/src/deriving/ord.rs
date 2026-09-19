use rustc_ast::Safety;
use rustc_expand::base::ExtCtxt;
use rustc_span::{Ident, Span, sym};
use thin_vec::thin_vec;

use crate::deriving::generic::ty::*;
use crate::deriving::generic::*;
use crate::deriving::partial_ord::discr_data_order;
use crate::deriving::path_std;

pub(crate) fn expand_deriving_ord(
    cx: &ExtCtxt<'_>,
    span: Span,
    item: &ast::Item,
    push: &mut dyn FnMut(Box<ast::Item>),
    is_const: bool,
) {
    let discr_then_data = discr_data_order(item);

    let trait_def = TraitDef {
        span,
        path: path_std!(cx, span, cmp::Ord),
        skip_path_as_bound: false,
        needs_copy_as_bound_if_packed: true,
        additional_bounds: SmallVec::new(),
        supports_unions: false,
        methods: smallvec![MethodDef {
            name: sym::cmp,
            generics: cx.empty_generics(span),
            explicit_self: true,
            nonself_args: smallvec![(self_ref(), sym::other)],
            ret_ty: Path(path_std!(cx, span, cmp::Ordering)),
            attributes: thin_vec![cx.attr_word(sym::inline, span)],
            fieldless_variants_strategy: FieldlessVariantsStrategy::Unify,
            combine_substructure: combine_substructure(|cx, span, substr| cs_cmp(
                cx,
                span,
                substr,
                discr_then_data
            )),
        }],
        associated_types: SmallVec::new(),
        is_const,
        safety: Safety::Default,
        document: true,
    };

    trait_def.expand(cx, item, push)
}

pub(crate) fn cs_cmp(
    cx: &ExtCtxt<'_>,
    span: Span,
    substr: Substructure<'_>,
    discr_then_data: bool,
) -> BlockOrExpr {
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
    let expr = cs_foldr(
        cx,
        span,
        substr,
        |field| {
            let other_expr =
                field.other_selflike_expr.expect("not exactly 2 arguments in `derive(Ord)`");
            let args = thin_vec![field.self_expr, other_expr];
            cx.expr_call_global(field.span, cmp_path.clone(), args)
        },
        |span, mut expr1, expr2| {
            if !discr_then_data
                && let ast::ExprKind::Match(_, arms, _) = &mut expr1.kind
                && let Some(last) = arms.last_mut()
                && let ast::PatKind::Wild = last.pat.kind
            {
                last.body = Some(expr2);
                expr1
            } else {
                let eq_arm = cx.arm(span, cx.pat_path(span, equal_path.clone()), expr1);
                let neq_arm =
                    cx.arm(span, cx.pat_ident(span, test_id), cx.expr_ident(span, test_id));
                cx.expr_match(span, expr2, thin_vec![eq_arm, neq_arm])
            }
        },
        || cx.expr_path(equal_path.clone()),
    );
    BlockOrExpr::new_expr(expr)
}
