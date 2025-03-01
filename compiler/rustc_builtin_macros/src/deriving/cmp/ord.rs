use rustc_ast::ptr::P;
use rustc_ast::{BinOpKind, BorrowKind, Expr, ExprKind, MetaItem, Mutability};
use rustc_expand::base::{Annotatable, ExtCtxt};
use rustc_span::{Ident, Span, sym};
use thin_vec::thin_vec;

use crate::deriving::generic::ty::*;
use crate::deriving::generic::*;
use crate::deriving::path_std;

pub(crate) fn expand_deriving_ord(
    cx: &ExtCtxt<'_>,
    span: Span,
    mitem: &MetaItem,
    item: &Annotatable,
    push: &mut dyn FnMut(Annotatable),
    is_const: bool,
) {
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
            attributes: thin_vec![cx.attr_word(sym::inline, span)],
            fieldless_variants_strategy: FieldlessVariantsStrategy::Unify,
            combine_substructure: combine_substructure(Box::new(|a, b, c| cs_cmp(a, b, c))),
        }],
        associated_types: Vec::new(),
        is_const,
    };

    trait_def.expand(cx, mitem, item, push)
}

pub(crate) fn cs_cmp(cx: &ExtCtxt<'_>, span: Span, substr: &Substructure<'_>) -> BlockOrExpr {
    let test_id = Ident::new(sym::cmp, span);
    let equal_path = cx.path_global(span, cx.std_path(&[sym::cmp, sym::Ordering, sym::Equal]));

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
                    cx.dcx().span_bug(field.span, "not exactly 2 arguments in `derive(Ord)`");
                };
                let convert = |expr: &P<Expr>| {
                    if let ExprKind::AddrOf(BorrowKind::Ref, Mutability::Not, inner) = &expr.kind {
                        if let ExprKind::Block(..) = &inner.kind {
                            // `&{ x }` form: remove the `&`, add parens.
                            cx.expr_paren(field.span, inner.clone())
                        } else {
                            // `&x` form: remove the `&`.
                            inner.clone()
                        }
                    } else {
                        cx.expr_deref(field.span, expr.clone())
                    }
                };

                let lhs = convert(&field.self_expr);
                let rhs = convert(&other_expr);
                cx.expr_binary(field.span, BinOpKind::Cmp, lhs, rhs)
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
