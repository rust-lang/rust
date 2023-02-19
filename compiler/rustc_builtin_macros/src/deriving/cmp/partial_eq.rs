use crate::deriving::generic::ty::*;
use crate::deriving::generic::*;
use crate::deriving::{path_local, path_std};
use rustc_ast::ptr::P;
use rustc_ast::{BinOpKind, BorrowKind, Expr, ExprKind, MetaItem, Mutability};
use rustc_expand::base::{Annotatable, ExtCtxt};
use rustc_span::symbol::sym;
use rustc_span::Span;
use thin_vec::thin_vec;

pub fn expand_deriving_partial_eq(
    cx: &mut ExtCtxt<'_>,
    span: Span,
    mitem: &MetaItem,
    item: &Annotatable,
    push: &mut dyn FnMut(Annotatable),
    is_const: bool,
) {
    fn cs_eq(cx: &mut ExtCtxt<'_>, span: Span, substr: &Substructure<'_>) -> BlockOrExpr {
        let base = true;
        let expr = cs_fold(
            true, // use foldl
            cx,
            span,
            substr,
            |cx, fold| match fold {
                CsFold::Single(field) => {
                    let [other_expr] = &field.other_selflike_exprs[..] else {
                        cx.span_bug(field.span, "not exactly 2 arguments in `derive(PartialEq)`");
                    };

                    // We received arguments of type `&T`. Convert them to type `T` by stripping
                    // any leading `&` or adding `*`. This isn't necessary for type checking, but
                    // it results in better error messages if something goes wrong.
                    //
                    // Note: for arguments that look like `&{ x }`, which occur with packed
                    // structs, this would cause expressions like `{ self.x } == { other.x }`,
                    // which isn't valid Rust syntax. This wouldn't break compilation because these
                    // AST nodes are constructed within the compiler. But it would mean that code
                    // printed by `-Zunpretty=expanded` (or `cargo expand`) would have invalid
                    // syntax, which would be suboptimal. So we wrap these in parens, giving
                    // `({ self.x }) == ({ other.x })`, which is valid syntax.
                    let convert = |expr: &P<Expr>| {
                        if let ExprKind::AddrOf(BorrowKind::Ref, Mutability::Not, inner) =
                            &expr.kind
                        {
                            if let ExprKind::Block(..) = &inner.kind {
                                // `&{ x }` form: remove the `&`, add parens.
                                cx.expr_paren(field.span, inner.clone())
                            } else {
                                // `&x` form: remove the `&`.
                                inner.clone()
                            }
                        } else {
                            // No leading `&`: add a leading `*`.
                            cx.expr_deref(field.span, expr.clone())
                        }
                    };
                    cx.expr_binary(
                        field.span,
                        BinOpKind::Eq,
                        convert(&field.self_expr),
                        convert(other_expr),
                    )
                }
                CsFold::Combine(span, expr1, expr2) => {
                    cx.expr_binary(span, BinOpKind::And, expr1, expr2)
                }
                CsFold::Fieldless => cx.expr_bool(span, base),
            },
        );
        BlockOrExpr::new_expr(expr)
    }

    super::inject_impl_of_structural_trait(
        cx,
        span,
        item,
        path_std!(marker::StructuralPartialEq),
        push,
    );

    // No need to generate `ne`, the default suffices, and not generating it is
    // faster.
    let attrs = thin_vec![cx.attr_word(sym::inline, span)];
    let methods = vec![MethodDef {
        name: sym::eq,
        generics: Bounds::empty(),
        explicit_self: true,
        nonself_args: vec![(self_ref(), sym::other)],
        ret_ty: Path(path_local!(bool)),
        attributes: attrs,
        fieldless_variants_strategy: FieldlessVariantsStrategy::Unify,
        combine_substructure: combine_substructure(Box::new(|a, b, c| cs_eq(a, b, c))),
    }];

    let trait_def = TraitDef {
        span,
        path: path_std!(cmp::PartialEq),
        skip_path_as_bound: false,
        needs_copy_as_bound_if_packed: true,
        additional_bounds: Vec::new(),
        supports_unions: false,
        methods,
        associated_types: Vec::new(),
        is_const,
    };
    trait_def.expand(cx, mitem, item, push)
}
