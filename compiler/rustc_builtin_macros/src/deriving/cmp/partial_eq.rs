use rustc_ast::ptr::P;
use rustc_ast::{BinOpKind, BorrowKind, Expr, ExprKind, MetaItem, Mutability};
use rustc_expand::base::{Annotatable, ExtCtxt};
use rustc_span::Span;
use rustc_span::symbol::sym;
use thin_vec::thin_vec;

use crate::deriving::generic::ty::*;
use crate::deriving::generic::*;
use crate::deriving::{path_local, path_std};

pub(crate) fn expand_deriving_partial_eq(
    cx: &ExtCtxt<'_>,
    span: Span,
    mitem: &MetaItem,
    item: &Annotatable,
    push: &mut dyn FnMut(Annotatable),
    is_const: bool,
) {
    fn cs_eq(cx: &ExtCtxt<'_>, span: Span, substr: &Substructure<'_>) -> BlockOrExpr {
        let base = true;
        let expr = cs_fold(
            true, // use foldl
            cx,
            span,
            substr,
            |cx, fold| match fold {
                CsFold::Single(field) => {
                    let [other_expr] = &field.other_selflike_exprs[..] else {
                        cx.dcx()
                            .span_bug(field.span, "not exactly 2 arguments in `derive(PartialEq)`");
                    };

                    // We received arguments of type `&T`. Convert them to type `T` by stripping
                    // any leading `&`. This isn't necessary for type checking, but
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
                            expr.clone()
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

    let structural_trait_def = TraitDef {
        span,
        path: path_std!(marker::StructuralPartialEq),
        skip_path_as_bound: true, // crucial!
        needs_copy_as_bound_if_packed: false,
        additional_bounds: Vec::new(),
        // We really don't support unions, but that's already checked by the impl generated below;
        // a second check here would lead to redundant error messages.
        supports_unions: true,
        methods: Vec::new(),
        associated_types: Vec::new(),
        is_const: false,
    };
    structural_trait_def.expand(cx, mitem, item, push);

    // No need to generate `ne`, the default suffices, and not generating it is
    // faster.
    let methods = vec![MethodDef {
        name: sym::eq,
        generics: Bounds::empty(),
        explicit_self: true,
        nonself_args: vec![(self_ref(), sym::other)],
        ret_ty: Path(path_local!(bool)),
        attributes: thin_vec![cx.attr_word(sym::inline, span)],
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
