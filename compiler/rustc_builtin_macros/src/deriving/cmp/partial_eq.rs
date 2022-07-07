use crate::deriving::generic::ty::*;
use crate::deriving::generic::*;
use crate::deriving::{path_local, path_std};

use rustc_ast::ptr::P;
use rustc_ast::{BinOpKind, BorrowKind, Expr, ExprKind, MetaItem, Mutability};
use rustc_expand::base::{Annotatable, ExtCtxt};
use rustc_span::symbol::sym;
use rustc_span::Span;

pub fn expand_deriving_partial_eq(
    cx: &mut ExtCtxt<'_>,
    span: Span,
    mitem: &MetaItem,
    item: &Annotatable,
    push: &mut dyn FnMut(Annotatable),
) {
    fn cs_op(
        cx: &mut ExtCtxt<'_>,
        span: Span,
        substr: &Substructure<'_>,
        op: BinOpKind,
        combiner: BinOpKind,
        base: bool,
    ) -> BlockOrExpr {
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

                    // We received `&T` arguments. Convert them to `T` by
                    // stripping `&` or adding `*`. This isn't necessary for
                    // type checking, but it results in much better error
                    // messages if something goes wrong.
                    let convert = |expr: &P<Expr>| {
                        if let ExprKind::AddrOf(BorrowKind::Ref, Mutability::Not, inner) =
                            &expr.kind
                        {
                            inner.clone()
                        } else {
                            cx.expr_deref(field.span, expr.clone())
                        }
                    };
                    cx.expr_binary(field.span, op, convert(&field.self_expr), convert(other_expr))
                }
                CsFold::Combine(span, expr1, expr2) => cx.expr_binary(span, combiner, expr1, expr2),
                CsFold::Fieldless => cx.expr_bool(span, base),
                CsFold::EnumNonMatching(span, _tag_tuple) => cx.expr_bool(span, !base),
            },
        );
        BlockOrExpr::new_expr(expr)
    }

    fn cs_eq(cx: &mut ExtCtxt<'_>, span: Span, substr: &Substructure<'_>) -> BlockOrExpr {
        cs_op(cx, span, substr, BinOpKind::Eq, BinOpKind::And, true)
    }
    fn cs_ne(cx: &mut ExtCtxt<'_>, span: Span, substr: &Substructure<'_>) -> BlockOrExpr {
        cs_op(cx, span, substr, BinOpKind::Ne, BinOpKind::Or, false)
    }

    macro_rules! md {
        ($name:expr, $f:ident) => {{
            let inline = cx.meta_word(span, sym::inline);
            let attrs = vec![cx.attribute(inline)];
            MethodDef {
                name: $name,
                generics: Bounds::empty(),
                explicit_self: true,
                nonself_args: vec![(self_ref(), sym::other)],
                ret_ty: Path(path_local!(bool)),
                attributes: attrs,
                unify_fieldless_variants: true,
                combine_substructure: combine_substructure(Box::new(|a, b, c| $f(a, b, c))),
            }
        }};
    }

    super::inject_impl_of_structural_trait(
        cx,
        span,
        item,
        path_std!(marker::StructuralPartialEq),
        push,
    );

    // avoid defining `ne` if we can
    // c-like enums, enums without any fields and structs without fields
    // can safely define only `eq`.
    let mut methods = vec![md!(sym::eq, cs_eq)];
    if !is_type_without_fields(item) {
        methods.push(md!(sym::ne, cs_ne));
    }

    let trait_def = TraitDef {
        span,
        attributes: Vec::new(),
        path: path_std!(cmp::PartialEq),
        additional_bounds: Vec::new(),
        generics: Bounds::empty(),
        supports_unions: false,
        methods,
        associated_types: Vec::new(),
    };
    trait_def.expand(cx, mitem, item, push)
}
