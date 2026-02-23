use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::res::{MaybeDef, MaybeQPath};
use clippy_utils::sugg::{Sugg, has_enclosing_paren};
use clippy_utils::{SpanlessEq, sym};
use rustc_ast::{BinOpKind, LitIntType, LitKind, UnOp};
use rustc_data_structures::packed::Pu128;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::LateContext;
use rustc_middle::ty::{self, Ty};
use rustc_span::source_map::Spanned;

use super::MANUAL_DIV_CEIL;

pub(super) fn check(cx: &LateContext<'_>, expr: &Expr<'_>, op: BinOpKind, lhs: &Expr<'_>, rhs: &Expr<'_>, msrv: Msrv) {
    let mut applicability = Applicability::MachineApplicable;

    if op == BinOpKind::Div
        && check_int_ty_and_feature(cx, cx.typeck_results().expr_ty(lhs))
        && check_int_ty_and_feature(cx, cx.typeck_results().expr_ty(rhs))
        && msrv.meets(cx, msrvs::DIV_CEIL)
    {
        match lhs.kind {
            ExprKind::Binary(inner_op, inner_lhs, inner_rhs) => {
                // (x + (y - 1)) / y
                if let ExprKind::Binary(sub_op, sub_lhs, sub_rhs) = inner_rhs.kind
                    && inner_op.node == BinOpKind::Add
                    && sub_op.node == BinOpKind::Sub
                    && check_literal(sub_rhs)
                    && check_eq_expr(cx, sub_lhs, rhs)
                {
                    build_suggestion(cx, expr, inner_lhs, rhs, &mut applicability);
                    return;
                }

                // ((y - 1) + x) / y
                if let ExprKind::Binary(sub_op, sub_lhs, sub_rhs) = inner_lhs.kind
                    && inner_op.node == BinOpKind::Add
                    && sub_op.node == BinOpKind::Sub
                    && check_literal(sub_rhs)
                    && check_eq_expr(cx, sub_lhs, rhs)
                {
                    build_suggestion(cx, expr, inner_rhs, rhs, &mut applicability);
                    return;
                }

                // (x + y - 1) / y
                if let ExprKind::Binary(add_op, add_lhs, add_rhs) = inner_lhs.kind
                    && inner_op.node == BinOpKind::Sub
                    && add_op.node == BinOpKind::Add
                    && check_literal(inner_rhs)
                    && check_eq_expr(cx, add_rhs, rhs)
                {
                    build_suggestion(cx, expr, add_lhs, rhs, &mut applicability);
                }

                // (x + (Y - 1)) / Y
                if inner_op.node == BinOpKind::Add && differ_by_one(inner_rhs, rhs) {
                    build_suggestion(cx, expr, inner_lhs, rhs, &mut applicability);
                }

                // ((Y - 1) + x) / Y
                if inner_op.node == BinOpKind::Add && differ_by_one(inner_lhs, rhs) {
                    build_suggestion(cx, expr, inner_rhs, rhs, &mut applicability);
                }

                // (x - (-Y - 1)) / Y
                if inner_op.node == BinOpKind::Sub
                    && let ExprKind::Unary(UnOp::Neg, abs_div_rhs) = rhs.kind
                    && differ_by_one(abs_div_rhs, inner_rhs)
                {
                    build_suggestion(cx, expr, inner_lhs, rhs, &mut applicability);
                }
            },
            ExprKind::MethodCall(method, receiver, [next_multiple_of_arg], _) => {
                // x.next_multiple_of(Y) / Y
                if method.ident.name == sym::next_multiple_of
                    && check_int_ty(cx.typeck_results().expr_ty(receiver))
                    && check_eq_expr(cx, next_multiple_of_arg, rhs)
                {
                    build_suggestion(cx, expr, receiver, rhs, &mut applicability);
                }
            },
            ExprKind::Call(callee, [receiver, next_multiple_of_arg]) => {
                // int_type::next_multiple_of(x, Y) / Y
                if let Some(impl_ty_binder) = callee
                    .ty_rel_def_if_named(cx, sym::next_multiple_of)
                    .assoc_fn_parent(cx)
                    .opt_impl_ty(cx)
                    && check_int_ty(impl_ty_binder.skip_binder())
                    && check_eq_expr(cx, next_multiple_of_arg, rhs)
                {
                    build_suggestion(cx, expr, receiver, rhs, &mut applicability);
                }
            },
            _ => (),
        }
    }
}

/// Checks if two expressions represent non-zero integer literals such that `small_expr + 1 ==
/// large_expr`.
fn differ_by_one(small_expr: &Expr<'_>, large_expr: &Expr<'_>) -> bool {
    if let ExprKind::Lit(small) = small_expr.kind
        && let ExprKind::Lit(large) = large_expr.kind
        && let LitKind::Int(s, _) = small.node
        && let LitKind::Int(l, _) = large.node
    {
        Some(l.get()) == s.get().checked_add(1)
    } else if let ExprKind::Unary(UnOp::Neg, small_inner_expr) = small_expr.kind
        && let ExprKind::Unary(UnOp::Neg, large_inner_expr) = large_expr.kind
    {
        differ_by_one(large_inner_expr, small_inner_expr)
    } else {
        false
    }
}

fn check_int_ty(expr_ty: Ty<'_>) -> bool {
    matches!(expr_ty.peel_refs().kind(), ty::Int(_) | ty::Uint(_))
}

fn check_int_ty_and_feature(cx: &LateContext<'_>, expr_ty: Ty<'_>) -> bool {
    match expr_ty.peel_refs().kind() {
        ty::Uint(_) => true,
        ty::Int(_) => cx.tcx.features().enabled(sym::int_roundings),
        _ => false,
    }
}

fn check_literal(expr: &Expr<'_>) -> bool {
    if let ExprKind::Lit(lit) = expr.kind
        && let LitKind::Int(Pu128(1), _) = lit.node
    {
        return true;
    }
    false
}

fn check_eq_expr(cx: &LateContext<'_>, lhs: &Expr<'_>, rhs: &Expr<'_>) -> bool {
    SpanlessEq::new(cx).eq_expr(lhs, rhs)
}

fn build_suggestion(
    cx: &LateContext<'_>,
    expr: &Expr<'_>,
    lhs: &Expr<'_>,
    rhs: &Expr<'_>,
    applicability: &mut Applicability,
) {
    let dividend_sugg = Sugg::hir_with_applicability(cx, lhs, "..", applicability).maybe_paren();
    let rhs_ty = cx.typeck_results().expr_ty(rhs);
    let type_suffix = if cx.typeck_results().expr_ty(lhs).is_numeric()
        && matches!(
            lhs.kind,
            ExprKind::Lit(Spanned {
                node: LitKind::Int(_, LitIntType::Unsuffixed),
                ..
            }) | ExprKind::Unary(
                UnOp::Neg,
                Expr {
                    kind: ExprKind::Lit(Spanned {
                        node: LitKind::Int(_, LitIntType::Unsuffixed),
                        ..
                    }),
                    ..
                }
            )
        ) {
        format!("_{rhs_ty}")
    } else {
        String::new()
    };
    let dividend_sugg_str = dividend_sugg.into_string();
    // If `dividend_sugg` has enclosing paren like `(-2048)` and we need to add type suffix in the
    // suggestion message, we want to make a suggestion string before `div_ceil` like
    // `(-2048_{type_suffix})`.
    let suggestion_before_div_ceil = if has_enclosing_paren(&dividend_sugg_str) {
        format!(
            "{}{})",
            &dividend_sugg_str[..dividend_sugg_str.len() - 1].to_string(),
            type_suffix
        )
    } else {
        format!("{dividend_sugg_str}{type_suffix}")
    };

    // Dereference the RHS if it is a reference type
    let divisor_snippet = match Sugg::hir_with_context(cx, rhs, expr.span.ctxt(), "_", applicability) {
        sugg if rhs_ty.is_ref() => sugg.deref(),
        sugg => sugg,
    };

    span_lint_and_sugg(
        cx,
        MANUAL_DIV_CEIL,
        expr.span,
        "manually reimplementing `div_ceil`",
        "consider using `.div_ceil()`",
        format!("{suggestion_before_div_ceil}.div_ceil({divisor_snippet})"),
        *applicability,
    );
}
