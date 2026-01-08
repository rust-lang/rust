use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::sugg::Sugg;
use clippy_utils::{eq_expr_value, sym};
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Expr, ExprKind, PathSegment};
use rustc_lint::LateContext;
use rustc_span::source_map::Spanned;

use super::SUBOPTIMAL_FLOPS;

fn are_same_base_logs(cx: &LateContext<'_>, expr_a: &Expr<'_>, expr_b: &Expr<'_>) -> bool {
    if let ExprKind::MethodCall(PathSegment { ident: method_a, .. }, _, args_a, _) = expr_a.kind
        && let ExprKind::MethodCall(PathSegment { ident: method_b, .. }, _, args_b, _) = expr_b.kind
    {
        return method_a.name == method_b.name
            && args_a.len() == args_b.len()
            && (matches!(method_a.name, sym::ln | sym::log2 | sym::log10)
                || method_a.name == sym::log && args_a.len() == 1 && eq_expr_value(cx, &args_a[0], &args_b[0]));
    }

    false
}

pub(super) fn check(cx: &LateContext<'_>, expr: &Expr<'_>) {
    // check if expression of the form x.logN() / y.logN()
    if let ExprKind::Binary(
        Spanned {
            node: BinOpKind::Div, ..
        },
        lhs,
        rhs,
    ) = expr.kind
        && are_same_base_logs(cx, lhs, rhs)
        && let ExprKind::MethodCall(_, largs_self, ..) = lhs.kind
        && let ExprKind::MethodCall(_, rargs_self, ..) = rhs.kind
    {
        let mut app = Applicability::MachineApplicable;
        span_lint_and_sugg(
            cx,
            SUBOPTIMAL_FLOPS,
            expr.span,
            "log base can be expressed more clearly",
            "consider using",
            format!(
                "{}.log({})",
                Sugg::hir_with_applicability(cx, largs_self, "_", &mut app).maybe_paren(),
                Sugg::hir_with_applicability(cx, rargs_self, "_", &mut app),
            ),
            app,
        );
    }
}
