use clippy_utils::consts::ConstEvalCtxt;
use clippy_utils::consts::Constant::{F32, F64};
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::sugg::Sugg;
use clippy_utils::sym;
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Expr, ExprKind};
use rustc_lint::LateContext;
use rustc_span::source_map::Spanned;

use super::IMPRECISE_FLOPS;

// TODO: Lint expressions of the form `x.exp() - y` where y > 1
// and suggest usage of `x.exp_m1() - (y - 1)` instead
pub(super) fn check(cx: &LateContext<'_>, expr: &Expr<'_>) {
    if let ExprKind::Binary(
        Spanned {
            node: BinOpKind::Sub, ..
        },
        lhs,
        rhs,
    ) = expr.kind
        && let ExprKind::MethodCall(path, self_arg, [], _) = lhs.kind
        && path.ident.name == sym::exp
        && cx.typeck_results().expr_ty(lhs).is_floating_point()
        && let Some(value) = ConstEvalCtxt::new(cx).eval(rhs)
        && (F32(1.0) == value || F64(1.0) == value)
        && cx.typeck_results().expr_ty(self_arg).is_floating_point()
    {
        span_lint_and_then(
            cx,
            IMPRECISE_FLOPS,
            expr.span,
            "(e.pow(x) - 1) can be computed more accurately",
            |diag| {
                let mut app = Applicability::MachineApplicable;
                let recv = Sugg::hir_with_applicability(cx, self_arg, "_", &mut app).maybe_paren();
                diag.span_suggestion(expr.span, "consider using", format!("{recv}.exp_m1()"), app);
            },
        );
    }
}
