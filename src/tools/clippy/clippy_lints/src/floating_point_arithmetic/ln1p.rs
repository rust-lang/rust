use clippy_utils::consts::ConstEvalCtxt;
use clippy_utils::consts::Constant::{F32, F64};
use clippy_utils::diagnostics::span_lint_and_then;
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Expr, ExprKind};
use rustc_lint::LateContext;
use rustc_span::source_map::Spanned;

use super::IMPRECISE_FLOPS;

// TODO: Lint expressions of the form `(x + y).ln()` where y > 1 and
// suggest usage of `(x + (y - 1)).ln_1p()` instead
pub(super) fn check(cx: &LateContext<'_>, expr: &Expr<'_>, receiver: &Expr<'_>) {
    if let ExprKind::Binary(
        Spanned {
            node: BinOpKind::Add, ..
        },
        lhs,
        rhs,
    ) = receiver.kind
    {
        let ecx = ConstEvalCtxt::new(cx);
        let recv = match (ecx.eval(lhs), ecx.eval(rhs)) {
            (Some(value), _) if F32(1.0) == value || F64(1.0) == value => rhs,
            (_, Some(value)) if F32(1.0) == value || F64(1.0) == value => lhs,
            _ => return,
        };

        span_lint_and_then(
            cx,
            IMPRECISE_FLOPS,
            expr.span,
            "ln(1 + x) can be computed more accurately",
            |diag| {
                let mut app = Applicability::MachineApplicable;
                let recv = super::lib::prepare_receiver_sugg(cx, recv, &mut app);
                diag.span_suggestion(expr.span, "consider using", format!("{recv}.ln_1p()"), app);
            },
        );
    }
}
