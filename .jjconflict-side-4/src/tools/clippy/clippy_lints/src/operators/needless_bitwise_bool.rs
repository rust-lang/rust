use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::snippet_with_context;
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Expr, ExprKind};
use rustc_lint::LateContext;

use super::NEEDLESS_BITWISE_BOOL;

pub(super) fn check(cx: &LateContext<'_>, e: &Expr<'_>, op: BinOpKind, lhs: &Expr<'_>, rhs: &Expr<'_>) {
    let op_str = match op {
        BinOpKind::BitAnd => "&&",
        BinOpKind::BitOr => "||",
        _ => return,
    };
    if matches!(
        rhs.kind,
        ExprKind::Call(..) | ExprKind::MethodCall(..) | ExprKind::Binary(..) | ExprKind::Unary(..)
    ) && cx.typeck_results().expr_ty(e).is_bool()
        && !rhs.can_have_side_effects()
    {
        span_lint_and_then(
            cx,
            NEEDLESS_BITWISE_BOOL,
            e.span,
            "use of bitwise operator instead of lazy operator between booleans",
            |diag| {
                let mut applicability = Applicability::MachineApplicable;
                let expr_span_ctxt = e.span.ctxt();
                let (lhs_snip, _) = snippet_with_context(cx, lhs.span, expr_span_ctxt, "..", &mut applicability);
                let (rhs_snip, _) = snippet_with_context(cx, rhs.span, expr_span_ctxt, "..", &mut applicability);

                diag.span_suggestion(e.span, "try", format!("{lhs_snip} {op_str} {rhs_snip}"), applicability);
            },
        );
    }
}
