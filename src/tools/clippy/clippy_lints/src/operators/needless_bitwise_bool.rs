use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::SpanRangeExt;
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
                if let Some(lhs_snip) = lhs.span.get_source_text(cx)
                    && let Some(rhs_snip) = rhs.span.get_source_text(cx)
                {
                    let sugg = format!("{lhs_snip} {op_str} {rhs_snip}");
                    diag.span_suggestion(e.span, "try", sugg, Applicability::MachineApplicable);
                }
            },
        );
    }
}
