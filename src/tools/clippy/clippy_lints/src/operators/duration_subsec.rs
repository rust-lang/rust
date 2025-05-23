use clippy_utils::consts::{ConstEvalCtxt, Constant};
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::sym;
use clippy_utils::ty::is_type_diagnostic_item;
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Expr, ExprKind};
use rustc_lint::LateContext;

use super::DURATION_SUBSEC;

pub(crate) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'_>,
    op: BinOpKind,
    left: &'tcx Expr<'_>,
    right: &'tcx Expr<'_>,
) {
    if op == BinOpKind::Div
        && let ExprKind::MethodCall(method_path, self_arg, [], _) = left.kind
        && is_type_diagnostic_item(cx, cx.typeck_results().expr_ty(self_arg).peel_refs(), sym::Duration)
        && let Some(Constant::Int(divisor)) = ConstEvalCtxt::new(cx).eval(right)
    {
        let suggested_fn = match (method_path.ident.name, divisor) {
            (sym::subsec_micros, 1_000) | (sym::subsec_nanos, 1_000_000) => "subsec_millis",
            (sym::subsec_nanos, 1_000) => "subsec_micros",
            _ => return,
        };
        let mut applicability = Applicability::MachineApplicable;
        span_lint_and_sugg(
            cx,
            DURATION_SUBSEC,
            expr.span,
            format!("calling `{suggested_fn}()` is more concise than this calculation"),
            "try",
            format!(
                "{}.{suggested_fn}()",
                snippet_with_applicability(cx, self_arg.span, "_", &mut applicability)
            ),
            applicability,
        );
    }
}
