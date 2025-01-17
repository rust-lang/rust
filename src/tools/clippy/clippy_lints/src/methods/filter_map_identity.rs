use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::{is_expr_identity_function, is_expr_untyped_identity_function, is_trait_method};
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::ExprKind;
use rustc_lint::LateContext;
use rustc_span::{Span, sym};

use super::FILTER_MAP_IDENTITY;

fn is_identity(cx: &LateContext<'_>, expr: &hir::Expr<'_>) -> Option<Applicability> {
    if is_expr_untyped_identity_function(cx, expr) {
        return Some(Applicability::MachineApplicable);
    }
    if is_expr_identity_function(cx, expr) {
        return Some(Applicability::Unspecified);
    }
    None
}

pub(super) fn check(cx: &LateContext<'_>, expr: &hir::Expr<'_>, filter_map_arg: &hir::Expr<'_>, filter_map_span: Span) {
    if is_trait_method(cx, expr, sym::Iterator)
        && let Some(applicability) = is_identity(cx, filter_map_arg)
    {
        // check if the iterator is from an empty array, see issue #12653
        if let ExprKind::MethodCall(_, recv, ..) = expr.kind
            && let ExprKind::MethodCall(_, recv2, ..) = recv.kind
            && let ExprKind::Array(arr) = recv2.kind
            && arr.is_empty()
        {
            return;
        }

        span_lint_and_sugg(
            cx,
            FILTER_MAP_IDENTITY,
            filter_map_span.with_hi(expr.span.hi()),
            "use of `filter_map` with an identity function",
            "try",
            "flatten()".to_string(),
            applicability,
        );
    }
}
