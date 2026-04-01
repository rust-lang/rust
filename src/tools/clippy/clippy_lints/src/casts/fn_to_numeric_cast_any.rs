use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::snippet_with_applicability;
use rustc_errors::Applicability;
use rustc_hir::Expr;
use rustc_lint::LateContext;
use rustc_middle::ty::Ty;

use super::FN_TO_NUMERIC_CAST_ANY;

pub(super) fn check(cx: &LateContext<'_>, expr: &Expr<'_>, cast_expr: &Expr<'_>, cast_from: Ty<'_>, cast_to: Ty<'_>) {
    // We allow casts from any function type to any function type.
    if cast_to.is_fn() {
        return;
    }

    if cast_from.is_fn() {
        let mut applicability = Applicability::MaybeIncorrect;
        let from_snippet = snippet_with_applicability(cx, cast_expr.span, "..", &mut applicability);

        span_lint_and_then(
            cx,
            FN_TO_NUMERIC_CAST_ANY,
            expr.span,
            format!("casting function pointer `{from_snippet}` to `{cast_to}`"),
            |diag| {
                diag.span_suggestion_verbose(
                    expr.span,
                    "did you mean to invoke the function?",
                    format!("{from_snippet}() as {cast_to}"),
                    applicability,
                );
            },
        );
    }
}
