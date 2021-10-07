use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_with_applicability;
use rustc_errors::Applicability;
use rustc_hir::Expr;
use rustc_lint::LateContext;
use rustc_middle::ty::{self, Ty};

use super::FN_TO_NUMERIC_CAST_ANY;

pub(super) fn check(cx: &LateContext<'_>, expr: &Expr<'_>, cast_expr: &Expr<'_>, cast_from: Ty<'_>, cast_to: Ty<'_>) {
    // We allow casts from any function type to any function type.
    match cast_to.kind() {
        ty::FnDef(..) | ty::FnPtr(..) => return,
        _ => { /* continue to checks */ },
    }

    match cast_from.kind() {
        ty::FnDef(..) | ty::FnPtr(_) => {
            let mut applicability = Applicability::MaybeIncorrect;
            let from_snippet = snippet_with_applicability(cx, cast_expr.span, "..", &mut applicability);

            span_lint_and_sugg(
                cx,
                FN_TO_NUMERIC_CAST_ANY,
                expr.span,
                &format!("casting function pointer `{}` to `{}`", from_snippet, cast_to),
                "did you mean to invoke the function?",
                format!("{}() as {}", from_snippet, cast_to),
                applicability,
            );
        },
        _ => {},
    }
}
