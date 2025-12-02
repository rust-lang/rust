use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::res::{MaybeDef, MaybeTypeckRes};
use clippy_utils::source::snippet;
use rustc_errors::Applicability;
use rustc_hir::Expr;
use rustc_lint::LateContext;
use rustc_span::sym;

use super::UNIT_HASH;

pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>, recv: &'tcx Expr<'_>, arg: &'tcx Expr<'_>) {
    if cx.ty_based_def(expr).opt_parent(cx).is_diag_item(cx, sym::Hash) && cx.typeck_results().expr_ty(recv).is_unit() {
        span_lint_and_then(
            cx,
            UNIT_HASH,
            expr.span,
            "this call to `hash` on the unit type will do nothing",
            |diag| {
                diag.span_suggestion(
                    expr.span,
                    "remove the call to `hash` or consider using",
                    format!("0_u8.hash({})", snippet(cx, arg.span, ".."),),
                    Applicability::MaybeIncorrect,
                );
                diag.note("the implementation of `Hash` for `()` is a no-op");
            },
        );
    }
}
