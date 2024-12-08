use crate::methods::utils::derefs_to_slice;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::ty::is_type_diagnostic_item;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::LateContext;
use rustc_span::sym;

use super::ITER_CLONED_COLLECT;

pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, method_name: &str, expr: &hir::Expr<'_>, recv: &'tcx hir::Expr<'_>) {
    if is_type_diagnostic_item(cx, cx.typeck_results().expr_ty(expr), sym::Vec)
        && let Some(slice) = derefs_to_slice(cx, recv, cx.typeck_results().expr_ty(recv))
        && let Some(to_replace) = expr.span.trim_start(slice.span.source_callsite())
    {
        span_lint_and_sugg(
            cx,
            ITER_CLONED_COLLECT,
            to_replace,
            format!(
                "called `iter().{method_name}().collect()` on a slice to create a `Vec`. Calling `to_vec()` is both faster and \
            more readable"
            ),
            "try",
            ".to_vec()".to_string(),
            Applicability::MachineApplicable,
        );
    }
}
