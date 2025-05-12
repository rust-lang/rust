use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::is_trait_method;
use clippy_utils::ty::is_type_diagnostic_item;
use rustc_hir::{Expr, ExprKind, QPath};
use rustc_lint::LateContext;
use rustc_span::sym;

use super::VERBOSE_FILE_READS;

pub(super) const READ_TO_END_MSG: (&str, &str) = ("use of `File::read_to_end`", "consider using `fs::read` instead");
pub(super) const READ_TO_STRING_MSG: (&str, &str) = (
    "use of `File::read_to_string`",
    "consider using `fs::read_to_string` instead",
);

pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'_>,
    recv: &'tcx Expr<'_>,
    (msg, help): (&'static str, &'static str),
) {
    if is_trait_method(cx, expr, sym::IoRead)
        && matches!(recv.kind, ExprKind::Path(QPath::Resolved(None, _)))
        && is_type_diagnostic_item(cx, cx.typeck_results().expr_ty_adjusted(recv).peel_refs(), sym::File)
    {
        #[expect(clippy::collapsible_span_lint_calls, reason = "rust-clippy#7797")]
        span_lint_and_then(cx, VERBOSE_FILE_READS, expr.span, msg, |diag| {
            diag.help(help);
        });
    }
}
