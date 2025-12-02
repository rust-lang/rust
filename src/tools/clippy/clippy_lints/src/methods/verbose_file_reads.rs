use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::res::{MaybeDef, MaybeTypeckRes};
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
    if cx.ty_based_def(expr).opt_parent(cx).is_diag_item(cx, sym::IoRead)
        && matches!(recv.kind, ExprKind::Path(QPath::Resolved(None, _)))
        && cx
            .typeck_results()
            .expr_ty_adjusted(recv)
            .peel_refs()
            .is_diag_item(cx, sym::File)
    {
        #[expect(clippy::collapsible_span_lint_calls, reason = "rust-clippy#7797")]
        span_lint_and_then(cx, VERBOSE_FILE_READS, expr.span, msg, |diag| {
            diag.help(help);
        });
    }
}
