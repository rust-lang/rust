use super::UNBUFFERED_BYTES;
use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::is_trait_method;
use clippy_utils::ty::implements_trait;
use rustc_hir as hir;
use rustc_lint::LateContext;
use rustc_span::sym;

pub(super) fn check(cx: &LateContext<'_>, expr: &hir::Expr<'_>, recv: &hir::Expr<'_>) {
    // Lint if the `.bytes()` call is from the `Read` trait and the implementor is not buffered.
    if is_trait_method(cx, expr, sym::IoRead)
        && let Some(buf_read) = cx.tcx.get_diagnostic_item(sym::IoBufRead)
        && let ty = cx.typeck_results().expr_ty_adjusted(recv)
        && !implements_trait(cx, ty, buf_read, &[])
    {
        span_lint_and_help(
            cx,
            UNBUFFERED_BYTES,
            expr.span,
            "calling .bytes() is very inefficient when data is not in memory",
            None,
            "consider using `BufReader`",
        );
    }
}
