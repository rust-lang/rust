use super::UNBUFFERED_BYTES;
use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::ty::implements_trait;
use clippy_utils::{get_trait_def_id, is_trait_method, paths};
use rustc_hir as hir;
use rustc_lint::LateContext;
use rustc_span::sym;

pub(super) fn check(cx: &LateContext<'_>, expr: &hir::Expr<'_>, recv: &hir::Expr<'_>) {
    let ty = cx.typeck_results().expr_ty_adjusted(recv);

    // If the .bytes() call is a call from the Read trait
    if is_trait_method(cx, expr, sym::IoRead) {
        // Retrieve the DefId of the BufRead trait
        // FIXME: add a diagnostic item for `BufRead`
        let Some(buf_read) = get_trait_def_id(cx.tcx, &paths::BUF_READ) else {
            return;
        };
        // And the implementor of the trait is not buffered
        if !implements_trait(cx, ty, buf_read, &[]) {
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
}
