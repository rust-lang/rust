use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::ty::has_debug_impl;
use rustc_hir as hir;
use rustc_lint::LateContext;
use rustc_middle::ty::{self, Ty};
use rustc_span::sym;

use super::OK_EXPECT;

/// lint use of `ok().expect()` for `Result`s
pub(super) fn check(cx: &LateContext<'_>, expr: &hir::Expr<'_>, recv: &hir::Expr<'_>) {
    let result_ty = cx.typeck_results().expr_ty(recv);
    // lint if the caller of `ok()` is a `Result`
    if let Some(error_type) = get_error_type(cx, result_ty)
        && has_debug_impl(cx, error_type)
    {
        span_lint_and_help(
            cx,
            OK_EXPECT,
            expr.span,
            "called `ok().expect()` on a `Result` value",
            None,
            "you can call `expect()` directly on the `Result`",
        );
    }
}

/// Given a `Result<T, E>` type, return its error type (`E`).
fn get_error_type<'a>(cx: &LateContext<'_>, ty: Ty<'a>) -> Option<Ty<'a>> {
    match ty.kind() {
        ty::Adt(adt, args) if cx.tcx.is_diagnostic_item(sym::Result, adt.did()) => args.types().nth(1),
        _ => None,
    }
}
