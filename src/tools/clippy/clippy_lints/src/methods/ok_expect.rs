use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::ty::{implements_trait, is_type_diagnostic_item};
use if_chain::if_chain;
use rustc_hir as hir;
use rustc_lint::LateContext;
use rustc_middle::ty::{self, Ty};
use rustc_span::sym;

use super::OK_EXPECT;

/// lint use of `ok().expect()` for `Result`s
pub(super) fn check(cx: &LateContext<'_>, expr: &hir::Expr<'_>, recv: &hir::Expr<'_>) {
    if_chain! {
        // lint if the caller of `ok()` is a `Result`
        if is_type_diagnostic_item(cx, cx.typeck_results().expr_ty(recv), sym::Result);
        let result_type = cx.typeck_results().expr_ty(recv);
        if let Some(error_type) = get_error_type(cx, result_type);
        if has_debug_impl(error_type, cx);

        then {
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
}

/// Given a `Result<T, E>` type, return its error type (`E`).
fn get_error_type<'a>(cx: &LateContext<'_>, ty: Ty<'a>) -> Option<Ty<'a>> {
    match ty.kind() {
        ty::Adt(_, substs) if is_type_diagnostic_item(cx, ty, sym::Result) => substs.types().nth(1),
        _ => None,
    }
}

/// This checks whether a given type is known to implement Debug.
fn has_debug_impl<'tcx>(ty: Ty<'tcx>, cx: &LateContext<'tcx>) -> bool {
    cx.tcx
        .get_diagnostic_item(sym::Debug)
        .map_or(false, |debug| implements_trait(cx, ty, debug, &[]))
}
