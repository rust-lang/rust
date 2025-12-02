use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::ty::has_debug_impl;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::{LateContext, LintContext};
use rustc_middle::ty::{self, Ty};
use rustc_span::sym;

use super::OK_EXPECT;

/// lint use of `ok().expect()` for `Result`s
pub(super) fn check(cx: &LateContext<'_>, expr: &hir::Expr<'_>, recv: &hir::Expr<'_>, recv_inner: &hir::Expr<'_>) {
    let result_ty = cx.typeck_results().expr_ty(recv_inner);
    // lint if the caller of `ok()` is a `Result`
    if let Some(error_type) = get_error_type(cx, result_ty)
        && has_debug_impl(cx, error_type)
        && let Some(span) = recv.span.trim_start(recv_inner.span)
    {
        span_lint_and_then(
            cx,
            OK_EXPECT,
            expr.span,
            "called `ok().expect()` on a `Result` value",
            |diag| {
                let span = cx.sess().source_map().span_extend_while_whitespace(span);
                diag.span_suggestion_verbose(
                    span,
                    "call `expect()` directly on the `Result`",
                    String::new(),
                    Applicability::MachineApplicable,
                );
            },
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
