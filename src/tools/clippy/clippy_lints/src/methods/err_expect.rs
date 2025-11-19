use super::ERR_EXPECT;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::ty::has_debug_impl;
use rustc_errors::Applicability;
use rustc_lint::LateContext;
use rustc_middle::ty;
use rustc_middle::ty::Ty;
use rustc_span::{Span, sym};

pub(super) fn check(
    cx: &LateContext<'_>,
    _expr: &rustc_hir::Expr<'_>,
    recv: &rustc_hir::Expr<'_>,
    expect_span: Span,
    err_span: Span,
    msrv: Msrv,
) {
    let result_ty = cx.typeck_results().expr_ty(recv);
    // Grabs the `Result<T, E>` type
    if let Some(data_type) = get_data_type(cx, result_ty)
        // Tests if the T type in a `Result<T, E>` implements Debug
        && has_debug_impl(cx, data_type)
        && msrv.meets(cx, msrvs::EXPECT_ERR)
    {
        span_lint_and_sugg(
            cx,
            ERR_EXPECT,
            err_span.to(expect_span),
            "called `.err().expect()` on a `Result` value",
            "try",
            "expect_err".to_string(),
            Applicability::MachineApplicable,
        );
    }
}

/// Given a `Result<T, E>` type, return its data (`T`).
fn get_data_type<'a>(cx: &LateContext<'_>, ty: Ty<'a>) -> Option<Ty<'a>> {
    match ty.kind() {
        ty::Adt(adt, args) if cx.tcx.is_diagnostic_item(sym::Result, adt.did()) => args.types().next(),
        _ => None,
    }
}
