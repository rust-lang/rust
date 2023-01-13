use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::is_in_cfg_test;
use clippy_utils::ty::is_type_diagnostic_item;
use rustc_hir as hir;
use rustc_lint::LateContext;
use rustc_span::sym;

use super::EXPECT_USED;

/// lint use of `expect()` or `expect_err` for `Result` and `expect()` for `Option`.
pub(super) fn check(
    cx: &LateContext<'_>,
    expr: &hir::Expr<'_>,
    recv: &hir::Expr<'_>,
    is_err: bool,
    allow_expect_in_tests: bool,
) {
    let obj_ty = cx.typeck_results().expr_ty(recv).peel_refs();

    let mess = if is_type_diagnostic_item(cx, obj_ty, sym::Option) && !is_err {
        Some((EXPECT_USED, "an `Option`", "None", ""))
    } else if is_type_diagnostic_item(cx, obj_ty, sym::Result) {
        Some((EXPECT_USED, "a `Result`", if is_err { "Ok" } else { "Err" }, "an "))
    } else {
        None
    };

    let method = if is_err { "expect_err" } else { "expect" };

    if allow_expect_in_tests && is_in_cfg_test(cx.tcx, expr.hir_id) {
        return;
    }

    if let Some((lint, kind, none_value, none_prefix)) = mess {
        span_lint_and_help(
            cx,
            lint,
            expr.span,
            &format!("used `{method}()` on {kind} value"),
            None,
            &format!("if this value is {none_prefix}`{none_value}`, it will panic"),
        );
    }
}
