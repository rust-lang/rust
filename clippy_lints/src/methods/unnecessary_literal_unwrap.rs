use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::ty::is_type_diagnostic_item;
use rustc_hir as hir;
use rustc_lint::LateContext;
use rustc_span::sym;

use super::UNNECESSARY_LITERAL_UNWRAP;

pub(super) fn check(cx: &LateContext<'_>, expr: &hir::Expr<'_>, recv: &hir::Expr<'_>) {
    let obj_ty = cx.typeck_results().expr_ty(recv).peel_refs();

    let mess = if is_type_diagnostic_item(cx, obj_ty, sym::Option) {
        Some((UNNECESSARY_LITERAL_UNWRAP, "an `Option`", "None", ""))
    } else {
        None
    };

    if let Some((lint, kind, none_value, none_prefix)) = mess {
        let help = format!("if this value is {none_prefix}`{none_value}`, it will panic");

        span_lint_and_help(
            cx,
            lint,
            expr.span,
            &format!("used `unwrap()` on {kind} value"),
            None,
            &help,
        );
    }
}
