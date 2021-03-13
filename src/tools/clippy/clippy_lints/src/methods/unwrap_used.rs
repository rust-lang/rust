use crate::utils::{is_type_diagnostic_item, span_lint_and_help};
use rustc_hir as hir;
use rustc_lint::LateContext;
use rustc_span::sym;

use super::UNWRAP_USED;

/// lint use of `unwrap()` for `Option`s and `Result`s
pub(super) fn check(cx: &LateContext<'_>, expr: &hir::Expr<'_>, unwrap_args: &[hir::Expr<'_>]) {
    let obj_ty = cx.typeck_results().expr_ty(&unwrap_args[0]).peel_refs();

    let mess = if is_type_diagnostic_item(cx, obj_ty, sym::option_type) {
        Some((UNWRAP_USED, "an Option", "None"))
    } else if is_type_diagnostic_item(cx, obj_ty, sym::result_type) {
        Some((UNWRAP_USED, "a Result", "Err"))
    } else {
        None
    };

    if let Some((lint, kind, none_value)) = mess {
        span_lint_and_help(
            cx,
            lint,
            expr.span,
            &format!("used `unwrap()` on `{}` value", kind,),
            None,
            &format!(
                "if you don't want to handle the `{}` case gracefully, consider \
                using `expect()` to provide a better panic message",
                none_value,
            ),
        );
    }
}
