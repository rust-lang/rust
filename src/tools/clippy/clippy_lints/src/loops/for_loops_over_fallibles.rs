use super::FOR_LOOPS_OVER_FALLIBLES;
use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::source::snippet;
use clippy_utils::ty::is_type_diagnostic_item;
use rustc_hir::{Expr, Pat};
use rustc_lint::LateContext;
use rustc_span::symbol::sym;

/// Checks for `for` loops over `Option`s and `Result`s.
pub(super) fn check(cx: &LateContext<'_>, pat: &Pat<'_>, arg: &Expr<'_>, method_name: Option<&str>) {
    let ty = cx.typeck_results().expr_ty(arg);
    if is_type_diagnostic_item(cx, ty, sym::Option) {
        let help_string = if let Some(method_name) = method_name {
            format!(
                "consider replacing `for {0} in {1}.{method_name}()` with `if let Some({0}) = {1}`",
                snippet(cx, pat.span, "_"),
                snippet(cx, arg.span, "_")
            )
        } else {
            format!(
                "consider replacing `for {0} in {1}` with `if let Some({0}) = {1}`",
                snippet(cx, pat.span, "_"),
                snippet(cx, arg.span, "_")
            )
        };
        span_lint_and_help(
            cx,
            FOR_LOOPS_OVER_FALLIBLES,
            arg.span,
            &format!(
                "for loop over `{0}`, which is an `Option`. This is more readably written as an \
                `if let` statement",
                snippet(cx, arg.span, "_")
            ),
            None,
            &help_string,
        );
    } else if is_type_diagnostic_item(cx, ty, sym::Result) {
        let help_string = if let Some(method_name) = method_name {
            format!(
                "consider replacing `for {0} in {1}.{method_name}()` with `if let Ok({0}) = {1}`",
                snippet(cx, pat.span, "_"),
                snippet(cx, arg.span, "_")
            )
        } else {
            format!(
                "consider replacing `for {0} in {1}` with `if let Ok({0}) = {1}`",
                snippet(cx, pat.span, "_"),
                snippet(cx, arg.span, "_")
            )
        };
        span_lint_and_help(
            cx,
            FOR_LOOPS_OVER_FALLIBLES,
            arg.span,
            &format!(
                "for loop over `{0}`, which is a `Result`. This is more readably written as an \
                `if let` statement",
                snippet(cx, arg.span, "_")
            ),
            None,
            &help_string,
        );
    }
}
