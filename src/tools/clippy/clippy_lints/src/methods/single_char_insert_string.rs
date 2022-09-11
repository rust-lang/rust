use super::utils::get_hint_if_single_char_arg;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_with_applicability;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::LateContext;

use super::SINGLE_CHAR_ADD_STR;

/// lint for length-1 `str`s as argument for `insert_str`
pub(super) fn check(cx: &LateContext<'_>, expr: &hir::Expr<'_>, receiver: &hir::Expr<'_>, args: &[hir::Expr<'_>]) {
    let mut applicability = Applicability::MachineApplicable;
    if let Some(extension_string) = get_hint_if_single_char_arg(cx, &args[1], &mut applicability) {
        let base_string_snippet =
            snippet_with_applicability(cx, receiver.span.source_callsite(), "_", &mut applicability);
        let pos_arg = snippet_with_applicability(cx, args[0].span, "..", &mut applicability);
        let sugg = format!("{}.insert({}, {})", base_string_snippet, pos_arg, extension_string);
        span_lint_and_sugg(
            cx,
            SINGLE_CHAR_ADD_STR,
            expr.span,
            "calling `insert_str()` using a single-character string literal",
            "consider using `insert` with a character literal",
            sugg,
            applicability,
        );
    }
}
