use crate::methods::get_hint_if_single_char_arg;
use crate::utils::span_lint_and_sugg;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::LateContext;

use super::SINGLE_CHAR_PATTERN;

/// lint for length-1 `str`s for methods in `PATTERN_METHODS`
pub(super) fn check(cx: &LateContext<'_>, _expr: &hir::Expr<'_>, arg: &hir::Expr<'_>) {
    let mut applicability = Applicability::MachineApplicable;
    if let Some(hint) = get_hint_if_single_char_arg(cx, arg, &mut applicability) {
        span_lint_and_sugg(
            cx,
            SINGLE_CHAR_PATTERN,
            arg.span,
            "single-character string constant used as pattern",
            "try using a `char` instead",
            hint,
            applicability,
        );
    }
}
