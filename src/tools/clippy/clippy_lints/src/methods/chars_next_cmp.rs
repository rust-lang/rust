use rustc_lint::LateContext;

use super::CHARS_NEXT_CMP;

/// Checks for the `CHARS_NEXT_CMP` lint.
pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, info: &crate::methods::BinaryExprInfo<'_>) -> bool {
    crate::methods::chars_cmp::check(cx, info, &["chars", "next"], CHARS_NEXT_CMP, "starts_with")
}
