use crate::methods::chars_cmp;
use clippy_utils::sym;
use rustc_lint::LateContext;

use super::CHARS_LAST_CMP;

/// Checks for the `CHARS_LAST_CMP` lint.
pub(super) fn check(cx: &LateContext<'_>, info: &crate::methods::BinaryExprInfo<'_>) -> bool {
    if chars_cmp::check(cx, info, &[sym::chars, sym::last], CHARS_LAST_CMP, "ends_with") {
        true
    } else {
        chars_cmp::check(cx, info, &[sym::chars, sym::next_back], CHARS_LAST_CMP, "ends_with")
    }
}
