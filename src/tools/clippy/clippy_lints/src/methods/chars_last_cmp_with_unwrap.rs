use crate::methods::chars_cmp_with_unwrap;
use rustc_lint::LateContext;

use super::CHARS_LAST_CMP;

/// Checks for the `CHARS_LAST_CMP` lint with `unwrap()`.
pub(super) fn check(cx: &LateContext<'_>, info: &crate::methods::BinaryExprInfo<'_>) -> bool {
    if chars_cmp_with_unwrap::check(cx, info, &["chars", "last", "unwrap"], CHARS_LAST_CMP, "ends_with") {
        true
    } else {
        chars_cmp_with_unwrap::check(cx, info, &["chars", "next_back", "unwrap"], CHARS_LAST_CMP, "ends_with")
    }
}
