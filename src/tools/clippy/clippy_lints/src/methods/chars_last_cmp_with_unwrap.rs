use crate::methods::chars_cmp_with_unwrap;
use clippy_utils::sym;
use rustc_lint::LateContext;

use super::CHARS_LAST_CMP;

/// Checks for the `CHARS_LAST_CMP` lint with `unwrap()`.
pub(super) fn check(cx: &LateContext<'_>, info: &crate::methods::BinaryExprInfo<'_>) -> bool {
    if chars_cmp_with_unwrap::check(
        cx,
        info,
        &[sym::chars, sym::last, sym::unwrap],
        CHARS_LAST_CMP,
        "ends_with",
    ) {
        true
    } else {
        chars_cmp_with_unwrap::check(
            cx,
            info,
            &[sym::chars, sym::next_back, sym::unwrap],
            CHARS_LAST_CMP,
            "ends_with",
        )
    }
}
