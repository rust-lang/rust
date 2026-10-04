use clippy_utils::sym;
use rustc_lint::LateContext;

use super::CHARS_NEXT_CMP;

/// Checks for the `CHARS_NEXT_CMP` lint with `unwrap()`.
pub(super) fn check(cx: &LateContext<'_>, info: &crate::methods::BinaryExprInfo<'_>) -> bool {
    crate::methods::chars_cmp_with_unwrap::check(
        cx,
        info,
        &[sym::chars, sym::next, sym::unwrap],
        CHARS_NEXT_CMP,
        "starts_with",
    )
}
