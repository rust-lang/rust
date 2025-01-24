use std::ops::Range;

use clippy_utils::diagnostics::span_lint;
use rustc_lint::LateContext;

use super::{DOC_LINK_WITH_QUOTES, Fragments};

pub fn check(cx: &LateContext<'_>, trimmed_text: &str, range: Range<usize>, fragments: Fragments<'_>) {
    if ((trimmed_text.starts_with('\'') && trimmed_text.ends_with('\''))
        || (trimmed_text.starts_with('"') && trimmed_text.ends_with('"')))
        && let Some(span) = fragments.span(cx, range)
    {
        span_lint(
            cx,
            DOC_LINK_WITH_QUOTES,
            span,
            "possible intra-doc link using quotes instead of backticks",
        );
    }
}
