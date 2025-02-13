use clippy_utils::diagnostics::span_lint_and_sugg;
use rustc_errors::Applicability;
use rustc_lint::LateContext;
use rustc_span::{BytePos, Span};

use super::DOC_COMMENT_DOUBLE_SPACE_LINEBREAK;

pub fn check(cx: &LateContext<'_>, collected_breaks: &[Span]) {
    for r_span in collected_breaks {
        span_lint_and_sugg(
            cx,
            DOC_COMMENT_DOUBLE_SPACE_LINEBREAK,
            r_span.with_hi(r_span.lo() + BytePos(2)),
            "doc comment uses two spaces for a hard line break",
            "replace this double space with a backslash",
            "\\".to_owned(),
            Applicability::MachineApplicable,
        );
    }
}
