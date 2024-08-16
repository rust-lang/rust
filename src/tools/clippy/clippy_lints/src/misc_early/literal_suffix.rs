use clippy_utils::diagnostics::span_lint_and_then;
use rustc_errors::Applicability;
use rustc_lint::EarlyContext;
use rustc_span::Span;

use super::{SEPARATED_LITERAL_SUFFIX, UNSEPARATED_LITERAL_SUFFIX};

pub(super) fn check(cx: &EarlyContext<'_>, lit_span: Span, lit_snip: &str, suffix: &str, sugg_type: &str) {
    let Some(maybe_last_sep_idx) = lit_snip.len().checked_sub(suffix.len() + 1) else {
        return; // It's useless so shouldn't lint.
    };
    // Do not lint when literal is unsuffixed.
    if !suffix.is_empty() {
        if lit_snip.as_bytes()[maybe_last_sep_idx] == b'_' {
            #[expect(clippy::collapsible_span_lint_calls, reason = "rust-clippy#7797")]
            span_lint_and_then(
                cx,
                SEPARATED_LITERAL_SUFFIX,
                lit_span,
                format!("{sugg_type} type suffix should not be separated by an underscore"),
                |diag| {
                    diag.span_suggestion(
                        lit_span,
                        "remove the underscore",
                        format!("{}{suffix}", &lit_snip[..maybe_last_sep_idx]),
                        Applicability::MachineApplicable,
                    );
                },
            );
        } else {
            #[expect(clippy::collapsible_span_lint_calls, reason = "rust-clippy#7797")]
            span_lint_and_then(
                cx,
                UNSEPARATED_LITERAL_SUFFIX,
                lit_span,
                format!("{sugg_type} type suffix should be separated by an underscore"),
                |diag| {
                    diag.span_suggestion(
                        lit_span,
                        "add an underscore",
                        format!("{}_{suffix}", &lit_snip[..=maybe_last_sep_idx]),
                        Applicability::MachineApplicable,
                    );
                },
            );
        }
    }
}
