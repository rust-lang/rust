use clippy_utils::diagnostics::span_lint_and_sugg;
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
            span_lint_and_sugg(
                cx,
                SEPARATED_LITERAL_SUFFIX,
                lit_span,
                &format!("{sugg_type} type suffix should not be separated by an underscore"),
                "remove the underscore",
                format!("{}{suffix}", &lit_snip[..maybe_last_sep_idx]),
                Applicability::MachineApplicable,
            );
        } else {
            span_lint_and_sugg(
                cx,
                UNSEPARATED_LITERAL_SUFFIX,
                lit_span,
                &format!("{sugg_type} type suffix should be separated by an underscore"),
                "add an underscore",
                format!("{}_{suffix}", &lit_snip[..=maybe_last_sep_idx]),
                Applicability::MachineApplicable,
            );
        }
    }
}
