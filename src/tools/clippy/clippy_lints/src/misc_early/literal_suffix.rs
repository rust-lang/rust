use clippy_utils::diagnostics::span_lint_and_sugg;
use rustc_ast::ast::Lit;
use rustc_errors::Applicability;
use rustc_lint::EarlyContext;

use super::{SEPARATED_LITERAL_SUFFIX, UNSEPARATED_LITERAL_SUFFIX};

pub(super) fn check(cx: &EarlyContext<'_>, lit: &Lit, lit_snip: &str, suffix: &str, sugg_type: &str) {
    let maybe_last_sep_idx = if let Some(val) = lit_snip.len().checked_sub(suffix.len() + 1) {
        val
    } else {
        return; // It's useless so shouldn't lint.
    };
    // Do not lint when literal is unsuffixed.
    if !suffix.is_empty() {
        if lit_snip.as_bytes()[maybe_last_sep_idx] == b'_' {
            span_lint_and_sugg(
                cx,
                SEPARATED_LITERAL_SUFFIX,
                lit.span,
                &format!("{} type suffix should not be separated by an underscore", sugg_type),
                "remove the underscore",
                format!("{}{}", &lit_snip[..maybe_last_sep_idx], suffix),
                Applicability::MachineApplicable,
            );
        } else {
            span_lint_and_sugg(
                cx,
                UNSEPARATED_LITERAL_SUFFIX,
                lit.span,
                &format!("{} type suffix should be separated by an underscore", sugg_type),
                "add an underscore",
                format!("{}_{}", &lit_snip[..=maybe_last_sep_idx], suffix),
                Applicability::MachineApplicable,
            );
        }
    }
}
