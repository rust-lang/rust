use clippy_utils::diagnostics::span_lint_and_then;
use rustc_ast::ast::Lit;
use rustc_errors::Applicability;
use rustc_lint::EarlyContext;

use super::ZERO_PREFIXED_LITERAL;

pub(super) fn check(cx: &EarlyContext<'_>, lit: &Lit, lit_snip: &str) {
    span_lint_and_then(
        cx,
        ZERO_PREFIXED_LITERAL,
        lit.span,
        "this is a decimal constant",
        |diag| {
            diag.span_suggestion(
                lit.span,
                "if you mean to use a decimal constant, remove the `0` to avoid confusion",
                lit_snip.trim_start_matches(|c| c == '_' || c == '0').to_string(),
                Applicability::MaybeIncorrect,
            );
            diag.span_suggestion(
                lit.span,
                "if you mean to use an octal constant, use `0o`",
                format!("0o{}", lit_snip.trim_start_matches(|c| c == '_' || c == '0')),
                Applicability::MaybeIncorrect,
            );
        },
    );
}
