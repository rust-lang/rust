use clippy_utils::diagnostics::span_lint_and_then;
use rustc_ast::ast::Lit;
use rustc_errors::Applicability;
use rustc_lint::EarlyContext;

use super::ZERO_PREFIXED_LITERAL;

pub(super) fn check(cx: &EarlyContext<'_>, lit: &Lit, lit_snip: &str) {
    let trimmed_lit_snip = lit_snip.trim_start_matches(|c| c == '_' || c == '0');
    span_lint_and_then(
        cx,
        ZERO_PREFIXED_LITERAL,
        lit.span,
        "this is a decimal constant",
        |diag| {
            diag.span_suggestion(
                lit.span,
                "if you mean to use a decimal constant, remove the `0` to avoid confusion",
                trimmed_lit_snip.to_string(),
                Applicability::MaybeIncorrect,
            );
            // do not advise to use octal form if the literal cannot be expressed in base 8.
            if !lit_snip.contains(|c| c == '8' || c == '9') {
                diag.span_suggestion(
                    lit.span,
                    "if you mean to use an octal constant, use `0o`",
                    format!("0o{trimmed_lit_snip}"),
                    Applicability::MaybeIncorrect,
                );
            }
        },
    );
}
