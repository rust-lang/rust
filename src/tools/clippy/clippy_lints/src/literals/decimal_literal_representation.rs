use super::DECIMAL_LITERAL_REPRESENTATION;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::numeric_literal::{NumericLiteral, Radix};
use clippy_utils::source::SpanExt as _;
use rustc_ast::{LitKind, token};
use rustc_lint::{Applicability, EarlyContext};
use rustc_span::Span;

pub(super) fn check(cx: &EarlyContext<'_>, lit: token::Lit, span: Span, threshold: u64) {
    // Lint integral literals.
    if let Ok(lit_kind) = LitKind::from_token_lit(lit)
        && let LitKind::Int(val, _) = lit_kind
        && val >= u128::from(threshold)
        && let Some(src) = span.get_text(cx)
        && let Some(num_lit) = NumericLiteral::from_lit_kind(&src, &lit_kind)
        && num_lit.radix == Radix::Decimal
        && let hex = format!("{val:#X}")
        && let num_lit = NumericLiteral::new(&hex, num_lit.suffix, false)
        && check_text(num_lit.integer)
    {
        #[expect(clippy::collapsible_span_lint_calls, reason = "rust-clippy#7797")]
        span_lint_and_then(
            cx,
            DECIMAL_LITERAL_REPRESENTATION,
            span,
            "integer literal has a better hexadecimal representation",
            |diag| {
                diag.span_suggestion(span, "consider", num_lit.format(), Applicability::MaybeIncorrect);
            },
        );
    }
}

fn check_text(digits: &str) -> bool {
    if digits.len() == 1 {
        // Lint for 1 digit literals, if someone really sets the threshold that low
        if digits == "1"
            || digits == "2"
            || digits == "4"
            || digits == "8"
            || digits == "3"
            || digits == "7"
            || digits == "F"
        {
            return true;
        }
    } else if digits.len() < 4 {
        // Lint for Literals with a hex-representation of 2 or 3 digits
        let f = &digits[0..1]; // first digit
        let s = &digits[1..]; // suffix

        // Powers of 2
        if ((f.eq("1") || f.eq("2") || f.eq("4") || f.eq("8")) && s.chars().all(|c| c == '0'))
            // Powers of 2 minus 1
            || ((f.eq("1") || f.eq("3") || f.eq("7") || f.eq("F")) && s.chars().all(|c| c == 'F'))
        {
            return true;
        }
    } else {
        // Lint for Literals with a hex-representation of 4 digits or more
        let f = &digits[0..1]; // first digit
        let m = &digits[1..digits.len() - 1]; // middle digits, except last
        let s = &digits[1..]; // suffix

        // Powers of 2 with a margin of +15/-16
        if ((f.eq("1") || f.eq("2") || f.eq("4") || f.eq("8")) && m.chars().all(|c| c == '0'))
            || ((f.eq("1") || f.eq("3") || f.eq("7") || f.eq("F")) && m.chars().all(|c| c == 'F'))
            // Lint for representations with only 0s and Fs, while allowing 7 as the first
            // digit
            || ((f.eq("7") || f.eq("F")) && s.chars().all(|c| c == '0' || c == 'F'))
        {
            return true;
        }
    }

    false
}
