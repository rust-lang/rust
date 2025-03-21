use clippy_utils::diagnostics::span_lint_and_help;
use rustc_lint::EarlyContext;
use rustc_span::Span;

use super::MIXED_CASE_HEX_LITERALS;

pub(super) fn check(cx: &EarlyContext<'_>, lit_span: Span, suffix: &str, lit_snip: &str) {
    let num_end_idx = match lit_snip.strip_suffix(suffix) {
        Some(p) if p.ends_with('_') => lit_snip.len() - (suffix.len() + 1),
        Some(_) => lit_snip.len() - suffix.len(),
        None => lit_snip.len(),
    };

    if num_end_idx <= 2 {
        // It's meaningless or causes range error.
        return;
    }

    let mut seen = (false, false);
    for ch in &lit_snip.as_bytes()[2..num_end_idx] {
        match ch {
            b'a'..=b'f' => seen.0 = true,
            b'A'..=b'F' => seen.1 = true,
            _ => {},
        }
        if seen.0 && seen.1 {
            let raw_digits = &lit_snip[2..num_end_idx];
            let (sugg_lower, sugg_upper) = if suffix.is_empty() {
                (
                    format!("0x{}", raw_digits.to_lowercase()),
                    format!("0x{}", raw_digits.to_uppercase()),
                )
            } else {
                (
                    format!("0x{}_{}", raw_digits.to_lowercase(), suffix),
                    format!("0x{}_{}", raw_digits.to_uppercase(), suffix),
                )
            };

            span_lint_and_help(
                cx,
                MIXED_CASE_HEX_LITERALS,
                lit_span,
                "inconsistent casing in hexadecimal literal",
                None,
                format!("consider using `{sugg_lower}` or `{sugg_upper}`"),
            );
            return;
        }
    }
}
