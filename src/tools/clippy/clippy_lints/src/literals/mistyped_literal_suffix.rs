use super::MISTYPED_LITERAL_SUFFIXES;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::numeric_literal::{NumericLiteral, Radix};
use rustc_lint::{Applicability, EarlyContext};
use rustc_span::Span;

// Returns `false` if the check fails
pub(super) fn check(cx: &EarlyContext<'_>, span: Span, num_lit: &mut NumericLiteral<'_>) -> bool {
    if num_lit.suffix.is_some() {
        return true;
    }

    let (part, mistyped_suffixes, is_float) = if let Some((_, exponent)) = &mut num_lit.exponent {
        (exponent, &["32", "64"][..], true)
    } else if num_lit.fraction.is_some() {
        return true;
    } else {
        (&mut num_lit.integer, &["8", "16", "32", "64"][..], false)
    };

    let mut split = part.rsplit('_');
    let last_group = split.next().expect("At least one group");
    if split.next().is_some() && mistyped_suffixes.contains(&last_group) {
        let main_part = &part[..part.len() - last_group.len()];
        let missing_char;
        if is_float {
            missing_char = 'f';
        } else {
            let radix = match num_lit.radix {
                Radix::Binary => 2,
                Radix::Octal => 8,
                Radix::Decimal => 10,
                Radix::Hexadecimal => 16,
            };
            if let Ok(int) = u64::from_str_radix(&main_part.replace('_', ""), radix) {
                missing_char = match (last_group, int) {
                    ("8", i) if i8::try_from(i).is_ok() => 'i',
                    ("16", i) if i16::try_from(i).is_ok() => 'i',
                    ("32", i) if i32::try_from(i).is_ok() => 'i',
                    ("64", i) if i64::try_from(i).is_ok() => 'i',
                    ("8", u) if u8::try_from(u).is_ok() => 'u',
                    ("16", u) if u16::try_from(u).is_ok() => 'u',
                    ("32", u) if u32::try_from(u).is_ok() => 'u',
                    ("64", _) => 'u',
                    _ => {
                        return true;
                    },
                }
            } else {
                return true;
            }
        }
        *part = main_part;
        span_lint_and_then(cx, MISTYPED_LITERAL_SUFFIXES, span, "mistyped literal suffix", |diag| {
            let mut sugg = num_lit.format();
            sugg.push('_');
            sugg.push(missing_char);
            sugg.push_str(last_group);
            diag.span_suggestion(span, "did you mean to write", sugg, Applicability::MaybeIncorrect);
        });
        false
    } else {
        true
    }
}
