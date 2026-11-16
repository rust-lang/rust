use super::{INCONSISTENT_DIGIT_GROUPING, LARGE_DIGIT_GROUPS, UNREADABLE_LITERAL, UNUSUAL_BYTE_GROUPINGS};
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::numeric_literal::{NumericLiteral, Radix};
use core::iter;
use rustc_lint::{Applicability, EarlyContext, Lint};
use rustc_span::Span;

pub(super) fn check(cx: &EarlyContext<'_>, span: Span, num_lit: &NumericLiteral<'_>, lint_fractions: bool) {
    if is_literal_uuid_formatted(num_lit) {
        return;
    }

    let result = (|| {
        let integral_group_size = get_group_size(num_lit.integer.split('_'), num_lit.radix, true)?;
        if let Some(fraction) = num_lit.fraction {
            let fractional_group_size = get_group_size(fraction.rsplit('_'), num_lit.radix, lint_fractions)?;

            let consistent = match (integral_group_size, fractional_group_size) {
                // No groups on either side of decimal point - trivially consistent.
                (None, None) => true,
                // Integral part has grouped digits, fractional part does not.
                (Some(int_group_size), None) => fraction.len() <= int_group_size,
                // Fractional part has grouped digits, integral part does not.
                (None, Some(frac_group_size)) => num_lit.integer.len() <= frac_group_size,
                // Both parts have grouped digits. Groups should be the same size.
                (Some(int_group_size), Some(frac_group_size)) => int_group_size == frac_group_size,
            };
            if !consistent {
                return Err(WarningType::InconsistentDigitGrouping);
            }
        }

        Ok(())
    })();

    if let Err(warning_type) = result
        && !span.from_expansion()
    {
        warning_type.display(num_lit, cx, span);
    }
}

enum WarningType {
    UnreadableLiteral,
    InconsistentDigitGrouping,
    LargeDigitGroups,
    UnusualByteGroupings,
}

impl WarningType {
    fn lint_and_text(&self) -> (&'static Lint, &'static str, &'static str) {
        match self {
            Self::UnreadableLiteral => (UNREADABLE_LITERAL, "long literal lacking separators", "consider"),
            Self::LargeDigitGroups => (LARGE_DIGIT_GROUPS, "digit groups should be smaller", "consider"),
            Self::InconsistentDigitGrouping => (
                INCONSISTENT_DIGIT_GROUPING,
                "digits grouped inconsistently by underscores",
                "consider",
            ),
            Self::UnusualByteGroupings => (
                UNUSUAL_BYTE_GROUPINGS,
                "digits of hex, binary or octal literal not in groups of equal size",
                "consider",
            ),
        }
    }

    fn display(&self, num_lit: &NumericLiteral<'_>, cx: &EarlyContext<'_>, span: Span) {
        let (lint, message, try_msg) = self.lint_and_text();
        #[expect(clippy::collapsible_span_lint_calls, reason = "rust-clippy#7797")]
        span_lint_and_then(cx, lint, span, message, |diag| {
            diag.span_suggestion(span, try_msg, num_lit.format(), Applicability::MaybeIncorrect);
        });
    }
}

fn is_literal_uuid_formatted(num_lit: &NumericLiteral<'_>) -> bool {
    // Length of each UUID hyphenated group in hex digits.
    const UUID_GROUP_LENS: [usize; 5] = [8, 4, 4, 4, 12];

    if num_lit.radix != Radix::Hexadecimal {
        return false;
    }

    // UUIDs should not have a fraction
    if num_lit.fraction.is_some() {
        return false;
    }

    let group_sizes: Vec<usize> = num_lit.integer.split('_').map(str::len).collect();
    UUID_GROUP_LENS.len() == group_sizes.len() && iter::zip(&UUID_GROUP_LENS, &group_sizes).all(|(&a, &b)| a == b)
}

/// Returns the size of the digit groups (or None if ungrouped) if successful,
/// otherwise returns a `WarningType` for linting.
fn get_group_size<'a>(
    groups: impl Iterator<Item = &'a str>,
    radix: Radix,
    lint_unreadable: bool,
) -> Result<Option<usize>, WarningType> {
    let mut groups = groups.map(str::len);

    let first = groups.next().expect("At least one group");

    if (radix == Radix::Binary || radix == Radix::Octal || radix == Radix::Hexadecimal)
        && let Some(second_size) = groups.next()
        && (!groups.all(|i| i == second_size) || first > second_size)
    {
        return Err(WarningType::UnusualByteGroupings);
    }

    if let Some(second) = groups.next() {
        if !groups.all(|x| x == second) || first > second {
            Err(WarningType::InconsistentDigitGrouping)
        } else if second > 4 {
            Err(WarningType::LargeDigitGroups)
        } else {
            Ok(Some(second))
        }
    } else if first > 5 && lint_unreadable {
        Err(WarningType::UnreadableLiteral)
    } else {
        Ok(None)
    }
}
