//! Lints concerned with the grouping of digits with underscores in integral or
//! floating-point literal expressions.

use crate::utils::{in_macro, snippet_opt, span_lint_and_sugg};
use if_chain::if_chain;
use rustc::lint::{in_external_macro, EarlyContext, EarlyLintPass, LintArray, LintContext, LintPass};
use rustc::{declare_lint_pass, declare_tool_lint, impl_lint_pass};
use rustc_errors::Applicability;
use syntax::ast::*;
use syntax_pos;

declare_clippy_lint! {
    /// **What it does:** Warns if a long integral or floating-point constant does
    /// not contain underscores.
    ///
    /// **Why is this bad?** Reading long numbers is difficult without separators.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    ///
    /// ```rust
    /// let x: u64 = 61864918973511;
    /// ```
    pub UNREADABLE_LITERAL,
    style,
    "long integer literal without underscores"
}

declare_clippy_lint! {
    /// **What it does:** Warns for mistyped suffix in literals
    ///
    /// **Why is this bad?** This is most probably a typo
    ///
    /// **Known problems:**
    /// - Recommends a signed suffix, even though the number might be too big and an unsigned
    ///   suffix is required
    /// - Does not match on `_128` since that is a valid grouping for decimal and octal numbers
    ///
    /// **Example:**
    ///
    /// ```rust
    /// 2_32;
    /// ```
    pub MISTYPED_LITERAL_SUFFIXES,
    correctness,
    "mistyped literal suffix"
}

declare_clippy_lint! {
    /// **What it does:** Warns if an integral or floating-point constant is
    /// grouped inconsistently with underscores.
    ///
    /// **Why is this bad?** Readers may incorrectly interpret inconsistently
    /// grouped digits.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    ///
    /// ```rust
    /// let x: u64 = 618_64_9189_73_511;
    /// ```
    pub INCONSISTENT_DIGIT_GROUPING,
    style,
    "integer literals with digits grouped inconsistently"
}

declare_clippy_lint! {
    /// **What it does:** Warns if the digits of an integral or floating-point
    /// constant are grouped into groups that
    /// are too large.
    ///
    /// **Why is this bad?** Negatively impacts readability.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    ///
    /// ```rust
    /// let x: u64 = 6186491_8973511;
    /// ```
    pub LARGE_DIGIT_GROUPS,
    pedantic,
    "grouping digits into groups that are too large"
}

declare_clippy_lint! {
    /// **What it does:** Warns if there is a better representation for a numeric literal.
    ///
    /// **Why is this bad?** Especially for big powers of 2 a hexadecimal representation is more
    /// readable than a decimal representation.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    ///
    /// `255` => `0xFF`
    /// `65_535` => `0xFFFF`
    /// `4_042_322_160` => `0xF0F0_F0F0`
    pub DECIMAL_LITERAL_REPRESENTATION,
    restriction,
    "using decimal representation when hexadecimal would be better"
}

#[derive(Debug, PartialEq)]
pub(super) enum Radix {
    Binary,
    Octal,
    Decimal,
    Hexadecimal,
}

impl Radix {
    /// Returns a reasonable digit group size for this radix.
    #[must_use]
    crate fn suggest_grouping(&self) -> usize {
        match *self {
            Self::Binary | Self::Hexadecimal => 4,
            Self::Octal | Self::Decimal => 3,
        }
    }
}

#[derive(Debug)]
pub(super) struct DigitInfo<'a> {
    /// Characters of a literal between the radix prefix and type suffix.
    crate digits: &'a str,
    /// Which radix the literal was represented in.
    crate radix: Radix,
    /// The radix prefix, if present.
    crate prefix: Option<&'a str>,
    /// The type suffix, including preceding underscore if present.
    crate suffix: Option<&'a str>,
    /// True for floating-point literals.
    crate float: bool,
}

impl<'a> DigitInfo<'a> {
    #[must_use]
    crate fn new(lit: &'a str, float: bool) -> Self {
        // Determine delimiter for radix prefix, if present, and radix.
        let radix = if lit.starts_with("0x") {
            Radix::Hexadecimal
        } else if lit.starts_with("0b") {
            Radix::Binary
        } else if lit.starts_with("0o") {
            Radix::Octal
        } else {
            Radix::Decimal
        };

        // Grab part of the literal after prefix, if present.
        let (prefix, sans_prefix) = if let Radix::Decimal = radix {
            (None, lit)
        } else {
            let (p, s) = lit.split_at(2);
            (Some(p), s)
        };

        let len = sans_prefix.len();
        let mut last_d = '\0';
        for (d_idx, d) in sans_prefix.char_indices() {
            let suffix_start = if last_d == '_' { d_idx - 1 } else { d_idx };
            if float
                && (d == 'f'
                    || is_possible_float_suffix_index(&sans_prefix, suffix_start, len)
                    || ((d == 'E' || d == 'e') && !has_possible_float_suffix(&sans_prefix)))
                || !float && (d == 'i' || d == 'u' || is_possible_suffix_index(&sans_prefix, suffix_start, len))
            {
                let (digits, suffix) = sans_prefix.split_at(suffix_start);
                return Self {
                    digits,
                    radix,
                    prefix,
                    suffix: Some(suffix),
                    float,
                };
            }
            last_d = d
        }

        // No suffix found
        Self {
            digits: sans_prefix,
            radix,
            prefix,
            suffix: None,
            float,
        }
    }

    fn split_digit_parts(&self) -> (&str, Option<&str>, Option<(char, &str)>) {
        let digits = self.digits;

        let mut integer = digits;
        let mut fraction = None;
        let mut exponent = None;

        if self.float {
            for (i, c) in digits.char_indices() {
                match c {
                    '.' => {
                        integer = &digits[..i];
                        fraction = Some(&digits[i + 1..]);
                    },
                    'e' | 'E' => {
                        if integer.len() > i {
                            integer = &digits[..i];
                        } else {
                            fraction = Some(&digits[integer.len() + 1..i]);
                        };
                        exponent = Some((c, &digits[i + 1..]));
                        break;
                    },
                    _ => {},
                }
            }
        }

        (integer, fraction, exponent)
    }

    /// Returns literal formatted in a sensible way.
    crate fn grouping_hint(&self) -> String {
        let mut output = String::new();

        if let Some(prefix) = self.prefix {
            output.push_str(prefix);
        }

        let group_size = self.radix.suggest_grouping();

        let (integer, fraction, exponent) = &self.split_digit_parts();

        Self::group_digits(&mut output, integer, group_size, true, self.radix == Radix::Hexadecimal);

        if let Some(fraction) = fraction {
            output.push('.');
            Self::group_digits(&mut output, fraction, group_size, false, false);
        }

        if let Some((separator, exponent)) = exponent {
            output.push(*separator);
            Self::group_digits(&mut output, exponent, group_size, true, false);
        }

        if let Some(suffix) = self.suffix {
            if self.float && is_mistyped_float_suffix(suffix) {
                output.push_str("_f");
                output.push_str(&suffix[1..]);
            } else if is_mistyped_suffix(suffix) {
                output.push_str("_i");
                output.push_str(&suffix[1..]);
            } else {
                output.push_str(suffix);
            }
        }

        output
    }

    fn group_digits(output: &mut String, input: &str, group_size: usize, partial_group_first: bool, pad: bool) {
        debug_assert!(group_size > 0);

        let mut digits = input.chars().filter(|&c| c != '_');

        let first_group_size;

        if partial_group_first {
            first_group_size = (digits.clone().count() + group_size - 1) % group_size + 1;
            if pad {
                for _ in 0..group_size - first_group_size {
                    output.push('0');
                }
            }
        } else {
            first_group_size = group_size;
        }

        for _ in 0..first_group_size {
            if let Some(digit) = digits.next() {
                output.push(digit);
            }
        }

        for (c, i) in digits.zip((0..group_size).cycle()) {
            if i == 0 {
                output.push('_');
            }
            output.push(c);
        }
    }
}

enum WarningType {
    UnreadableLiteral,
    InconsistentDigitGrouping,
    LargeDigitGroups,
    DecimalRepresentation,
    MistypedLiteralSuffix,
}

impl WarningType {
    crate fn display(&self, grouping_hint: &str, cx: &EarlyContext<'_>, span: syntax_pos::Span) {
        match self {
            Self::MistypedLiteralSuffix => span_lint_and_sugg(
                cx,
                MISTYPED_LITERAL_SUFFIXES,
                span,
                "mistyped literal suffix",
                "did you mean to write",
                grouping_hint.to_string(),
                Applicability::MaybeIncorrect,
            ),
            Self::UnreadableLiteral => span_lint_and_sugg(
                cx,
                UNREADABLE_LITERAL,
                span,
                "long literal lacking separators",
                "consider",
                grouping_hint.to_owned(),
                Applicability::MachineApplicable,
            ),
            Self::LargeDigitGroups => span_lint_and_sugg(
                cx,
                LARGE_DIGIT_GROUPS,
                span,
                "digit groups should be smaller",
                "consider",
                grouping_hint.to_owned(),
                Applicability::MachineApplicable,
            ),
            Self::InconsistentDigitGrouping => span_lint_and_sugg(
                cx,
                INCONSISTENT_DIGIT_GROUPING,
                span,
                "digits grouped inconsistently by underscores",
                "consider",
                grouping_hint.to_owned(),
                Applicability::MachineApplicable,
            ),
            Self::DecimalRepresentation => span_lint_and_sugg(
                cx,
                DECIMAL_LITERAL_REPRESENTATION,
                span,
                "integer literal has a better hexadecimal representation",
                "consider",
                grouping_hint.to_owned(),
                Applicability::MachineApplicable,
            ),
        };
    }
}

declare_lint_pass!(LiteralDigitGrouping => [
    UNREADABLE_LITERAL,
    INCONSISTENT_DIGIT_GROUPING,
    LARGE_DIGIT_GROUPS,
    MISTYPED_LITERAL_SUFFIXES,
]);

impl EarlyLintPass for LiteralDigitGrouping {
    fn check_expr(&mut self, cx: &EarlyContext<'_>, expr: &Expr) {
        if in_external_macro(cx.sess(), expr.span) {
            return;
        }

        if let ExprKind::Lit(ref lit) = expr.kind {
            Self::check_lit(cx, lit)
        }
    }
}

impl LiteralDigitGrouping {
    fn check_lit(cx: &EarlyContext<'_>, lit: &Lit) {
        let in_macro = in_macro(lit.span);

        if_chain! {
            if let Some(src) = snippet_opt(cx, lit.span);
            if let Some(firstch) = src.chars().next();
            if char::is_digit(firstch, 10);
            then {

                let digit_info = match lit.kind {
                        LitKind::Int(..) => DigitInfo::new(&src, false),
                        LitKind::Float(..) => DigitInfo::new(&src, true),
                        _ => return,
                };

                let result = (|| {
                    if let Some(suffix) = digit_info.suffix {
                        if is_mistyped_suffix(suffix) {
                            return Err(WarningType::MistypedLiteralSuffix);
                        }
                    }

                    let (integer, fraction, _) = digit_info.split_digit_parts();

                    let integral_group_size = Self::get_group_size(integer.split('_'), in_macro)?;
                    if let Some(fraction) = fraction {
                        let fractional_group_size = Self::get_group_size(fraction.rsplit('_'), in_macro)?;

                        let consistent = Self::parts_consistent(integral_group_size,
                                                                fractional_group_size,
                                                                integer.len(),
                                                                fraction.len());
                        if !consistent {
                            return Err(WarningType::InconsistentDigitGrouping);
                        };
                    }
                    Ok(())
                })();


                if let Err(warning_type) = result {
                    warning_type.display(&digit_info.grouping_hint(), cx, lit.span)
                }
            }
        }
    }

    /// Given the sizes of the digit groups of both integral and fractional
    /// parts, and the length
    /// of both parts, determine if the digits have been grouped consistently.
    #[must_use]
    fn parts_consistent(
        int_group_size: Option<usize>,
        frac_group_size: Option<usize>,
        int_size: usize,
        frac_size: usize,
    ) -> bool {
        match (int_group_size, frac_group_size) {
            // No groups on either side of decimal point - trivially consistent.
            (None, None) => true,
            // Integral part has grouped digits, fractional part does not.
            (Some(int_group_size), None) => frac_size <= int_group_size,
            // Fractional part has grouped digits, integral part does not.
            (None, Some(frac_group_size)) => int_size <= frac_group_size,
            // Both parts have grouped digits. Groups should be the same size.
            (Some(int_group_size), Some(frac_group_size)) => int_group_size == frac_group_size,
        }
    }

    /// Returns the size of the digit groups (or None if ungrouped) if successful,
    /// otherwise returns a `WarningType` for linting.
    fn get_group_size<'a>(groups: impl Iterator<Item = &'a str>, in_macro: bool) -> Result<Option<usize>, WarningType> {
        let mut groups = groups.map(str::len);

        let first = groups.next().expect("At least one group");

        if let Some(second) = groups.next() {
            if !groups.all(|x| x == second) || first > second {
                Err(WarningType::InconsistentDigitGrouping)
            } else if second > 4 {
                Err(WarningType::LargeDigitGroups)
            } else {
                Ok(Some(second))
            }
        } else if first > 5 && !in_macro {
            Err(WarningType::UnreadableLiteral)
        } else {
            Ok(None)
        }
    }
}

#[allow(clippy::module_name_repetitions)]
#[derive(Copy, Clone)]
pub struct DecimalLiteralRepresentation {
    threshold: u64,
}

impl_lint_pass!(DecimalLiteralRepresentation => [DECIMAL_LITERAL_REPRESENTATION]);

impl EarlyLintPass for DecimalLiteralRepresentation {
    fn check_expr(&mut self, cx: &EarlyContext<'_>, expr: &Expr) {
        if in_external_macro(cx.sess(), expr.span) {
            return;
        }

        if let ExprKind::Lit(ref lit) = expr.kind {
            self.check_lit(cx, lit)
        }
    }
}

impl DecimalLiteralRepresentation {
    #[must_use]
    pub fn new(threshold: u64) -> Self {
        Self { threshold }
    }
    fn check_lit(self, cx: &EarlyContext<'_>, lit: &Lit) {
        // Lint integral literals.
        if_chain! {
            if let LitKind::Int(val, _) = lit.kind;
            if let Some(src) = snippet_opt(cx, lit.span);
            if let Some(firstch) = src.chars().next();
            if char::is_digit(firstch, 10);
            let digit_info = DigitInfo::new(&src, false);
            if digit_info.radix == Radix::Decimal;
            if val >= u128::from(self.threshold);
            then {
                let hex = format!("{:#X}", val);
                let digit_info = DigitInfo::new(&hex, false);
                let _ = Self::do_lint(digit_info.digits).map_err(|warning_type| {
                    warning_type.display(&digit_info.grouping_hint(), cx, lit.span)
                });
            }
        }
    }

    fn do_lint(digits: &str) -> Result<(), WarningType> {
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
                return Err(WarningType::DecimalRepresentation);
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
                return Err(WarningType::DecimalRepresentation);
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
                return Err(WarningType::DecimalRepresentation);
            }
        }

        Ok(())
    }
}

#[must_use]
fn is_mistyped_suffix(suffix: &str) -> bool {
    ["_8", "_16", "_32", "_64"].contains(&suffix)
}

#[must_use]
fn is_possible_suffix_index(lit: &str, idx: usize, len: usize) -> bool {
    ((len > 3 && idx == len - 3) || (len > 2 && idx == len - 2)) && is_mistyped_suffix(lit.split_at(idx).1)
}

#[must_use]
fn is_mistyped_float_suffix(suffix: &str) -> bool {
    ["_32", "_64"].contains(&suffix)
}

#[must_use]
fn is_possible_float_suffix_index(lit: &str, idx: usize, len: usize) -> bool {
    (len > 3 && idx == len - 3) && is_mistyped_float_suffix(lit.split_at(idx).1)
}

#[must_use]
fn has_possible_float_suffix(lit: &str) -> bool {
    lit.ends_with("_32") || lit.ends_with("_64")
}
