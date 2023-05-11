//! Lints concerned with the grouping of digits with underscores in integral or
//! floating-point literal expressions.

use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::numeric_literal::{NumericLiteral, Radix};
use clippy_utils::source::snippet_opt;
use if_chain::if_chain;
use rustc_ast::ast::{Expr, ExprKind, LitKind};
use rustc_ast::token;
use rustc_errors::Applicability;
use rustc_lint::{EarlyContext, EarlyLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::Span;
use std::iter;

declare_clippy_lint! {
    /// ### What it does
    /// Warns if a long integral or floating-point constant does
    /// not contain underscores.
    ///
    /// ### Why is this bad?
    /// Reading long numbers is difficult without separators.
    ///
    /// ### Example
    /// ```rust
    /// # let _: u64 =
    /// 61864918973511
    /// # ;
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// # let _: u64 =
    /// 61_864_918_973_511
    /// # ;
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub UNREADABLE_LITERAL,
    pedantic,
    "long literal without underscores"
}

declare_clippy_lint! {
    /// ### What it does
    /// Warns for mistyped suffix in literals
    ///
    /// ### Why is this bad?
    /// This is most probably a typo
    ///
    /// ### Known problems
    /// - Does not match on integers too large to fit in the corresponding unsigned type
    /// - Does not match on `_127` since that is a valid grouping for decimal and octal numbers
    ///
    /// ### Example
    /// ```ignore
    /// `2_32` => `2_i32`
    /// `250_8 => `250_u8`
    /// ```
    #[clippy::version = "1.30.0"]
    pub MISTYPED_LITERAL_SUFFIXES,
    correctness,
    "mistyped literal suffix"
}

declare_clippy_lint! {
    /// ### What it does
    /// Warns if an integral or floating-point constant is
    /// grouped inconsistently with underscores.
    ///
    /// ### Why is this bad?
    /// Readers may incorrectly interpret inconsistently
    /// grouped digits.
    ///
    /// ### Example
    /// ```rust
    /// # let _: u64 =
    /// 618_64_9189_73_511
    /// # ;
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// # let _: u64 =
    /// 61_864_918_973_511
    /// # ;
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub INCONSISTENT_DIGIT_GROUPING,
    style,
    "integer literals with digits grouped inconsistently"
}

declare_clippy_lint! {
    /// ### What it does
    /// Warns if hexadecimal or binary literals are not grouped
    /// by nibble or byte.
    ///
    /// ### Why is this bad?
    /// Negatively impacts readability.
    ///
    /// ### Example
    /// ```rust
    /// let x: u32 = 0xFFF_FFF;
    /// let y: u8 = 0b01_011_101;
    /// ```
    #[clippy::version = "1.49.0"]
    pub UNUSUAL_BYTE_GROUPINGS,
    style,
    "binary or hex literals that aren't grouped by four"
}

declare_clippy_lint! {
    /// ### What it does
    /// Warns if the digits of an integral or floating-point
    /// constant are grouped into groups that
    /// are too large.
    ///
    /// ### Why is this bad?
    /// Negatively impacts readability.
    ///
    /// ### Example
    /// ```rust
    /// let x: u64 = 6186491_8973511;
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub LARGE_DIGIT_GROUPS,
    pedantic,
    "grouping digits into groups that are too large"
}

declare_clippy_lint! {
    /// ### What it does
    /// Warns if there is a better representation for a numeric literal.
    ///
    /// ### Why is this bad?
    /// Especially for big powers of 2 a hexadecimal representation is more
    /// readable than a decimal representation.
    ///
    /// ### Example
    /// ```text
    /// `255` => `0xFF`
    /// `65_535` => `0xFFFF`
    /// `4_042_322_160` => `0xF0F0_F0F0`
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub DECIMAL_LITERAL_REPRESENTATION,
    restriction,
    "using decimal representation when hexadecimal would be better"
}

enum WarningType {
    UnreadableLiteral,
    InconsistentDigitGrouping,
    LargeDigitGroups,
    DecimalRepresentation,
    MistypedLiteralSuffix,
    UnusualByteGroupings,
}

impl WarningType {
    fn display(&self, suggested_format: String, cx: &EarlyContext<'_>, span: rustc_span::Span) {
        match self {
            Self::MistypedLiteralSuffix => span_lint_and_sugg(
                cx,
                MISTYPED_LITERAL_SUFFIXES,
                span,
                "mistyped literal suffix",
                "did you mean to write",
                suggested_format,
                Applicability::MaybeIncorrect,
            ),
            Self::UnreadableLiteral => span_lint_and_sugg(
                cx,
                UNREADABLE_LITERAL,
                span,
                "long literal lacking separators",
                "consider",
                suggested_format,
                Applicability::MachineApplicable,
            ),
            Self::LargeDigitGroups => span_lint_and_sugg(
                cx,
                LARGE_DIGIT_GROUPS,
                span,
                "digit groups should be smaller",
                "consider",
                suggested_format,
                Applicability::MachineApplicable,
            ),
            Self::InconsistentDigitGrouping => span_lint_and_sugg(
                cx,
                INCONSISTENT_DIGIT_GROUPING,
                span,
                "digits grouped inconsistently by underscores",
                "consider",
                suggested_format,
                Applicability::MachineApplicable,
            ),
            Self::DecimalRepresentation => span_lint_and_sugg(
                cx,
                DECIMAL_LITERAL_REPRESENTATION,
                span,
                "integer literal has a better hexadecimal representation",
                "consider",
                suggested_format,
                Applicability::MachineApplicable,
            ),
            Self::UnusualByteGroupings => span_lint_and_sugg(
                cx,
                UNUSUAL_BYTE_GROUPINGS,
                span,
                "digits of hex, binary or octal literal not in groups of equal size",
                "consider",
                suggested_format,
                Applicability::MachineApplicable,
            ),
        };
    }
}

#[derive(Copy, Clone)]
pub struct LiteralDigitGrouping {
    lint_fraction_readability: bool,
}

impl_lint_pass!(LiteralDigitGrouping => [
    UNREADABLE_LITERAL,
    INCONSISTENT_DIGIT_GROUPING,
    LARGE_DIGIT_GROUPS,
    MISTYPED_LITERAL_SUFFIXES,
    UNUSUAL_BYTE_GROUPINGS,
]);

impl EarlyLintPass for LiteralDigitGrouping {
    fn check_expr(&mut self, cx: &EarlyContext<'_>, expr: &Expr) {
        if in_external_macro(cx.sess(), expr.span) {
            return;
        }

        if let ExprKind::Lit(lit) = expr.kind {
            self.check_lit(cx, lit, expr.span);
        }
    }
}

// Length of each UUID hyphenated group in hex digits.
const UUID_GROUP_LENS: [usize; 5] = [8, 4, 4, 4, 12];

impl LiteralDigitGrouping {
    pub fn new(lint_fraction_readability: bool) -> Self {
        Self {
            lint_fraction_readability,
        }
    }

    fn check_lit(self, cx: &EarlyContext<'_>, lit: token::Lit, span: Span) {
        if_chain! {
            if let Some(src) = snippet_opt(cx, span);
            if let Ok(lit_kind) = LitKind::from_token_lit(lit);
            if let Some(mut num_lit) = NumericLiteral::from_lit_kind(&src, &lit_kind);
            then {
                if !Self::check_for_mistyped_suffix(cx, span, &mut num_lit) {
                    return;
                }

                if Self::is_literal_uuid_formatted(&mut num_lit) {
                    return;
                }

                let result = (|| {

                    let integral_group_size = Self::get_group_size(num_lit.integer.split('_'), num_lit.radix, true)?;
                    if let Some(fraction) = num_lit.fraction {
                        let fractional_group_size = Self::get_group_size(
                            fraction.rsplit('_'),
                            num_lit.radix,
                            self.lint_fraction_readability)?;

                        let consistent = Self::parts_consistent(integral_group_size,
                                                                fractional_group_size,
                                                                num_lit.integer.len(),
                                                                fraction.len());
                        if !consistent {
                            return Err(WarningType::InconsistentDigitGrouping);
                        };
                    }

                    Ok(())
                })();


                if let Err(warning_type) = result {
                    let should_warn = match warning_type {
                        | WarningType::UnreadableLiteral
                        | WarningType::InconsistentDigitGrouping
                        | WarningType::UnusualByteGroupings
                        | WarningType::LargeDigitGroups => {
                            !span.from_expansion()
                        }
                        WarningType::DecimalRepresentation | WarningType::MistypedLiteralSuffix => {
                            true
                        }
                    };
                    if should_warn {
                        warning_type.display(num_lit.format(), cx, span);
                    }
                }
            }
        }
    }

    // Returns `false` if the check fails
    fn check_for_mistyped_suffix(
        cx: &EarlyContext<'_>,
        span: rustc_span::Span,
        num_lit: &mut NumericLiteral<'_>,
    ) -> bool {
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
            let mut sugg = num_lit.format();
            sugg.push('_');
            sugg.push(missing_char);
            sugg.push_str(last_group);
            WarningType::MistypedLiteralSuffix.display(sugg, cx, span);
            false
        } else {
            true
        }
    }

    /// Checks whether the numeric literal matches the formatting of a UUID.
    ///
    /// Returns `true` if the radix is hexadecimal, and the groups match the
    /// UUID format of 8-4-4-4-12.
    fn is_literal_uuid_formatted(num_lit: &mut NumericLiteral<'_>) -> bool {
        if num_lit.radix != Radix::Hexadecimal {
            return false;
        }

        // UUIDs should not have a fraction
        if num_lit.fraction.is_some() {
            return false;
        }

        let group_sizes: Vec<usize> = num_lit.integer.split('_').map(str::len).collect();
        if UUID_GROUP_LENS.len() == group_sizes.len() {
            iter::zip(&UUID_GROUP_LENS, &group_sizes).all(|(&a, &b)| a == b)
        } else {
            false
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
    fn get_group_size<'a>(
        groups: impl Iterator<Item = &'a str>,
        radix: Radix,
        lint_unreadable: bool,
    ) -> Result<Option<usize>, WarningType> {
        let mut groups = groups.map(str::len);

        let first = groups.next().expect("At least one group");

        if radix == Radix::Binary || radix == Radix::Octal || radix == Radix::Hexadecimal {
            if let Some(second_size) = groups.next() {
                if !groups.all(|i| i == second_size) || first > second_size {
                    return Err(WarningType::UnusualByteGroupings);
                }
            }
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
}

#[expect(clippy::module_name_repetitions)]
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

        if let ExprKind::Lit(lit) = expr.kind {
            self.check_lit(cx, lit, expr.span);
        }
    }
}

impl DecimalLiteralRepresentation {
    #[must_use]
    pub fn new(threshold: u64) -> Self {
        Self { threshold }
    }
    fn check_lit(self, cx: &EarlyContext<'_>, lit: token::Lit, span: Span) {
        // Lint integral literals.
        if_chain! {
            if let Ok(lit_kind) = LitKind::from_token_lit(lit);
            if let LitKind::Int(val, _) = lit_kind;
            if let Some(src) = snippet_opt(cx, span);
            if let Some(num_lit) = NumericLiteral::from_lit_kind(&src, &lit_kind);
            if num_lit.radix == Radix::Decimal;
            if val >= u128::from(self.threshold);
            then {
                let hex = format!("{val:#X}");
                let num_lit = NumericLiteral::new(&hex, num_lit.suffix, false);
                let _: Result<(), ()> = Self::do_lint(num_lit.integer).map_err(|warning_type| {
                    warning_type.display(num_lit.format(), cx, span);
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
