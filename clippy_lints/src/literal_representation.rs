//! Lints concerned with the grouping of digits with underscores in integral or
//! floating-point literal expressions.

use crate::utils::{in_macro, snippet_opt, span_lint_and_sugg};
use if_chain::if_chain;
use rustc::lint::in_external_macro;
use rustc_errors::Applicability;
use rustc_lint::{EarlyContext, EarlyLintPass, LintContext};
use rustc_session::{declare_lint_pass, declare_tool_lint, impl_lint_pass};
use syntax::ast::*;

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
    fn suggest_grouping(&self) -> usize {
        match *self {
            Self::Binary | Self::Hexadecimal => 4,
            Self::Octal | Self::Decimal => 3,
        }
    }
}

/// A helper method to format numeric literals with digit grouping.
/// `lit` must be a valid numeric literal without suffix.
pub fn format_numeric_literal(lit: &str, type_suffix: Option<&str>, float: bool) -> String {
    NumericLiteral::new(lit, type_suffix, float).format()
}

#[derive(Debug)]
pub(super) struct NumericLiteral<'a> {
    /// Which radix the literal was represented in.
    radix: Radix,
    /// The radix prefix, if present.
    prefix: Option<&'a str>,

    /// The integer part of the number.
    integer: &'a str,
    /// The fraction part of the number.
    fraction: Option<&'a str>,
    /// The character used as exponent seperator (b'e' or b'E') and the exponent part.
    exponent: Option<(char, &'a str)>,

    /// The type suffix, including preceding underscore if present.
    suffix: Option<&'a str>,
}

impl<'a> NumericLiteral<'a> {
    fn from_lit(src: &'a str, lit: &Lit) -> Option<NumericLiteral<'a>> {
        if lit.kind.is_numeric() && src.chars().next().map_or(false, |c| c.is_digit(10)) {
            let (unsuffixed, suffix) = split_suffix(&src, &lit.kind);
            let float = if let LitKind::Float(..) = lit.kind { true } else { false };
            Some(NumericLiteral::new(unsuffixed, suffix, float))
        } else {
            None
        }
    }

    #[must_use]
    fn new(lit: &'a str, suffix: Option<&'a str>, float: bool) -> Self {
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
        let (prefix, mut sans_prefix) = if let Radix::Decimal = radix {
            (None, lit)
        } else {
            let (p, s) = lit.split_at(2);
            (Some(p), s)
        };

        if suffix.is_some() && sans_prefix.ends_with('_') {
            // The '_' before the suffix isn't part of the digits
            sans_prefix = &sans_prefix[..sans_prefix.len() - 1];
        }

        let (integer, fraction, exponent) = Self::split_digit_parts(sans_prefix, float);

        Self {
            radix,
            prefix,
            integer,
            fraction,
            exponent,
            suffix,
        }
    }

    fn split_digit_parts(digits: &str, float: bool) -> (&str, Option<&str>, Option<(char, &str)>) {
        let mut integer = digits;
        let mut fraction = None;
        let mut exponent = None;

        if float {
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
    fn format(&self) -> String {
        let mut output = String::new();

        if let Some(prefix) = self.prefix {
            output.push_str(prefix);
        }

        let group_size = self.radix.suggest_grouping();

        Self::group_digits(
            &mut output,
            self.integer,
            group_size,
            true,
            self.radix == Radix::Hexadecimal,
        );

        if let Some(fraction) = self.fraction {
            output.push('.');
            Self::group_digits(&mut output, fraction, group_size, false, false);
        }

        if let Some((separator, exponent)) = self.exponent {
            output.push(separator);
            Self::group_digits(&mut output, exponent, group_size, true, false);
        }

        if let Some(suffix) = self.suffix {
            output.push('_');
            output.push_str(suffix);
        }

        output
    }

    fn group_digits(output: &mut String, input: &str, group_size: usize, partial_group_first: bool, pad: bool) {
        debug_assert!(group_size > 0);

        let mut digits = input.chars().filter(|&c| c != '_');

        let first_group_size;

        if partial_group_first {
            first_group_size = (digits.clone().count() - 1) % group_size + 1;
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

fn split_suffix<'a>(src: &'a str, lit_kind: &LitKind) -> (&'a str, Option<&'a str>) {
    debug_assert!(lit_kind.is_numeric());
    if let Some(suffix_length) = lit_suffix_length(lit_kind) {
        let (unsuffixed, suffix) = src.split_at(src.len() - suffix_length);
        (unsuffixed, Some(suffix))
    } else {
        (src, None)
    }
}

fn lit_suffix_length(lit_kind: &LitKind) -> Option<usize> {
    debug_assert!(lit_kind.is_numeric());
    let suffix = match lit_kind {
        LitKind::Int(_, int_lit_kind) => match int_lit_kind {
            LitIntType::Signed(int_ty) => Some(int_ty.name_str()),
            LitIntType::Unsigned(uint_ty) => Some(uint_ty.name_str()),
            LitIntType::Unsuffixed => None,
        },
        LitKind::Float(_, float_lit_kind) => match float_lit_kind {
            LitFloatType::Suffixed(float_ty) => Some(float_ty.name_str()),
            LitFloatType::Unsuffixed => None,
        },
        _ => None,
    };

    suffix.map(str::len)
}

enum WarningType {
    UnreadableLiteral,
    InconsistentDigitGrouping,
    LargeDigitGroups,
    DecimalRepresentation,
    MistypedLiteralSuffix,
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
        if_chain! {
            if let Some(src) = snippet_opt(cx, lit.span);
            if let Some(mut num_lit) = NumericLiteral::from_lit(&src, &lit);
            then {
                if !Self::check_for_mistyped_suffix(cx, lit.span, &mut num_lit) {
                    return;
                }

                let result = (|| {

                    let integral_group_size = Self::get_group_size(num_lit.integer.split('_'))?;
                    if let Some(fraction) = num_lit.fraction {
                        let fractional_group_size = Self::get_group_size(fraction.rsplit('_'))?;

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
                        | WarningType::LargeDigitGroups => {
                            !in_macro(lit.span)
                        }
                        WarningType::DecimalRepresentation | WarningType::MistypedLiteralSuffix => {
                            true
                        }
                    };
                    if should_warn {
                        warning_type.display(num_lit.format(), cx, lit.span)
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

        let (part, mistyped_suffixes, missing_char) = if let Some((_, exponent)) = &mut num_lit.exponent {
            (exponent, &["32", "64"][..], 'f')
        } else if let Some(fraction) = &mut num_lit.fraction {
            (fraction, &["32", "64"][..], 'f')
        } else {
            (&mut num_lit.integer, &["8", "16", "32", "64"][..], 'i')
        };

        let mut split = part.rsplit('_');
        let last_group = split.next().expect("At least one group");
        if split.next().is_some() && mistyped_suffixes.contains(&last_group) {
            *part = &part[..part.len() - last_group.len()];
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
    fn get_group_size<'a>(groups: impl Iterator<Item = &'a str>) -> Result<Option<usize>, WarningType> {
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
        } else if first > 5 {
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
            if let Some(num_lit) = NumericLiteral::from_lit(&src, &lit);
            if num_lit.radix == Radix::Decimal;
            if val >= u128::from(self.threshold);
            then {
                let hex = format!("{:#X}", val);
                let num_lit = NumericLiteral::new(&hex, num_lit.suffix, false);
                let _ = Self::do_lint(num_lit.integer).map_err(|warning_type| {
                    warning_type.display(num_lit.format(), cx, lit.span)
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
