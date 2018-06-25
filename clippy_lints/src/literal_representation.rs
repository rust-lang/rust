//! Lints concerned with the grouping of digits with underscores in integral or
//! floating-point literal expressions.

use rustc::lint::*;
use syntax::ast::*;
use syntax_pos;
use crate::utils::{in_external_macro, snippet_opt, span_lint_and_sugg};

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
/// 61864918973511
/// ```
declare_clippy_lint! {
    pub UNREADABLE_LITERAL,
    style,
    "long integer literal without underscores"
}

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
/// 618_64_9189_73_511
/// ```
declare_clippy_lint! {
    pub INCONSISTENT_DIGIT_GROUPING,
    style,
    "integer literals with digits grouped inconsistently"
}

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
/// 6186491_8973511
/// ```
declare_clippy_lint! {
    pub LARGE_DIGIT_GROUPS,
    style,
    "grouping digits into groups that are too large"
}

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
declare_clippy_lint! {
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
    /// Return a reasonable digit group size for this radix.
    crate fn suggest_grouping(&self) -> usize {
        match *self {
            Radix::Binary | Radix::Hexadecimal => 4,
            Radix::Octal | Radix::Decimal => 3,
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

        let mut last_d = '\0';
        for (d_idx, d) in sans_prefix.char_indices() {
            if !float && (d == 'i' || d == 'u') || float && (d == 'f' || d == 'e' || d == 'E') {
                let suffix_start = if last_d == '_' { d_idx - 1 } else { d_idx };
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

    /// Returns digits grouped in a sensible way.
    crate fn grouping_hint(&self) -> String {
        let group_size = self.radix.suggest_grouping();
        if self.digits.contains('.') {
            let mut parts = self.digits.split('.');
            let int_part_hint = parts
                .next()
                .expect("split always returns at least one element")
                .chars()
                .rev()
                .filter(|&c| c != '_')
                .collect::<Vec<_>>()
                .chunks(group_size)
                .map(|chunk| chunk.into_iter().rev().collect())
                .rev()
                .collect::<Vec<String>>()
                .join("_");
            let frac_part_hint = parts
                .next()
                .expect("already checked that there is a `.`")
                .chars()
                .filter(|&c| c != '_')
                .collect::<Vec<_>>()
                .chunks(group_size)
                .map(|chunk| chunk.into_iter().collect())
                .collect::<Vec<String>>()
                .join("_");
            format!(
                "{}.{}{}",
                int_part_hint,
                frac_part_hint,
                self.suffix.unwrap_or("")
            )
        } else {
            let filtered_digits_vec = self.digits
                .chars()
                .filter(|&c| c != '_')
                .rev()
                .collect::<Vec<_>>();
            let mut hint = filtered_digits_vec
                .chunks(group_size)
                .map(|chunk| chunk.into_iter().rev().collect())
                .rev()
                .collect::<Vec<String>>()
                .join("_");
            // Forces hexadecimal values to be grouped by 4 being filled with zeroes (e.g 0x00ab_cdef)
            let nb_digits_to_fill = filtered_digits_vec.len() % 4;
            if self.radix == Radix::Hexadecimal && nb_digits_to_fill != 0 {
                hint = format!("{:0>4}{}", &hint[..nb_digits_to_fill], &hint[nb_digits_to_fill..]);
            }
            format!(
                "{}{}{}",
                self.prefix.unwrap_or(""),
                hint,
                self.suffix.unwrap_or("")
            )
        }
    }
}

enum WarningType {
    UnreadableLiteral,
    InconsistentDigitGrouping,
    LargeDigitGroups,
    DecimalRepresentation,
}

impl WarningType {
    crate fn display(&self, grouping_hint: &str, cx: &EarlyContext, span: syntax_pos::Span) {
        match self {
            WarningType::UnreadableLiteral => span_lint_and_sugg(
                cx,
                UNREADABLE_LITERAL,
                span,
                "long literal lacking separators",
                "consider",
                grouping_hint.to_owned(),
            ),
            WarningType::LargeDigitGroups => span_lint_and_sugg(
                cx,
                LARGE_DIGIT_GROUPS,
                span,
                "digit groups should be smaller",
                "consider",
                grouping_hint.to_owned(),
            ),
            WarningType::InconsistentDigitGrouping => span_lint_and_sugg(
                cx,
                INCONSISTENT_DIGIT_GROUPING,
                span,
                "digits grouped inconsistently by underscores",
                "consider",
                grouping_hint.to_owned(),
            ),
            WarningType::DecimalRepresentation => span_lint_and_sugg(
                cx,
                DECIMAL_LITERAL_REPRESENTATION,
                span,
                "integer literal has a better hexadecimal representation",
                "consider",
                grouping_hint.to_owned(),
            ),
        };
    }
}

#[derive(Copy, Clone)]
pub struct LiteralDigitGrouping;

impl LintPass for LiteralDigitGrouping {
    fn get_lints(&self) -> LintArray {
        lint_array!(
            UNREADABLE_LITERAL,
            INCONSISTENT_DIGIT_GROUPING,
            LARGE_DIGIT_GROUPS
        )
    }
}

impl EarlyLintPass for LiteralDigitGrouping {
    fn check_expr(&mut self, cx: &EarlyContext, expr: &Expr) {
        if in_external_macro(cx, expr.span) {
            return;
        }

        if let ExprKind::Lit(ref lit) = expr.node {
            self.check_lit(cx, lit)
        }
    }
}

impl LiteralDigitGrouping {
    fn check_lit(self, cx: &EarlyContext, lit: &Lit) {
        match lit.node {
            LitKind::Int(..) => {
                // Lint integral literals.
                if_chain! {
                    if let Some(src) = snippet_opt(cx, lit.span);
                    if let Some(firstch) = src.chars().next();
                    if char::to_digit(firstch, 10).is_some();
                    then {
                        let digit_info = DigitInfo::new(&src, false);
                        let _ = Self::do_lint(digit_info.digits).map_err(|warning_type| {
                            warning_type.display(&digit_info.grouping_hint(), cx, lit.span)
                        });
                    }
                }
            },
            LitKind::Float(..) | LitKind::FloatUnsuffixed(..) => {
                // Lint floating-point literals.
                if_chain! {
                    if let Some(src) = snippet_opt(cx, lit.span);
                    if let Some(firstch) = src.chars().next();
                    if char::to_digit(firstch, 10).is_some();
                    then {
                        let digit_info = DigitInfo::new(&src, true);
                        // Separate digits into integral and fractional parts.
                        let parts: Vec<&str> = digit_info
                            .digits
                            .split_terminator('.')
                            .collect();

                        // Lint integral and fractional parts separately, and then check consistency of digit
                        // groups if both pass.
                        let _ = Self::do_lint(parts[0])
                            .map(|integral_group_size| {
                                if parts.len() > 1 {
                                    // Lint the fractional part of literal just like integral part, but reversed.
                                    let fractional_part = &parts[1].chars().rev().collect::<String>();
                                    let _ = Self::do_lint(fractional_part)
                                        .map(|fractional_group_size| {
                                            let consistent = Self::parts_consistent(integral_group_size,
                                                                                    fractional_group_size,
                                                                                    parts[0].len(),
                                                                                    parts[1].len());
                                            if !consistent {
                                                WarningType::InconsistentDigitGrouping.display(&digit_info.grouping_hint(),
                                                cx,
                                                lit.span);
                                            }
                                        })
                                    .map_err(|warning_type| warning_type.display(&digit_info.grouping_hint(),
                                    cx,
                                    lit.span));
                                }
                            })
                        .map_err(|warning_type| warning_type.display(&digit_info.grouping_hint(), cx, lit.span));
                    }
                }
            },
            _ => (),
        }
    }

    /// Given the sizes of the digit groups of both integral and fractional
    /// parts, and the length
    /// of both parts, determine if the digits have been grouped consistently.
    fn parts_consistent(int_group_size: usize, frac_group_size: usize, int_size: usize, frac_size: usize) -> bool {
        match (int_group_size, frac_group_size) {
            // No groups on either side of decimal point - trivially consistent.
            (0, 0) => true,
            // Integral part has grouped digits, fractional part does not.
            (_, 0) => frac_size <= int_group_size,
            // Fractional part has grouped digits, integral part does not.
            (0, _) => int_size <= frac_group_size,
            // Both parts have grouped digits. Groups should be the same size.
            (_, _) => int_group_size == frac_group_size,
        }
    }

    /// Performs lint on `digits` (no decimal point) and returns the group
    /// size on success or `WarningType` when emitting a warning.
    fn do_lint(digits: &str) -> Result<usize, WarningType> {
        // Grab underscore indices with respect to the units digit.
        let underscore_positions: Vec<usize> = digits
            .chars()
            .rev()
            .enumerate()
            .filter_map(|(idx, digit)| if digit == '_' { Some(idx) } else { None })
            .collect();

        if underscore_positions.is_empty() {
            // Check if literal needs underscores.
            if digits.len() > 5 {
                Err(WarningType::UnreadableLiteral)
            } else {
                Ok(0)
            }
        } else {
            // Check consistency and the sizes of the groups.
            let group_size = underscore_positions[0];
            let consistent = underscore_positions
                .windows(2)
                .all(|ps| ps[1] - ps[0] == group_size + 1)
                // number of digits to the left of the last group cannot be bigger than group size.
                && (digits.len() - underscore_positions.last()
                                                       .expect("there's at least one element") <= group_size + 1);

            if !consistent {
                return Err(WarningType::InconsistentDigitGrouping);
            } else if group_size > 4 {
                return Err(WarningType::LargeDigitGroups);
            }
            Ok(group_size)
        }
    }
}

#[derive(Copy, Clone)]
pub struct LiteralRepresentation {
    threshold: u64,
}

impl LintPass for LiteralRepresentation {
    fn get_lints(&self) -> LintArray {
        lint_array!(DECIMAL_LITERAL_REPRESENTATION)
    }
}

impl EarlyLintPass for LiteralRepresentation {
    fn check_expr(&mut self, cx: &EarlyContext, expr: &Expr) {
        if in_external_macro(cx, expr.span) {
            return;
        }

        if let ExprKind::Lit(ref lit) = expr.node {
            self.check_lit(cx, lit)
        }
    }
}

impl LiteralRepresentation {
    pub fn new(threshold: u64) -> Self {
        Self {
            threshold,
        }
    }
    fn check_lit(self, cx: &EarlyContext, lit: &Lit) {
        // Lint integral literals.
        if_chain! {
            if let LitKind::Int(..) = lit.node;
            if let Some(src) = snippet_opt(cx, lit.span);
            if let Some(firstch) = src.chars().next();
            if char::to_digit(firstch, 10).is_some();
            then {
                let digit_info = DigitInfo::new(&src, false);
                if digit_info.radix == Radix::Decimal {
                    let val = digit_info.digits
                        .chars()
                        .filter(|&c| c != '_')
                        .collect::<String>()
                        .parse::<u128>().unwrap();
                    if val < u128::from(self.threshold) {
                        return
                    }
                    let hex = format!("{:#X}", val);
                    let digit_info = DigitInfo::new(&hex[..], false);
                    let _ = Self::do_lint(digit_info.digits).map_err(|warning_type| {
                        warning_type.display(&digit_info.grouping_hint(), cx, lit.span)
                    });
                }
            }
        }
    }

    fn do_lint(digits: &str) -> Result<(), WarningType> {
        if digits.len() == 1 {
            // Lint for 1 digit literals, if someone really sets the threshold that low
            if digits == "1" || digits == "2" || digits == "4" || digits == "8" || digits == "3" || digits == "7"
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
