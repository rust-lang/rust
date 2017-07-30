//! Lints concerned with the grouping of digits with underscores in integral or
//! floating-point literal expressions.

use rustc::lint::*;
use syntax::ast::*;
use syntax_pos;
use utils::{span_help_and_lint, snippet_opt, in_external_macro};

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
declare_lint! {
    pub UNREADABLE_LITERAL,
    Warn,
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
declare_lint! {
    pub INCONSISTENT_DIGIT_GROUPING,
    Warn,
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
declare_lint! {
    pub LARGE_DIGIT_GROUPS,
    Warn,
    "grouping digits into groups that are too large"
}

#[derive(Debug)]
enum Radix {
    Binary,
    Octal,
    Decimal,
    Hexadecimal,
}

impl Radix {
    /// Return a reasonable digit group size for this radix.
    pub fn suggest_grouping(&self) -> usize {
        match *self {
            Radix::Binary | Radix::Hexadecimal => 4,
            Radix::Octal | Radix::Decimal => 3,
        }
    }
}

#[derive(Debug)]
struct DigitInfo<'a> {
    /// Characters of a literal between the radix prefix and type suffix.
    pub digits: &'a str,
    /// Which radix the literal was represented in.
    pub radix: Radix,
    /// The radix prefix, if present.
    pub prefix: Option<&'a str>,
    /// The type suffix, including preceding underscore if present.
    pub suffix: Option<&'a str>,
    /// True for floating-point literals.
    pub float: bool,
}

impl<'a> DigitInfo<'a> {
    pub fn new(lit: &str, float: bool) -> DigitInfo {
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
            if !float && (d == 'i' || d == 'u') || float && d == 'f' {
                let suffix_start = if last_d == '_' { d_idx - 1 } else { d_idx };
                let (digits, suffix) = sans_prefix.split_at(suffix_start);
                return DigitInfo {
                    digits: digits,
                    radix: radix,
                    prefix: prefix,
                    suffix: Some(suffix),
                    float: float,
                };
            }
            last_d = d
        }

        // No suffix found
        DigitInfo {
            digits: sans_prefix,
            radix: radix,
            prefix: prefix,
            suffix: None,
            float: float,
        }
    }

    /// Returns digits grouped in a sensible way.
    fn grouping_hint(&self) -> String {
        let group_size = self.radix.suggest_grouping();
        if self.digits.contains('.') {
            let mut parts = self.digits.split(".");
            let int_part_hint = parts
                .next()
                .unwrap()
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
                .unwrap()
                .chars()
                .filter(|&c| c != '_')
                .collect::<Vec<_>>()
                .chunks(group_size)
                .map(|chunk| chunk.into_iter().collect())
                .collect::<Vec<String>>()
                .join("_");
            format!("{}.{}{}", int_part_hint, frac_part_hint, self.suffix.unwrap_or(""))
        } else {
            let hint = self.digits
                .chars()
                .rev()
                .filter(|&c| c != '_')
                .collect::<Vec<_>>()
                .chunks(group_size)
                .map(|chunk| chunk.into_iter().rev().collect())
                .rev()
                .collect::<Vec<String>>()
                .join("_");
            format!("{}{}{}", self.prefix.unwrap_or(""), hint, self.suffix.unwrap_or(""))
        }
    }
}

enum WarningType {
    UnreadableLiteral,
    InconsistentDigitGrouping,
    LargeDigitGroups,
}


impl WarningType {
    pub fn display(&self, grouping_hint: &str, cx: &EarlyContext, span: &syntax_pos::Span) {
        match *self {
            WarningType::UnreadableLiteral => {
                span_help_and_lint(
                    cx,
                    UNREADABLE_LITERAL,
                    *span,
                    "long literal lacking separators",
                    &format!("consider: {}", grouping_hint),
                )
            },
            WarningType::LargeDigitGroups => {
                span_help_and_lint(
                    cx,
                    LARGE_DIGIT_GROUPS,
                    *span,
                    "digit groups should be smaller",
                    &format!("consider: {}", grouping_hint),
                )
            },
            WarningType::InconsistentDigitGrouping => {
                span_help_and_lint(
                    cx,
                    INCONSISTENT_DIGIT_GROUPING,
                    *span,
                    "digits grouped inconsistently by underscores",
                    &format!("consider: {}", grouping_hint),
                )
            },
        };
    }
}

#[derive(Copy, Clone)]
pub struct LiteralDigitGrouping;

impl LintPass for LiteralDigitGrouping {
    fn get_lints(&self) -> LintArray {
        lint_array!(UNREADABLE_LITERAL, INCONSISTENT_DIGIT_GROUPING, LARGE_DIGIT_GROUPS)
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
    fn check_lit(&self, cx: &EarlyContext, lit: &Lit) {
        // Lint integral literals.
        if_let_chain! {[
            let LitKind::Int(..) = lit.node,
            let Some(src) = snippet_opt(cx, lit.span),
            let Some(firstch) = src.chars().next(),
            char::to_digit(firstch, 10).is_some()
        ], {
            let digit_info = DigitInfo::new(&src, false);
            let _ = LiteralDigitGrouping::do_lint(digit_info.digits).map_err(|warning_type| {
                warning_type.display(&digit_info.grouping_hint(), cx, &lit.span)
            });
        }}

        // Lint floating-point literals.
        if_let_chain! {[
            let LitKind::Float(..) = lit.node,
            let Some(src) = snippet_opt(cx, lit.span),
            let Some(firstch) = src.chars().next(),
            char::to_digit(firstch, 10).is_some()
        ], {
            let digit_info = DigitInfo::new(&src, true);
            // Separate digits into integral and fractional parts.
            let parts: Vec<&str> = digit_info
                .digits
                .split_terminator('.')
                .collect();

            // Lint integral and fractional parts separately, and then check consistency of digit
            // groups if both pass.
            let _ = LiteralDigitGrouping::do_lint(parts[0])
                .map(|integral_group_size| {
                    if parts.len() > 1 {
                        // Lint the fractional part of literal just like integral part, but reversed.
                        let fractional_part = &parts[1].chars().rev().collect::<String>();
                        let _ = LiteralDigitGrouping::do_lint(fractional_part)
                            .map(|fractional_group_size| {
                                let consistent = LiteralDigitGrouping::parts_consistent(integral_group_size, fractional_group_size, parts[0].len(), parts[1].len());
                                if !consistent {
                                    WarningType::InconsistentDigitGrouping.display(&digit_info.grouping_hint(), cx, &lit.span);
                                }
                            })
                            .map_err(|warning_type| warning_type.display(&digit_info.grouping_hint(), cx, &lit.span));
                    }
                })
                .map_err(|warning_type| warning_type.display(&digit_info.grouping_hint(), cx, &lit.span));
        }}
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
            if digits.len() > 4 {
                return Err(WarningType::UnreadableLiteral);
            } else {
                return Ok(0);
            }
        } else {
            // Check consistency and the sizes of the groups.
            let group_size = underscore_positions[0];
            let consistent = underscore_positions
                .windows(2)
                .all(|ps| ps[1] - ps[0] == group_size + 1)
                // number of digits to the left of the last group cannot be bigger than group size.
                && (digits.len() - underscore_positions.last().unwrap() <= group_size + 1);

            if !consistent {
                return Err(WarningType::InconsistentDigitGrouping);
            } else if group_size > 4 {
                return Err(WarningType::LargeDigitGroups);
            }
            return Ok(group_size);
        }
    }
}
