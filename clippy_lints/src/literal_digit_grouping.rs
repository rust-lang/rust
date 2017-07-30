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
            let digits = LiteralDigitGrouping::get_digits(&src, false);

            LiteralDigitGrouping::do_lint(digits, cx, &lit.span);
        }}

        // Lint floating-point literals.
        if_let_chain! {[
            let LitKind::Float(..) = lit.node,
            let Some(src) = snippet_opt(cx, lit.span),
            let Some(firstch) = src.chars().next(),
            char::to_digit(firstch, 10).is_some()
        ], {
            let digits: Vec<&str> = LiteralDigitGrouping::get_digits(&src, true)
                .split_terminator('.')
                .collect();

            // Lint integral and fractional parts separately, and then check consistency of digit
            // groups if both pass.
            if let Some(integral_group_size) = LiteralDigitGrouping::do_lint(digits[0], cx, &lit.span) {
                if digits.len() > 1 {
                    // Lint the fractional part of literal just like integral part, but reversed.
                    let fractional_part = &digits[1].chars().rev().collect::<String>();
                    if let Some(fractional_group_size) = LiteralDigitGrouping::do_lint(fractional_part, cx, &lit.span) {
                        let consistent = match (integral_group_size, fractional_group_size) {
                            // No groups on either side of decimal point - good to go.
                            (0, 0) => true,
                            // Integral part has grouped digits, fractional part does not.
                            (_, 0) => digits[1].len() <= integral_group_size,
                            // Fractional part has grouped digits, integral part does not.
                            (0, _) => digits[0].len() <= fractional_group_size,
                            // Both parts have grouped digits. Groups should be the same size.
                            (_, _) => integral_group_size == fractional_group_size,
                        };

                        if !consistent {
                            span_help_and_lint(cx, INCONSISTENT_DIGIT_GROUPING, lit.span,
                                           "digits grouped inconsistently by underscores",
                                           "consider making each group three or four digits");
                        }
                    }
                }
            }
        }}
    }

    /// Returns the digits of an integral or floating-point literal.
    fn get_digits(lit: &str, float: bool) -> &str {
        // Determine delimiter for radix prefix, if present.
        let mb_r = if lit.starts_with("0x") {
            Some('x')
        } else if lit.starts_with("0b") {
            Some('b')
        } else if lit.starts_with("0o") {
            Some('o')
        } else {
            None
        };

        let has_suffix = !float && (lit.contains('i') || lit.contains('u')) || float && lit.contains('f');

        // Grab part of literal between the radix prefix and type suffix.
        let mut digits = if let Some(r) = mb_r {
            lit.split(|c| c == 'i' || c == 'u' || c == r || (float && c == 'f')).nth(1).unwrap()
        } else {
            lit.split(|c| c == 'i' || c == 'u' || (float && c == 'f')).next().unwrap()
        };

        // If there was an underscore before type suffix, drop it.
        if has_suffix && digits.chars().last().unwrap() == '_' {
            digits = digits.split_at(digits.len() - 1).0;
        }

        digits
    }

    /// Performs lint on `digits` (no decimal point) and returns the group size. `None` is
    /// returned when emitting a warning.
    fn do_lint(digits: &str, cx: &EarlyContext, span: &syntax_pos::Span) -> Option<usize> {
        // Grab underscore indices with respect to the units digit.
        let underscore_positions: Vec<usize> = digits.chars().rev().enumerate()
            .filter_map(|(idx, digit)|
                if digit == '_' {
                    Some(idx)
                } else {
                    None
                })
            .collect();

        if underscore_positions.is_empty() {
            // Check if literal needs underscores.
            if digits.len() > 4 {
                span_help_and_lint(cx, UNREADABLE_LITERAL, *span,
                                   "long literal lacking separators",
                                   "consider using underscores to make literal more readable");
                return None;
            } else {
                return Some(0);
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
                span_help_and_lint(cx, INCONSISTENT_DIGIT_GROUPING, *span,
                                   "digits grouped inconsistently by underscores",
                                   "consider making each group three or four digits");
                return None;
            } else if group_size > 4 {
                span_help_and_lint(cx, LARGE_DIGIT_GROUPS, *span,
                                   "digit groups should be smaller",
                                   "consider using groups of three or four digits");
                return None;
            }
            return Some(group_size);
        }
    }
}
