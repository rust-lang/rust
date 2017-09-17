//! lint on blocks unnecessarily using >= with a + 1 or - 1

use rustc::lint::*;
use syntax::ast::*;

use utils::span_help_and_lint;

/// **What it does:** Checks for usage of `x >= y + 1` or `x - 1 >= y` (and `<=`) in a block
///
///
/// **Why is this bad?** Readability -- better to use `> y` instead of `>= y + 1`.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// x >= y + 1
/// ```
///
/// Could be written:
///
/// ```rust
/// x > y
/// ```
declare_lint! {
    pub INT_PLUS_ONE,
    Allow,
    "instead of using x >= y + 1, use x > y"
}

pub struct IntPlusOne;

impl LintPass for IntPlusOne {
    fn get_lints(&self) -> LintArray {
        lint_array!(INT_PLUS_ONE)
    }
}

// cases:
// BinOpKind::Ge
// x >= y + 1
// x - 1 >= y
//
// BinOpKind::Le
// x + 1 <= y
// x <= y - 1

impl IntPlusOne {
    #[allow(cast_sign_loss)]
    fn check_lit(&self, lit: &Lit, target_value: i128) -> bool {
        if let LitKind::Int(value, ..) = lit.node {
            return value == (target_value as u128)
        }
        false
    }

    fn check_binop(&self, binop: BinOpKind, lhs: &Expr, rhs: &Expr) -> bool {
        match (binop, &lhs.node, &rhs.node) {
            // case where `x - 1 >= ...` or `-1 + x >= ...`
            (BinOpKind::Ge, &ExprKind::Binary(ref lhskind, ref lhslhs, ref lhsrhs), _) => {
                match (lhskind.node, &lhslhs.node, &lhsrhs.node) {
                    // `-1 + x`
                    (BinOpKind::Add, &ExprKind::Lit(ref lit), _) => self.check_lit(lit, -1),
                    // `x - 1`
                    (BinOpKind::Sub, _, &ExprKind::Lit(ref lit)) => self.check_lit(lit, 1),
                    _ => false
                }
            },
            // case where `... >= y + 1` or `... >= 1 + y`
            (BinOpKind::Ge, _, &ExprKind::Binary(ref rhskind, ref rhslhs, ref rhsrhs)) if rhskind.node == BinOpKind::Add => {
                match (&rhslhs.node, &rhsrhs.node) {
                    // `y + 1` and `1 + y`
                    (&ExprKind::Lit(ref lit), _)|(_, &ExprKind::Lit(ref lit)) => self.check_lit(lit, 1),
                    _ => false
                }
            },
            // case where `x + 1 <= ...` or `1 + x <= ...`
            (BinOpKind::Le, &ExprKind::Binary(ref lhskind, ref lhslhs, ref lhsrhs), _) if lhskind.node == BinOpKind::Add => {
                match (&lhslhs.node, &lhsrhs.node) {
                    // `1 + x` and `x + 1`
                    (&ExprKind::Lit(ref lit), _)|(_, &ExprKind::Lit(ref lit)) => self.check_lit(lit, 1),
                    _ => false
                }
            },
            // case where `... >= y - 1` or `... >= -1 + y`
            (BinOpKind::Le, _, &ExprKind::Binary(ref rhskind, ref rhslhs, ref rhsrhs)) => {
                match (rhskind.node, &rhslhs.node, &rhsrhs.node) {
                    // `-1 + y`
                    (BinOpKind::Add, &ExprKind::Lit(ref lit), _) => self.check_lit(lit, -1),
                    // `y - 1`
                    (BinOpKind::Sub, _, &ExprKind::Lit(ref lit)) => self.check_lit(lit, 1),
                    _ => false
                }
            },
            _ => false
        }
    }

}

impl EarlyLintPass for IntPlusOne {
    fn check_expr(&mut self, cx: &EarlyContext, item: &Expr) {
        if let ExprKind::Binary(ref kind, ref lhs, ref rhs) = item.node {
            if self.check_binop(kind.node, lhs, rhs) {
                span_help_and_lint(
                    cx,
                    INT_PLUS_ONE,
                    item.span,
                    "Unnecessary `>= y + 1` or `x - 1 >=`",
                    "Consider reducing `x >= y + 1` or `x - 1 >= y` to `x > y`",
                );
            }
        }
    }
}
