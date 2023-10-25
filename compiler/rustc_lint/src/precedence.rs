use rustc_ast::token::LitKind;
use rustc_ast::{BinOpKind, Expr, ExprKind, MethodCall, UnOp};
use rustc_span::source_map::Spanned;

use crate::lints::{PrecedenceDiag, PrecedenceUnarySuggestion, PrecedenceUnwarySuggestion};
use crate::{EarlyContext, EarlyLintPass, LintContext};

// List of functions `f(x)` where `f(-x)=-f(x)` so the
// precedence doens't matter.
const ALLOWED_ODD_FUNCTIONS: [&str; 14] = [
    "asin",
    "asinh",
    "atan",
    "atanh",
    "cbrt",
    "fract",
    "round",
    "signum",
    "sin",
    "sinh",
    "tan",
    "tanh",
    "to_degrees",
    "to_radians",
];

declare_lint! {
    /// The `ambiguous_precedence` lint checks for operations where
    /// precedence may be unclear and suggests adding parentheses.
    ///
    /// ### Example
    ///
    /// ```rust
    /// # #![allow(unused)]
    /// 1 << 2 + 3; // equals 32, while `(1 << 2) + 3` equals 7
    /// -1i32.abs(); // equals -1, while `(-1i32).abs()` equals 1
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Unary operations take precedence on binary operations and method
    /// calls take precedence over unary precedence. Setting the precedence
    /// explicitly makes the code clearer and avoid potential bugs.
    pub AMBIGUOUS_PRECEDENCE,
    Warn,
    "operations where precedence may be unclear"
}

declare_lint_pass!(Precedence => [AMBIGUOUS_PRECEDENCE]);

impl EarlyLintPass for Precedence {
    fn check_expr(&mut self, cx: &EarlyContext<'_>, expr: &Expr) {
        if expr.span.from_expansion() {
            return;
        }

        if let ExprKind::Binary(Spanned { node: op, .. }, ref left, ref right) = expr.kind {
            if !is_bit_op(op) {
                return;
            }

            let suggestion = match (is_arith_expr(left), is_arith_expr(right)) {
                (true, true) => PrecedenceUnwarySuggestion::TwoExpr {
                    start_span: left.span.shrink_to_lo(),
                    end_span: left.span.shrink_to_hi(),
                    start2_span: right.span.shrink_to_lo(),
                    end2_span: right.span.shrink_to_hi(),
                },
                (true, false) => PrecedenceUnwarySuggestion::OneExpr {
                    start_span: left.span.shrink_to_lo(),
                    end_span: left.span.shrink_to_hi(),
                },
                (false, true) => PrecedenceUnwarySuggestion::OneExpr {
                    start_span: right.span.shrink_to_lo(),
                    end_span: right.span.shrink_to_hi(),
                },
                (false, false) => return,
            };

            cx.emit_spanned_lint(
                AMBIGUOUS_PRECEDENCE,
                expr.span,
                PrecedenceDiag::Unwary { suggestion },
            );
        }

        if let ExprKind::Unary(UnOp::Neg, operand) = &expr.kind {
            let mut arg = operand;

            let mut all_odd = true;
            while let ExprKind::MethodCall(box MethodCall { seg, receiver, .. }) = &arg.kind {
                let seg_str = seg.ident.name.as_str();
                all_odd &=
                    ALLOWED_ODD_FUNCTIONS.iter().any(|odd_function| **odd_function == *seg_str);
                arg = receiver;
            }

            if !all_odd
                && let ExprKind::Lit(lit) = &arg.kind
                && let LitKind::Integer | LitKind::Float = &lit.kind
                && !arg.span.from_expansion()
            {
                cx.emit_spanned_lint(
                    AMBIGUOUS_PRECEDENCE,
                    expr.span,
                    PrecedenceDiag::Unary {
                        suggestion: PrecedenceUnarySuggestion {
                            start_span: operand.span.shrink_to_lo(),
                            end_span: operand.span.shrink_to_hi(),
                        },
                    },
                );
            }
        }
    }
}

fn is_arith_expr(expr: &Expr) -> bool {
    match expr.kind {
        ExprKind::Binary(Spanned { node: op, .. }, _, _) => is_arith_op(op),
        _ => false,
    }
}

#[must_use]
fn is_bit_op(op: BinOpKind) -> bool {
    use rustc_ast::ast::BinOpKind::{BitAnd, BitOr, BitXor, Shl, Shr};
    matches!(op, BitXor | BitAnd | BitOr | Shl | Shr)
}

#[must_use]
fn is_arith_op(op: BinOpKind) -> bool {
    use rustc_ast::ast::BinOpKind::{Add, Div, Mul, Rem, Sub};
    matches!(op, Add | Sub | Mul | Div | Rem)
}
