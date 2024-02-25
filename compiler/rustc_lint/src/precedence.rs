use rustc_ast::token::LitKind;
use rustc_ast::{Expr, ExprKind, MethodCall, UnOp};

use crate::lints::{UnaryPrecedenceDiag, UnaryPrecedenceSuggestion};
use crate::{EarlyContext, EarlyLintPass, LintContext};

declare_lint! {
    /// The `ambiguous_unary_precedence` lint checks for use the negative
    /// unary operator with a literal and method calls.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// # #![allow(unused)]
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
    pub AMBIGUOUS_UNARY_PRECEDENCE,
    Deny,
    "operations where precedence may be unclear",
    report_in_external_macro
}

declare_lint_pass!(Precedence => [AMBIGUOUS_UNARY_PRECEDENCE]);

impl EarlyLintPass for Precedence {
    fn check_expr(&mut self, cx: &EarlyContext<'_>, expr: &Expr) {
        let ExprKind::Unary(UnOp::Neg, operand) = &expr.kind else {
            return;
        };

        let mut arg = operand;
        let mut at_least_one = false;
        while let ExprKind::MethodCall(box MethodCall { receiver, .. }) = &arg.kind {
            at_least_one = true;
            arg = receiver;
        }

        if at_least_one
            && let ExprKind::Lit(lit) = &arg.kind
            && let LitKind::Integer | LitKind::Float = &lit.kind
        {
            cx.emit_span_lint(
                AMBIGUOUS_UNARY_PRECEDENCE,
                expr.span,
                UnaryPrecedenceDiag {
                    suggestion: UnaryPrecedenceSuggestion {
                        start_span: operand.span.shrink_to_lo(),
                        end_span: operand.span.shrink_to_hi(),
                    },
                },
            );
        }
    }
}
