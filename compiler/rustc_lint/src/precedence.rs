use rustc_ast::token::LitKind;
use rustc_ast::{Expr, ExprKind, MethodCall, UnOp};
use rustc_session::{declare_lint, declare_lint_pass};

use crate::lints::{
    AmbiguousNegativeLiteralsCurrentBehaviorSuggestion, AmbiguousNegativeLiteralsDiag,
    AmbiguousNegativeLiteralsNegativeLiteralSuggestion,
};
use crate::{EarlyContext, EarlyLintPass, LintContext};

declare_lint! {
    /// The `ambiguous_negative_literals` lint checks for cases that are
    /// confusing between a negative literal and a negation that's not part
    /// of the literal.
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
    /// Method calls take precedence over unary precedence. Setting the
    /// precedence explicitly makes the code clearer and avoid potential bugs.
    pub AMBIGUOUS_NEGATIVE_LITERALS,
    Deny,
    "ambiguous negative literals operations",
    report_in_external_macro
}

declare_lint_pass!(Precedence => [AMBIGUOUS_NEGATIVE_LITERALS]);

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
                AMBIGUOUS_NEGATIVE_LITERALS,
                expr.span,
                AmbiguousNegativeLiteralsDiag {
                    negative_literal: AmbiguousNegativeLiteralsNegativeLiteralSuggestion {
                        start_span: expr.span.shrink_to_lo(),
                        end_span: arg.span.shrink_to_hi(),
                    },
                    current_behavior: AmbiguousNegativeLiteralsCurrentBehaviorSuggestion {
                        start_span: operand.span.shrink_to_lo(),
                        end_span: operand.span.shrink_to_hi(),
                    },
                },
            );
        }
    }
}
