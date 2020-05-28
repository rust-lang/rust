use rustc_ast::ast::{Expr, ExprKind};
use rustc_lint::{EarlyContext, EarlyLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_session::{declare_lint_pass, declare_tool_lint};

use crate::utils::span_lint_and_help;

declare_clippy_lint! {
    /// **What it does:** Checks for usage of `as` conversions.
    ///
    /// **Why is this bad?** `as` conversions will perform many kinds of
    /// conversions, including silently lossy conversions and dangerous coercions.
    /// There are cases when it makes sense to use `as`, so the lint is
    /// Allow by default.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust,ignore
    /// let a: u32;
    /// ...
    /// f(a as u16);
    /// ```
    ///
    /// Usually better represents the semantics you expect:
    /// ```rust,ignore
    /// f(a.try_into()?);
    /// ```
    /// or
    /// ```rust,ignore
    /// f(a.try_into().expect("Unexpected u16 overflow in f"));
    /// ```
    ///
    pub AS_CONVERSIONS,
    restriction,
    "using a potentially dangerous silent `as` conversion"
}

declare_lint_pass!(AsConversions => [AS_CONVERSIONS]);

impl EarlyLintPass for AsConversions {
    fn check_expr(&mut self, cx: &EarlyContext<'_>, expr: &Expr) {
        if in_external_macro(cx.sess(), expr.span) {
            return;
        }

        if let ExprKind::Cast(_, _) = expr.kind {
            span_lint_and_help(
                cx,
                AS_CONVERSIONS,
                expr.span,
                "using a potentially dangerous silent `as` conversion",
                None,
                "consider using a safe wrapper for this conversion",
            );
        }
    }
}
