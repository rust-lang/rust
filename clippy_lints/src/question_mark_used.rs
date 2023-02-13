use clippy_utils::diagnostics::span_lint_and_help;

use rustc_hir::{Expr, ExprKind, MatchSource};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for expressions that use the question mark operator and rejects them.
    ///
    /// ### Why is this bad?
    /// Sometimes code wants to avoid the question mark operator because for instance a local
    /// block requires a macro to re-throw errors to attach additional information to the
    /// error.
    ///
    /// ### Example
    /// ```ignore
    /// let result = expr?;
    /// ```
    ///
    /// Could be written:
    ///
    /// ```ignore
    /// utility_macro!(expr);
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub QUESTION_MARK_USED,
    restriction,
    "complains if the question mark operator is used"
}

declare_lint_pass!(QuestionMarkUsed => [QUESTION_MARK_USED]);

impl<'tcx> LateLintPass<'tcx> for QuestionMarkUsed {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if let ExprKind::Match(_, _, MatchSource::TryDesugar) = expr.kind {
            span_lint_and_help(
                cx,
                QUESTION_MARK_USED,
                expr.span,
                "question mark operator was used",
                None,
                "consider using a custom macro or match expression",
            );
        }
    }
}
