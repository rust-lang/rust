use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::macros::span_is_local;
use rustc_hir::{Expr, ExprKind, MatchSource};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for expressions that use the `?` operator and rejects them.
    ///
    /// ### Why restrict this?
    /// Sometimes code wants to avoid the `?` operator because for instance a local
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
    #[clippy::version = "1.69.0"]
    pub QUESTION_MARK_USED,
    restriction,
    "checks if the `?` operator is used"
}

declare_lint_pass!(QuestionMarkUsed => [QUESTION_MARK_USED]);

impl<'tcx> LateLintPass<'tcx> for QuestionMarkUsed {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if let ExprKind::Match(_, _, MatchSource::TryDesugar(_)) = expr.kind {
            if !span_is_local(expr.span) {
                return;
            }

            #[expect(clippy::collapsible_span_lint_calls, reason = "rust-clippy#7797")]
            span_lint_and_then(cx, QUESTION_MARK_USED, expr.span, "the `?` operator was used", |diag| {
                diag.help("consider using a custom macro or match expression");
            });
        }
    }
}
