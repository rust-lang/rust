use clippy_utils::diagnostics::span_lint_and_sugg;
use rustc_errors::Applicability;
use rustc_hir::{ExprKind, MatchSource, Stmt, StmtKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for the presence of a semicolon at the end of
    /// a `match` or `if` statement evaluating to `()`.
    ///
    /// ### Why is this bad?
    /// The semicolon is not needed, and may be removed to
    /// avoid confusion and visual clutter.
    ///
    /// ### Example
    /// ```no_run
    /// # let a: u32 = 42;
    /// if a > 10 {
    ///     println!("a is greater than 10");
    /// };
    /// ```
    /// Use instead:
    /// ```no_run
    /// # let a: u32 = 42;
    /// if a > 10 {
    ///    println!("a is greater than 10");
    /// }
    /// ```
    #[clippy::version = "1.86.0"]
    pub UNNECESSARY_SEMICOLON,
    pedantic,
    "unnecessary semicolon after expression returning `()`"
}

declare_lint_pass!(UnnecessarySemicolon => [UNNECESSARY_SEMICOLON]);

impl LateLintPass<'_> for UnnecessarySemicolon {
    fn check_stmt(&mut self, cx: &LateContext<'_>, stmt: &Stmt<'_>) {
        // rustfmt already takes care of removing semicolons at the end
        // of loops.
        if let StmtKind::Semi(expr) = stmt.kind
            && !stmt.span.from_expansion()
            && !expr.span.from_expansion()
            && matches!(
                expr.kind,
                ExprKind::If(..) | ExprKind::Match(_, _, MatchSource::Normal | MatchSource::Postfix)
            )
            && cx.typeck_results().expr_ty(expr) == cx.tcx.types.unit
        {
            let semi_span = expr.span.shrink_to_hi().to(stmt.span.shrink_to_hi());
            span_lint_and_sugg(
                cx,
                UNNECESSARY_SEMICOLON,
                semi_span,
                "unnecessary semicolon",
                "remove",
                String::new(),
                Applicability::MachineApplicable,
            );
        }
    }
}
