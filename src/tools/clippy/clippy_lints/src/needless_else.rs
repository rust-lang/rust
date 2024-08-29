use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::{IntoSpan, SpanRangeExt};
use rustc_ast::ast::{Expr, ExprKind};
use rustc_errors::Applicability;
use rustc_lint::{EarlyContext, EarlyLintPass};
use rustc_session::declare_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for empty `else` branches.
    ///
    /// ### Why is this bad?
    /// An empty else branch does nothing and can be removed.
    ///
    /// ### Example
    /// ```no_run
    ///# fn check() -> bool { true }
    /// if check() {
    ///     println!("Check successful!");
    /// } else {
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
    ///# fn check() -> bool { true }
    /// if check() {
    ///     println!("Check successful!");
    /// }
    /// ```
    #[clippy::version = "1.72.0"]
    pub NEEDLESS_ELSE,
    style,
    "empty else branch"
}
declare_lint_pass!(NeedlessElse => [NEEDLESS_ELSE]);

impl EarlyLintPass for NeedlessElse {
    fn check_expr(&mut self, cx: &EarlyContext<'_>, expr: &Expr) {
        if let ExprKind::If(_, then_block, Some(else_clause)) = &expr.kind
            && let ExprKind::Block(block, _) = &else_clause.kind
            && !expr.span.from_expansion()
            && !else_clause.span.from_expansion()
            && block.stmts.is_empty()
            && let range = (then_block.span.hi()..expr.span.hi()).trim_start(cx)
            && range.clone().check_source_text(cx, |src| {
                // Ignore else blocks that contain comments or #[cfg]s
                !src.contains(['/', '#'])
            })
        {
            span_lint_and_sugg(
                cx,
                NEEDLESS_ELSE,
                range.with_ctxt(expr.span.ctxt()),
                "this `else` branch is empty",
                "you can remove it",
                String::new(),
                Applicability::MachineApplicable,
            );
        }
    }
}
