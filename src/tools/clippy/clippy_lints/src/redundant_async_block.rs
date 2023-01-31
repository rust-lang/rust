use clippy_utils::{diagnostics::span_lint_and_sugg, source::snippet};
use rustc_ast::ast::*;
use rustc_ast::visit::Visitor as AstVisitor;
use rustc_errors::Applicability;
use rustc_lint::{EarlyContext, EarlyLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `async` block that only returns `await` on a future.
    ///
    /// ### Why is this bad?
    /// It is simpler and more efficient to use the future directly.
    ///
    /// ### Example
    /// ```rust
    /// async fn f() -> i32 {
    ///     1 + 2
    /// }
    ///
    /// let fut = async {
    ///     f().await
    /// };
    /// ```
    /// Use instead:
    /// ```rust
    /// async fn f() -> i32 {
    ///     1 + 2
    /// }
    ///
    /// let fut = f();
    /// ```
    #[clippy::version = "1.69.0"]
    pub REDUNDANT_ASYNC_BLOCK,
    complexity,
    "`async { future.await }` can be replaced by `future`"
}
declare_lint_pass!(RedundantAsyncBlock => [REDUNDANT_ASYNC_BLOCK]);

impl EarlyLintPass for RedundantAsyncBlock {
    fn check_expr(&mut self, cx: &EarlyContext<'_>, expr: &Expr) {
        if expr.span.from_expansion() {
            return;
        }
        if let ExprKind::Async(_, block) = &expr.kind && block.stmts.len() == 1 &&
            let Some(Stmt { kind: StmtKind::Expr(last), .. }) = block.stmts.last() &&
            let ExprKind::Await(future) = &last.kind &&
            !future.span.from_expansion() &&
            !await_in_expr(future)
        {
            span_lint_and_sugg(
                cx,
                REDUNDANT_ASYNC_BLOCK,
                expr.span,
                "this async expression only awaits a single future",
                "you can reduce it to",
                snippet(cx, future.span, "..").into_owned(),
                Applicability::MachineApplicable,
            );
        }
    }
}

/// Check whether an expression contains `.await`
fn await_in_expr(expr: &Expr) -> bool {
    let mut detector = AwaitDetector::default();
    detector.visit_expr(expr);
    detector.await_found
}

#[derive(Default)]
struct AwaitDetector {
    await_found: bool,
}

impl<'ast> AstVisitor<'ast> for AwaitDetector {
    fn visit_expr(&mut self, ex: &'ast Expr) {
        match (&ex.kind, self.await_found) {
            (ExprKind::Await(_), _) => self.await_found = true,
            (_, false) => rustc_ast::visit::walk_expr(self, ex),
            _ => (),
        }
    }
}
