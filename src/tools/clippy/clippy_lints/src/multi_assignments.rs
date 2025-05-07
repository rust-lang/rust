use clippy_utils::diagnostics::span_lint;
use rustc_ast::ast::{Expr, ExprKind, Stmt, StmtKind};
use rustc_lint::{EarlyContext, EarlyLintPass};
use rustc_session::declare_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for nested assignments.
    ///
    /// ### Why is this bad?
    /// While this is in most cases already a type mismatch,
    /// the result of an assignment being `()` can throw off people coming from languages like python or C,
    /// where such assignments return a copy of the assigned value.
    ///
    /// ### Example
    /// ```no_run
    ///# let (a, b);
    /// a = b = 42;
    /// ```
    /// Use instead:
    /// ```no_run
    ///# let (a, b);
    /// b = 42;
    /// a = b;
    /// ```
    #[clippy::version = "1.65.0"]
    pub MULTI_ASSIGNMENTS,
    suspicious,
    "instead of using `a = b = c;` use `a = c; b = c;`"
}

declare_lint_pass!(MultiAssignments => [MULTI_ASSIGNMENTS]);

fn strip_paren_blocks(expr: &Expr) -> &Expr {
    match &expr.kind {
        ExprKind::Paren(e) => strip_paren_blocks(e),
        ExprKind::Block(b, _) => {
            if let [
                Stmt {
                    kind: StmtKind::Expr(e),
                    ..
                },
            ] = &b.stmts[..]
            {
                strip_paren_blocks(e)
            } else {
                expr
            }
        },
        _ => expr,
    }
}

impl EarlyLintPass for MultiAssignments {
    fn check_expr(&mut self, cx: &EarlyContext<'_>, expr: &Expr) {
        if let ExprKind::Assign(target, source, _) = &expr.kind {
            if let ExprKind::Assign(_target, _source, _) = &strip_paren_blocks(target).kind {
                span_lint(cx, MULTI_ASSIGNMENTS, expr.span, "assignments don't nest intuitively");
            }
            if let ExprKind::Assign(_target, _source, _) = &strip_paren_blocks(source).kind {
                span_lint(cx, MULTI_ASSIGNMENTS, expr.span, "assignments don't nest intuitively");
            }
        }
    }
}
