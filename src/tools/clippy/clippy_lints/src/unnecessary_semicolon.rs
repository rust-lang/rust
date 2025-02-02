use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::leaks_droppable_temporary_with_limited_lifetime;
use rustc_errors::Applicability;
use rustc_hir::{Block, ExprKind, HirId, MatchSource, Stmt, StmtKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::impl_lint_pass;
use rustc_span::edition::Edition::Edition2021;

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

#[derive(Default)]
pub struct UnnecessarySemicolon {
    last_statements: Vec<HirId>,
}

impl_lint_pass!(UnnecessarySemicolon => [UNNECESSARY_SEMICOLON]);

impl UnnecessarySemicolon {
    /// Enter or leave a block, remembering the last statement of the block.
    fn handle_block(&mut self, cx: &LateContext<'_>, block: &Block<'_>, enter: bool) {
        // Up to edition 2021, removing the semicolon of the last statement of a block
        // may result in the scrutinee temporary values to live longer than the block
        // variables. To avoid this problem, we do not lint the last statement of an
        // expressionless block.
        if cx.tcx.sess.edition() <= Edition2021
            && block.expr.is_none()
            && let Some(last_stmt) = block.stmts.last()
        {
            if enter {
                self.last_statements.push(last_stmt.hir_id);
            } else {
                self.last_statements.pop();
            }
        }
    }

    /// Checks if `stmt` is the last statement in an expressionless block for edition ≤ 2021.
    fn is_last_in_block(&self, stmt: &Stmt<'_>) -> bool {
        self.last_statements
            .last()
            .is_some_and(|last_stmt_id| last_stmt_id == &stmt.hir_id)
    }
}

impl<'tcx> LateLintPass<'tcx> for UnnecessarySemicolon {
    fn check_block(&mut self, cx: &LateContext<'_>, block: &Block<'_>) {
        self.handle_block(cx, block, true);
    }

    fn check_block_post(&mut self, cx: &LateContext<'_>, block: &Block<'_>) {
        self.handle_block(cx, block, false);
    }

    fn check_stmt(&mut self, cx: &LateContext<'tcx>, stmt: &Stmt<'tcx>) {
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
            if self.is_last_in_block(stmt) && leaks_droppable_temporary_with_limited_lifetime(cx, expr) {
                return;
            }

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
