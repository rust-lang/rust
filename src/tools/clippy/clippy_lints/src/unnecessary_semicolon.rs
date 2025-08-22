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
    ///     println!("a is greater than 10");
    /// }
    /// ```
    #[clippy::version = "1.86.0"]
    pub UNNECESSARY_SEMICOLON,
    pedantic,
    "unnecessary semicolon after expression returning `()`"
}

#[derive(Default)]
pub struct UnnecessarySemicolon {
    last_statements: Vec<(HirId, bool)>,
}

impl_lint_pass!(UnnecessarySemicolon => [UNNECESSARY_SEMICOLON]);

impl UnnecessarySemicolon {
    /// Enter or leave a block, remembering the last statement of the block.
    fn handle_block(&mut self, cx: &LateContext<'_>, block: &Block<'_>, enter: bool) {
        // The last statement of an expressionless block deserves a special treatment.
        if block.expr.is_none()
            && let Some(last_stmt) = block.stmts.last()
        {
            if enter {
                let block_ty = cx.typeck_results().node_type(block.hir_id);
                self.last_statements.push((last_stmt.hir_id, block_ty.is_unit()));
            } else {
                self.last_statements.pop();
            }
        }
    }

    /// Checks if `stmt` is the last statement in an expressionless block. In this case,
    /// return `Some` with a boolean which is `true` if the block type is `()`.
    fn is_last_in_block(&self, stmt: &Stmt<'_>) -> Option<bool> {
        self.last_statements
            .last()
            .and_then(|&(stmt_id, is_unit)| (stmt_id == stmt.hir_id).then_some(is_unit))
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
            && cx.typeck_results().expr_ty(expr).is_unit()
            // if a stmt has attrs, then turning it into an expr will break the code, since attrs aren't allowed on exprs
            && cx.tcx.hir_attrs(stmt.hir_id).is_empty()
        {
            if let Some(block_is_unit) = self.is_last_in_block(stmt) {
                if cx.tcx.sess.edition() <= Edition2021 && leaks_droppable_temporary_with_limited_lifetime(cx, expr) {
                    // The expression contains temporaries with limited lifetimes in edition lower than 2024. Those may
                    // survive until after the end of the current scope instead of until the end of the statement, so do
                    // not lint this situation.
                    return;
                }

                if !block_is_unit {
                    // Although the expression returns `()`, the block doesn't. This may happen if the expression
                    // returns early in all code paths, such as a `return value` in the condition of an `if` statement,
                    // in which case the block type would be `!`. Do not lint in this case, as the statement would
                    // become the block expression; the block type would become `()` and this may not type correctly
                    // if the expected type for the block is not `()`.
                    return;
                }
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
