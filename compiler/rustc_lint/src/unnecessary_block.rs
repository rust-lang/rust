use crate::lints::UnnecessaryBlock as UnnecessaryBlockErr;
use crate::{LateContext, LateLintPass, LintContext};
use rustc_hir::{BlockCheckMode, Expr, ExprKind, Stmt, StmtKind};

declare_lint! {
    /// The `unnecessary_block` lint checks for unnecessary block.
    ///
    /// ### Example
    ///
    /// ```rust
    /// fn main() {
    ///     {}
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Having an unnecessary block is almost always a mistake.
    pub UNNECESSARY_BLOCK,
    Warn,
    "block that unnecessary"
}

declare_lint_pass!(UnnecessaryBlock => [UNNECESSARY_BLOCK]);

impl<'tcx> LateLintPass<'tcx> for UnnecessaryBlock {
    fn check_stmt(&mut self, cx: &LateContext<'tcx>, stmt: &Stmt<'_>) {
        if let StmtKind::Expr(expr) = stmt.kind &&
            let ExprKind::Block(block, _) = expr.kind &&
            block.stmts.is_empty() &&
            block.expr.is_none() &&
            matches!(block.rules, BlockCheckMode::DefaultBlock)
        {
            cx.emit_spanned_lint(
                UNNECESSARY_BLOCK,
                block.span,
                UnnecessaryBlockErr {
                    suggestion: block.span,
                },
            )
        }
    }

    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if let ExprKind::Block(blk, _) = expr.kind &&
            let Some(blk_exp) = blk.expr &&
            let ExprKind::Block(block, _) = blk_exp.kind &&
            block.stmts.is_empty() &&
            block.expr.is_none() &&
            matches!(block.rules, BlockCheckMode::DefaultBlock)
        {
            cx.emit_spanned_lint(
                UNNECESSARY_BLOCK,
                block.span,
                UnnecessaryBlockErr {
                    suggestion: block.span,
                },
            )
        }
    }
}
