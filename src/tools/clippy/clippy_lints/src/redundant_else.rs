use crate::utils::span_lint_and_help;
use rustc_ast::ast::{Block, Expr, ExprKind, Stmt, StmtKind};
use rustc_ast::visit::{walk_expr, Visitor};
use rustc_lint::{EarlyContext, EarlyLintPass};
use rustc_middle::lint::in_external_macro;
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// **What it does:** Checks for `else` blocks that can be removed without changing semantics.
    ///
    /// **Why is this bad?** The `else` block adds unnecessary indentation and verbosity.
    ///
    /// **Known problems:** Some may prefer to keep the `else` block for clarity.
    ///
    /// **Example:**
    ///
    /// ```rust
    /// fn my_func(count: u32) {
    ///     if count == 0 {
    ///         print!("Nothing to do");
    ///         return;
    ///     } else {
    ///         print!("Moving on...");
    ///     }
    /// }
    /// ```
    /// Use instead:
    /// ```rust
    /// fn my_func(count: u32) {
    ///     if count == 0 {
    ///         print!("Nothing to do");
    ///         return;
    ///     }
    ///     print!("Moving on...");
    /// }
    /// ```
    pub REDUNDANT_ELSE,
    pedantic,
    "`else` branch that can be removed without changing semantics"
}

declare_lint_pass!(RedundantElse => [REDUNDANT_ELSE]);

impl EarlyLintPass for RedundantElse {
    fn check_stmt(&mut self, cx: &EarlyContext<'_>, stmt: &Stmt) {
        if in_external_macro(cx.sess, stmt.span) {
            return;
        }
        // Only look at expressions that are a whole statement
        let expr: &Expr = match &stmt.kind {
            StmtKind::Expr(expr) | StmtKind::Semi(expr) => expr,
            _ => return,
        };
        // if else
        let (mut then, mut els): (&Block, &Expr) = match &expr.kind {
            ExprKind::If(_, then, Some(els)) => (then, els),
            _ => return,
        };
        loop {
            if !BreakVisitor::default().check_block(then) {
                // then block does not always break
                return;
            }
            match &els.kind {
                // else if else
                ExprKind::If(_, next_then, Some(next_els)) => {
                    then = next_then;
                    els = next_els;
                    continue;
                },
                // else if without else
                ExprKind::If(..) => return,
                // done
                _ => break,
            }
        }
        span_lint_and_help(
            cx,
            REDUNDANT_ELSE,
            els.span,
            "redundant else block",
            None,
            "remove the `else` block and move the contents out",
        );
    }
}

/// Call `check` functions to check if an expression always breaks control flow
#[derive(Default)]
struct BreakVisitor {
    is_break: bool,
}

impl<'ast> Visitor<'ast> for BreakVisitor {
    fn visit_block(&mut self, block: &'ast Block) {
        self.is_break = match block.stmts.as_slice() {
            [.., last] => self.check_stmt(last),
            _ => false,
        };
    }

    fn visit_expr(&mut self, expr: &'ast Expr) {
        self.is_break = match expr.kind {
            ExprKind::Break(..) | ExprKind::Continue(..) | ExprKind::Ret(..) => true,
            ExprKind::Match(_, ref arms) => arms.iter().all(|arm| self.check_expr(&arm.body)),
            ExprKind::If(_, ref then, Some(ref els)) => self.check_block(then) && self.check_expr(els),
            ExprKind::If(_, _, None)
            // ignore loops for simplicity
            | ExprKind::While(..) | ExprKind::ForLoop(..) | ExprKind::Loop(..) => false,
            _ => {
                walk_expr(self, expr);
                return;
            },
        };
    }
}

impl BreakVisitor {
    fn check<T>(&mut self, item: T, visit: fn(&mut Self, T)) -> bool {
        visit(self, item);
        std::mem::replace(&mut self.is_break, false)
    }

    fn check_block(&mut self, block: &Block) -> bool {
        self.check(block, Self::visit_block)
    }

    fn check_expr(&mut self, expr: &Expr) -> bool {
        self.check(expr, Self::visit_expr)
    }

    fn check_stmt(&mut self, stmt: &Stmt) -> bool {
        self.check(stmt, Self::visit_stmt)
    }
}
