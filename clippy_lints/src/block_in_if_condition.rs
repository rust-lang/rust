use crate::utils::*;
use matches::matches;
use rustc::hir::intravisit::{walk_expr, NestedVisitorMap, Visitor};
use rustc::hir::*;
use rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use rustc::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// **What it does:** Checks for `if` conditions that use blocks to contain an
    /// expression.
    ///
    /// **Why is this bad?** It isn't really Rust style, same as using parentheses
    /// to contain expressions.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// if { true } { /* ... */ }
    /// ```
    pub BLOCK_IN_IF_CONDITION_EXPR,
    style,
    "braces that can be eliminated in conditions, e.g., `if { true } ...`"
}

declare_clippy_lint! {
    /// **What it does:** Checks for `if` conditions that use blocks containing
    /// statements, or conditions that use closures with blocks.
    ///
    /// **Why is this bad?** Using blocks in the condition makes it hard to read.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```ignore
    /// if { let x = somefunc(); x } {}
    /// // or
    /// if somefunc(|x| { x == 47 }) {}
    /// ```
    pub BLOCK_IN_IF_CONDITION_STMT,
    style,
    "complex blocks in conditions, e.g., `if { let x = true; x } ...`"
}

declare_lint_pass!(BlockInIfCondition => [BLOCK_IN_IF_CONDITION_EXPR, BLOCK_IN_IF_CONDITION_STMT]);

struct ExVisitor<'a, 'tcx> {
    found_block: Option<&'tcx Expr>,
    cx: &'a LateContext<'a, 'tcx>,
}

impl<'a, 'tcx> Visitor<'tcx> for ExVisitor<'a, 'tcx> {
    fn visit_expr(&mut self, expr: &'tcx Expr) {
        if let ExprKind::Closure(_, _, eid, _, _) = expr.node {
            let body = self.cx.tcx.hir().body(eid);
            let ex = &body.value;
            if matches!(ex.node, ExprKind::Block(_, _)) && !in_macro_or_desugar(body.value.span) {
                self.found_block = Some(ex);
                return;
            }
        }
        walk_expr(self, expr);
    }
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::None
    }
}

const BRACED_EXPR_MESSAGE: &str = "omit braces around single expression condition";
const COMPLEX_BLOCK_MESSAGE: &str = "in an 'if' condition, avoid complex blocks or closures with blocks; \
                                     instead, move the block or closure higher and bind it with a 'let'";

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for BlockInIfCondition {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr) {
        if let Some((check, then, _)) = higher::if_block(&expr) {
            if let ExprKind::Block(block, _) = &check.node {
                if block.rules == DefaultBlock {
                    if block.stmts.is_empty() {
                        if let Some(ex) = &block.expr {
                            // don't dig into the expression here, just suggest that they remove
                            // the block
                            if in_macro_or_desugar(expr.span) || differing_macro_contexts(expr.span, ex.span) {
                                return;
                            }
                            span_help_and_lint(
                                cx,
                                BLOCK_IN_IF_CONDITION_EXPR,
                                check.span,
                                BRACED_EXPR_MESSAGE,
                                &format!(
                                    "try\nif {} {} ... ",
                                    snippet_block(cx, ex.span, ".."),
                                    snippet_block(cx, then.span, "..")
                                ),
                            );
                        }
                    } else {
                        let span = block.expr.as_ref().map_or_else(|| block.stmts[0].span, |e| e.span);
                        if in_macro_or_desugar(span) || differing_macro_contexts(expr.span, span) {
                            return;
                        }
                        // move block higher
                        span_help_and_lint(
                            cx,
                            BLOCK_IN_IF_CONDITION_STMT,
                            check.span,
                            COMPLEX_BLOCK_MESSAGE,
                            &format!(
                                "try\nlet res = {};\nif res {} ... ",
                                snippet_block(cx, block.span, ".."),
                                snippet_block(cx, then.span, "..")
                            ),
                        );
                    }
                }
            } else {
                let mut visitor = ExVisitor { found_block: None, cx };
                walk_expr(&mut visitor, check);
                if let Some(block) = visitor.found_block {
                    span_lint(cx, BLOCK_IN_IF_CONDITION_STMT, block.span, COMPLEX_BLOCK_MESSAGE);
                }
            }
        }
    }
}
