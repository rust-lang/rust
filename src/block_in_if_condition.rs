use rustc_front::hir::*;
use rustc::lint::{LateLintPass, LateContext, LintArray, LintPass};
use rustc_front::intravisit::{Visitor, walk_expr};
use utils::*;

declare_lint! {
    pub BLOCK_IN_IF_CONDITION_EXPR, Warn,
    "braces can be eliminated in conditions that are expressions, e.g `if { true } ...`"
}

declare_lint! {
    pub BLOCK_IN_IF_CONDITION_STMT, Warn,
    "avoid complex blocks in conditions, instead move the block higher and bind it \
    with 'let'; e.g: `if { let x = true; x } ...`"
}

#[derive(Copy,Clone)]
pub struct BlockInIfCondition;

impl LintPass for BlockInIfCondition {
    fn get_lints(&self) -> LintArray {
        lint_array!(BLOCK_IN_IF_CONDITION_EXPR, BLOCK_IN_IF_CONDITION_STMT)
    }
}

struct ExVisitor<'v> {
    found_block: Option<&'v Expr>
}

impl<'v> Visitor<'v> for ExVisitor<'v> {
    fn visit_expr(&mut self, expr: &'v Expr) {
        if let ExprClosure(_, _, ref block) = expr.node {
            let complex = {
                if !block.stmts.is_empty() {
                    true
                } else {
                    if let Some(ref ex) = block.expr {
                        match ex.node {
                            ExprBlock(_) => true,
                            _ => false
                        }
                    } else {
                        false
                    }
                }
            };
            if complex {
                self.found_block = Some(& expr);
                return;
            }
        }
        walk_expr(self, expr);
    }
}

const BRACED_EXPR_MESSAGE:&'static str = "omit braces around single expression condition";
const COMPLEX_BLOCK_MESSAGE:&'static str = "in an 'if' condition, avoid complex blocks or closures with blocks; instead, move the block or closure higher and bind it with a 'let'";

impl LateLintPass for BlockInIfCondition {
    fn check_expr(&mut self, cx: &LateContext, expr: &Expr) {
        if let ExprIf(ref check, ref then, _) = expr.node {
            if let ExprBlock(ref block) = check.node {
                if block.stmts.is_empty() {
                    if let Some(ref ex) = block.expr {
                        // don't dig into the expression here, just suggest that they remove
                        // the block

                        span_help_and_lint(cx, BLOCK_IN_IF_CONDITION_EXPR, check.span,
                            BRACED_EXPR_MESSAGE,
                            &format!("try\nif {} {} ... ", snippet_block(cx, ex.span, ".."),
                            snippet_block(cx, then.span, "..")));
                    }
                } else {
                    // move block higher
                    span_help_and_lint(cx, BLOCK_IN_IF_CONDITION_STMT, check.span,
                        COMPLEX_BLOCK_MESSAGE,
                        &format!("try\nlet res = {};\nif res {} ... ",
                        snippet_block(cx, block.span, ".."),
                        snippet_block(cx, then.span, "..")));
                }
            } else {
                let mut visitor = ExVisitor { found_block: None };
                walk_expr(&mut visitor, check);
                if let Some(ref block) = visitor.found_block {
                    span_help_and_lint(cx, BLOCK_IN_IF_CONDITION_STMT, block.span,
                        COMPLEX_BLOCK_MESSAGE, "");
                }
            }
        }
    }
}
