use syntax::ast;
use syntax::ast::*;
use syntax::codemap::Span;
use syntax::visit::FnKind;
use rustc::lint::{Context, LintPass, LintArray};

use utils::{span_lint, snippet};

declare_lint!(pub NEEDLESS_RETURN, Warn,
              "Warn on using a return statement where an expression would be enough");

#[derive(Copy,Clone)]
pub struct ReturnPass;

impl ReturnPass {
    // Check the final stmt or expr in a block for unnecessary return.
    fn check_block_return(&mut self, cx: &Context, block: &Block) {
        if let Some(ref expr) = block.expr {
            self.check_final_expr(cx, expr);
        } else if let Some(stmt) = block.stmts.last() {
            if let StmtSemi(ref expr, _) = stmt.node {
                if let ExprRet(Some(ref inner)) = expr.node {
                    self.emit_lint(cx, (expr.span, inner.span));
                }
            }
        }
    }

    // Check a the final expression in a block if it's a return.
    fn check_final_expr(&mut self, cx: &Context, expr: &Expr) {
        match expr.node {
            // simple return is always "bad"
            ExprRet(Some(ref inner)) => {
                self.emit_lint(cx, (expr.span, inner.span));
            }
            // a whole block? check it!
            ExprBlock(ref block) => {
                self.check_block_return(cx, block);
            }
            // an if/if let expr, check both exprs
            // note, if without else is going to be a type checking error anyways
            // (except for unit type functions) so we don't match it
            ExprIf(_, ref ifblock, Some(ref elsexpr)) |
            ExprIfLet(_, _, ref ifblock, Some(ref elsexpr)) => {
                self.check_block_return(cx, ifblock);
                self.check_final_expr(cx, elsexpr);
            }
            // a match expr, check all arms
            ExprMatch(_, ref arms, _) => {
                for arm in arms {
                    self.check_final_expr(cx, &*arm.body);
                }
            }
            _ => { }
        }
    }

    fn emit_lint(&mut self, cx: &Context, spans: (Span, Span)) {
        span_lint(cx, NEEDLESS_RETURN, spans.0, &format!(
            "unneeded return statement. Consider using {} \
             without the trailing semicolon",
            snippet(cx, spans.1, "..")))
    }
}

impl LintPass for ReturnPass {
    fn get_lints(&self) -> LintArray {
        lint_array!(NEEDLESS_RETURN)
    }

    fn check_fn(&mut self, cx: &Context, _: FnKind, _: &FnDecl,
                block: &Block, _: Span, _: ast::NodeId) {
        self.check_block_return(cx, block);
    }
}
