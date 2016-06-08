use rustc::lint::*;
use syntax::ast::*;
use syntax::codemap::{Span, Spanned};
use syntax::visit::FnKind;

use utils::{span_note_and_lint, span_lint_and_then, snippet_opt, match_path_ast, in_external_macro};

/// **What it does:** This lint checks for return statements at the end of a block.
///
/// **Why is this bad?** Removing the `return` and semicolon will make the code more rusty.
///
/// **Known problems:** Following this lint's advice may currently run afoul of Rust issue [#31439](https://github.com/rust-lang/rust/issues/31439), so if you get lifetime errors, please roll back the change until that issue is fixed.
///
/// **Example:** `fn foo(x: usize) { return x; }`
declare_lint! {
    pub NEEDLESS_RETURN, Warn,
    "using a return statement like `return expr;` where an expression would suffice"
}

/// **What it does:** This lint checks for `let`-bindings, which are subsequently returned.
///
/// **Why is this bad?** It is just extraneous code. Remove it to make your code more rusty.
///
/// **Known problems:** Following this lint's advice may currently run afoul of Rust issue [#31439](https://github.com/rust-lang/rust/issues/31439), so if you get lifetime errors, please roll back the change until that issue is fixed.
///
/// **Example:** `{ let x = ..; x }`
declare_lint! {
    pub LET_AND_RETURN, Warn,
    "creating a let-binding and then immediately returning it like `let x = expr; x` at \
     the end of a block"
}

#[derive(Copy, Clone)]
pub struct ReturnPass;

impl ReturnPass {
    // Check the final stmt or expr in a block for unnecessary return.
    fn check_block_return(&mut self, cx: &EarlyContext, block: &Block) {
        if let Some(ref expr) = block.expr {
            self.check_final_expr(cx, expr);
        } else if let Some(stmt) = block.stmts.last() {
            if let StmtKind::Semi(ref expr, _) = stmt.node {
                if let ExprKind::Ret(Some(ref inner)) = expr.node {
                    self.emit_return_lint(cx, (stmt.span, inner.span));
                }
            }
        }
    }

    // Check a the final expression in a block if it's a return.
    fn check_final_expr(&mut self, cx: &EarlyContext, expr: &Expr) {
        match expr.node {
            // simple return is always "bad"
            ExprKind::Ret(Some(ref inner)) => {
                self.emit_return_lint(cx, (expr.span, inner.span));
            }
            // a whole block? check it!
            ExprKind::Block(ref block) => {
                self.check_block_return(cx, block);
            }
            // an if/if let expr, check both exprs
            // note, if without else is going to be a type checking error anyways
            // (except for unit type functions) so we don't match it
            ExprKind::If(_, ref ifblock, Some(ref elsexpr)) => {
                self.check_block_return(cx, ifblock);
                self.check_final_expr(cx, elsexpr);
            }
            // a match expr, check all arms
            ExprKind::Match(_, ref arms) => {
                for arm in arms {
                    self.check_final_expr(cx, &arm.body);
                }
            }
            _ => (),
        }
    }

    fn emit_return_lint(&mut self, cx: &EarlyContext, spans: (Span, Span)) {
        if in_external_macro(cx, spans.1) {
            return;
        }
        span_lint_and_then(cx, NEEDLESS_RETURN, spans.0, "unneeded return statement", |db| {
            if let Some(snippet) = snippet_opt(cx, spans.1) {
                db.span_suggestion(spans.0, "remove `return` as shown:", snippet);
            }
        });
    }

    // Check for "let x = EXPR; x"
    fn check_let_return(&mut self, cx: &EarlyContext, block: &Block) {
        // we need both a let-binding stmt and an expr
        if_let_chain! {
            [
                let Some(stmt) = block.stmts.last(),
                let Some(ref retexpr) = block.expr,
                let StmtKind::Decl(ref decl, _) = stmt.node,
                let DeclKind::Local(ref local) = decl.node,
                local.ty.is_none(),
                let Some(ref initexpr) = local.init,
                let PatKind::Ident(_, Spanned { node: id, .. }, _) = local.pat.node,
                let ExprKind::Path(_, ref path) = retexpr.node,
                match_path_ast(path, &[&id.name.as_str()]),
                !in_external_macro(cx, initexpr.span),
            ], {
                span_note_and_lint(cx,
                                   LET_AND_RETURN,
                                   retexpr.span,
                                   "returning the result of a let binding from a block. \
                                   Consider returning the expression directly.",
                                   initexpr.span,
                                   "this expression can be directly returned");
            }
        }
    }
}

impl LintPass for ReturnPass {
    fn get_lints(&self) -> LintArray {
        lint_array!(NEEDLESS_RETURN, LET_AND_RETURN)
    }
}

impl EarlyLintPass for ReturnPass {
    fn check_fn(&mut self, cx: &EarlyContext, _: FnKind, _: &FnDecl, block: &Block, _: Span, _: NodeId) {
        self.check_block_return(cx, block);
    }

    fn check_block(&mut self, cx: &EarlyContext, block: &Block) {
        self.check_let_return(cx, block);
    }
}
