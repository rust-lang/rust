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
        if let Some(stmt) = block.stmts.last() {
            match stmt.node {
                StmtKind::Expr(ref expr) | StmtKind::Semi(ref expr) => {
                    self.check_final_expr(cx, expr, Some(stmt.span));
                }
                _ => (),
            }
        }
    }

    // Check a the final expression in a block if it's a return.
    fn check_final_expr(&mut self, cx: &EarlyContext, expr: &Expr, span: Option<Span>) {
        match expr.node {
            // simple return is always "bad"
            ExprKind::Ret(Some(ref inner)) => {
                self.emit_return_lint(cx, span.expect("`else return` is not possible"), inner.span);
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
                self.check_final_expr(cx, elsexpr, None);
            }
            // a match expr, check all arms
            ExprKind::Match(_, ref arms) => {
                for arm in arms {
                    self.check_final_expr(cx, &arm.body, Some(arm.body.span));
                }
            }
            _ => (),
        }
    }

    fn emit_return_lint(&mut self, cx: &EarlyContext, ret_span: Span, inner_span: Span) {
        if in_external_macro(cx, inner_span) {
            return;
        }
        span_lint_and_then(cx, NEEDLESS_RETURN, ret_span, "unneeded return statement", |db| {
            if let Some(snippet) = snippet_opt(cx, inner_span) {
                db.span_suggestion(ret_span, "remove `return` as shown:", snippet);
            }
        });
    }

    // Check for "let x = EXPR; x"
    fn check_let_return(&mut self, cx: &EarlyContext, block: &Block) {
        let mut it = block.stmts.iter();

        // we need both a let-binding stmt and an expr
        if_let_chain! {[
            let Some(ref retexpr) = it.next_back(),
            let StmtKind::Expr(ref retexpr) = retexpr.node,
            let Some(stmt) = it.next_back(),
            let StmtKind::Local(ref local) = stmt.node,
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
        }}
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
