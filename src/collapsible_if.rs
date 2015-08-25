//! Checks for if expressions that contain only an if expression.
//!
//! For example, the lint would catch:
//!
//! ```
//! if x {
//!     if y {
//!         println!("Hello world");
//!     }
//! }
//! ```
//!
//! This lint is **warn** by default

use rustc::lint::*;
use syntax::ast::*;
use syntax::codemap::{Spanned, ExpnInfo};

use utils::{in_macro, span_help_and_lint, snippet, snippet_block};

declare_lint! {
    pub COLLAPSIBLE_IF,
    Warn,
    "two nested `if`-expressions can be collapsed into one, e.g. `if x { if y { foo() } }` \
     can be written as `if x && y { foo() }`"
}

#[derive(Copy,Clone)]
pub struct CollapsibleIf;

impl LintPass for CollapsibleIf {
    fn get_lints(&self) -> LintArray {
        lint_array!(COLLAPSIBLE_IF)
    }

    fn check_expr(&mut self, cx: &Context, expr: &Expr) {
        cx.sess().codemap().with_expn_info(expr.span.expn_id,
            |info| check_expr_expd(cx, expr, info))
    }
}

fn check_expr_expd(cx: &Context, e: &Expr, info: Option<&ExpnInfo>) {
    if in_macro(cx, info) { return; }

    if let ExprIf(ref check, ref then, None) = e.node {
        if let Some(&Expr{ node: ExprIf(ref check_inner, ref content, None), span: sp, ..}) =
            single_stmt_of_block(then) {
                if e.span.expn_id != sp.expn_id {
                    return;
                }
                span_help_and_lint(cx, COLLAPSIBLE_IF, e.span,
                    "this if statement can be collapsed",
                    &format!("try\nif {} && {} {}",
                             check_to_string(cx, check), check_to_string(cx, check_inner),
                             snippet_block(cx, content.span, "..")));
            }
    }
}

fn requires_brackets(e: &Expr) -> bool {
    match e.node {
        ExprBinary(Spanned {node: n, ..}, _, _) if n == BiEq => false,
        _ => true
    }
}

fn check_to_string(cx: &Context, e: &Expr) -> String {
    if requires_brackets(e) {
        format!("({})", snippet(cx, e.span, ".."))
    } else {
        format!("{}", snippet(cx, e.span, ".."))
    }
}

fn single_stmt_of_block(block: &Block) -> Option<&Expr> {
    if block.stmts.len() == 1 && block.expr.is_none() {
        if let StmtExpr(ref expr, _) = block.stmts[0].node {
            single_stmt_of_expr(expr)
        } else { None }
    } else {
        if block.stmts.is_empty() {
            if let Some(ref p) = block.expr { Some(p) } else { None }
        } else { None }
    }
}

fn single_stmt_of_expr(expr: &Expr) -> Option<&Expr> {
    if let ExprBlock(ref block) = expr.node {
        single_stmt_of_block(block)
    } else { Some(expr) }
}
