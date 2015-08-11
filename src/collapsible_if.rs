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

use rustc::plugin::Registry;
use rustc::lint::*;
use rustc::middle::def::*;
use syntax::ast::*;
use syntax::ptr::P;
use syntax::codemap::{Span, Spanned, ExpnInfo};
use syntax::print::pprust::expr_to_string;
use utils::{in_macro, span_lint};

declare_lint! {
    pub COLLAPSIBLE_IF,
    Warn,
    "Warn on if expressions that can be collapsed"
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
        if let Some(&Expr{ node: ExprIf(ref check_inner, _, None), ..}) =
            single_stmt_of_block(then) {
                span_lint(cx, COLLAPSIBLE_IF, e.span, &format!(
                    "This if statement can be collapsed. Try: if {} && {}\n{:?}",
                    check_to_string(check), check_to_string(check_inner), e));
            }
    }
}

fn requires_brackets(e: &Expr) -> bool {
    match e.node {
        ExprBinary(Spanned {node: n, ..}, _, _) if n == BiEq => false,
        _ => true
    }
}

fn check_to_string(e: &Expr) -> String {
    if requires_brackets(e) {
        format!("({})", expr_to_string(e))
    } else {
        format!("{}", expr_to_string(e))
    }
}

fn single_stmt_of_block(block: &Block) -> Option<&Expr> {
    if block.stmts.len() == 1 && block.expr.is_none() {
        if let StmtExpr(ref expr, _) = block.stmts[0].node {
            single_stmt_of_expr(expr)
        } else { None }
    } else {
        if block.stmts.is_empty() {
            if let Some(ref p) = block.expr { Some(&*p) } else { None }
        } else { None }
    }
}

fn single_stmt_of_expr(expr: &Expr) -> Option<&Expr> {
    if let ExprBlock(ref block) = expr.node {
        single_stmt_of_block(block)
    } else { Some(expr) }
}
