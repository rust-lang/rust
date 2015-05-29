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
use syntax::codemap::{Span, Spanned};
use syntax::print::pprust::expr_to_string;

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
    
    fn check_expr(&mut self, cx: &Context, e: &Expr) {
        if let ExprIf(ref check, ref then_block, None) = e.node {
            let expr = check_block(then_block);
            let expr = match expr {
                Some(e) => e,
                None => return
            };
            if let ExprIf(ref check_inner, _, None) = expr.node {
                let (check, check_inner) = (check_to_string(check), check_to_string(check_inner));
                cx.span_lint(COLLAPSIBLE_IF, e.span,
                             &format!("This if statement can be collapsed. Try: if {} && {}", check, check_inner));
            }
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

fn check_block(b: &Block) -> Option<&P<Expr>> {
    if b.stmts.len() == 1 && b.expr.is_none() {
        let stmt = &b.stmts[0];
        return match stmt.node {
            StmtExpr(ref e, _) => Some(e),
            _ => None
        };
    }
    if let Some(ref e) = b.expr {
        return Some(e);
    }
    None
}
