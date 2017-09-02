use rustc::lint::*;
use syntax::ast::*;
use syntax::codemap::Spanned;
use utils::{span_lint_and_sugg, snippet};


/// **What it does:** Checks for 
///  - () being assigned to a variable
///  - () being passed to a function
///
/// **Why is this bad?** It is extremely unlikely that a user intended to assign '()' to valiable. Instead,
///   Unit is what a block evaluates to when it returns nothing. This is typically caused by a trailing 
///   unintended semicolon. 
///
/// **Known problems:** None.
///
/// **Example:**
/// * `let x = {"foo" ;}` when the user almost certainly intended `let x ={"foo"}`

declare_lint! {
    pub UNIT_EXPR,
    Warn,
    "unintended assignment or use of a unit typed value"
}

#[derive(Copy, Clone)]
pub struct UnitExpr;

impl LintPass for UnitExpr {
    fn get_lints(&self) -> LintArray {
        lint_array!(UNIT_EXPR)
    }
}

impl EarlyLintPass for UnitExpr {
    fn check_expr(&mut self, cx: &EarlyContext, expr: &Expr) {
        if let ExprKind::Assign(ref left, ref right) = expr.node {
            unimplemented!();
        }
        if let ExprKind::MethodCall(ref path, ref args) = expr.node {
            unimplemented!();
        }
        if let ExprKind::Call(ref path, ref args) = expr.node{
            unimplemented!();
        }
    }
}
