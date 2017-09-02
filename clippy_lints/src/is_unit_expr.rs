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
            if is_unit_expr(right){
                span_lint_and_sugg(
                    cx,
                    UNIT_EXPR,
                    right.span,
                    "trailing semicolons can be tricky",
                    "remove the last semicolon",
                    "TODO".to_owned()
                )
            }
        }
        if let ExprKind::MethodCall(_, ref args) = expr.node {
            for ref arg in args{
                if is_unit_expr(arg){
                    span_lint_and_sugg(
                        cx,
                        UNIT_EXPR,
                        arg.span,
                        "trailing semicolons can be tricky",
                        "remove the last semicolon",
                        "TODO".to_owned()
                    )
                }            }
        }
        if let ExprKind::Call( _, ref args) = expr.node{
            for ref arg in args{
                if is_unit_expr(arg){
                    span_lint_and_sugg(
                        cx,
                        UNIT_EXPR,
                        arg.span,
                        "trailing semicolons can be tricky",
                        "remove the last semicolon",
                        "TODO".to_owned()
                    )
                }            }        }
    }

    fn check_stmt(&mut self, cx: &EarlyContext, stmt: &Stmt) {
        if let StmtKind::Local(ref local) = stmt.node{
            if local.pat.node == PatKind::Wild {return;}
            if let Some(ref expr) = local.init{
                if is_unit_expr(expr){
                    span_lint_and_sugg(
                        cx,
                        UNIT_EXPR,
                        local.span,
                        "trailing semicolons can be tricky",
                        "remove the last semicolon",
                        "TODO".to_owned()
                    )
                }
            }        
            }
    }
}

fn is_unit_expr(expr: &Expr)->bool{
    match expr.node{
         ExprKind::Block(ref next) => {
            let ref final_stmt = &next.stmts[next.stmts.len()-1];
            if let StmtKind::Expr(_) = final_stmt.node{
                return false;
            }
            else{
                return true;
            }
        },
        _ => return false,
    }
}