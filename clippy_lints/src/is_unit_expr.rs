use rustc::lint::*;
use syntax::ast::*;
use syntax::codemap::Spanned;
use utils::{span_lint_and_sugg, snippet};
use std::ops::Deref;

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
        if let ExprKind::Assign(ref _left, ref right) = expr.node {
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
        if let ExprKind::MethodCall(ref _left, ref args) = expr.node {
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
                }            
            }
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
                }            
            }        
        }
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
         ExprKind::Block(ref block) => {
             return check_last_stmt_in_block(block);
        },
        ExprKind::If(_, ref then, ref else_)=>{
            let check_then = check_last_stmt_in_block(then);
            if let Some(ref else_) = *else_{
                return check_then && is_unit_expr(else_.deref());
            } 
            return check_then;
        }
        _ => return false,
    }
}

fn check_last_stmt_in_block(block: &Block)->bool{
    let ref final_stmt = &block.stmts[block.stmts.len()-1];
        if let StmtKind::Expr(_) = final_stmt.node{
                return false;
            }
            else{
                return true;
            }
}