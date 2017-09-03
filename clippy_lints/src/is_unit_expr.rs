use rustc::lint::*;
use syntax::ast::*;
use std::ops::Deref;
use syntax::ext::quote::rt::Span;
use utils::span_note_and_lint;

/// **What it does:** Checks for
///  - () being assigned to a variable
///  - () being passed to a function
///
/// **Why is this bad?** It is extremely unlikely that a user intended to
/// assign '()' to valiable. Instead,
/// Unit is what a block evaluates to when it returns nothing. This is
/// typically caused by a trailing
///   unintended semicolon.
///
/// **Known problems:** None.
///
/// **Example:**
/// * `let x = {"foo" ;}` when the user almost certainly intended `let x
/// ={"foo"}`

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
            if let Some(span) = is_unit_expr(right) {
                span_note_and_lint(cx, UNIT_EXPR, expr.span,"This expression assigns the Unit type ()",span,"Consider removing the trailing semicolon");
            }
        }
        if let ExprKind::MethodCall(ref _left, ref args) = expr.node {
            for ref arg in args {
                if let Some(span) = is_unit_expr(arg) {
                span_note_and_lint(cx, UNIT_EXPR, expr.span,"This expression assigns the Unit type ()",span,"Consider removing the trailing semicolon");
                }
            }
        }
        if let ExprKind::Call(_, ref args) = expr.node {
            for ref arg in args {
                if let Some(span) = is_unit_expr(arg) {
                span_note_and_lint(cx, UNIT_EXPR, expr.span,"This expression assigns the Unit type ()",span,"Consider removing the trailing semicolon");
                }
            }
        }
    }

    fn check_stmt(&mut self, cx: &EarlyContext, stmt: &Stmt) {
        if let StmtKind::Local(ref local) = stmt.node {
            if local.pat.node == PatKind::Wild {
                return;
            }
            if let Some(ref expr) = local.init {
                if let Some(span) = is_unit_expr(expr) {
                span_note_and_lint(cx, UNIT_EXPR, expr.span,"This expression assigns the Unit type ()",span,"Consider removing the trailing semicolon");
                }
            }
        }
    }
}
fn is_unit_expr(expr: &Expr) -> Option<Span> {
    match expr.node {
        ExprKind::Block(ref block) => if check_last_stmt_in_block(block) {
            return Some(block.stmts[block.stmts.len() - 1].span.clone());
        } else {
            return None;
        },
        ExprKind::If(_, ref then, ref else_) => {
            let check_then = check_last_stmt_in_block(then);
            if let Some(ref else_) = *else_ {
                let check_else = is_unit_expr(else_.deref());
                if let Some(ref expr_else) = check_else {
                    return Some(expr_else.clone());
                } 
            }
            if check_then {
                return Some(expr.span.clone());
            } else {
                return None;
            }
        },
        ExprKind::Match(ref _pattern, ref arms) => {
            for ref arm in arms {
                if let Some(expr) = is_unit_expr(&arm.body) {
                    return Some(expr);
                }
            }
            return None;
        },
        _ => return None,
    }
}

fn check_last_stmt_in_block(block: &Block) -> bool {
    let ref final_stmt = &block.stmts[block.stmts.len() - 1];
    
    match final_stmt.node{
        StmtKind::Expr(_) => return false,
        StmtKind::Semi(ref expr)=>{
               match expr.node{
                ExprKind::Break(_,_) => return false,
                ExprKind::Ret(_) => return false,
                _ => return true,
            }         
        },
        _ => return true,
    }

}
