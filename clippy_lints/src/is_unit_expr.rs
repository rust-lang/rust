use rustc::lint::*;
use syntax::ast::*;
use syntax::ext::quote::rt::Span;
use utils::{span_lint, span_note_and_lint};

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
enum UnitCause {
    SemiColon,
    EmptyBlock,
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
            check_for_unit(cx, right);
        }
        if let ExprKind::MethodCall(ref _left, ref args) = expr.node {
            for arg in args {
                check_for_unit(cx, arg);
            }
        }
        if let ExprKind::Call(_, ref args) = expr.node {
            for arg in args {
                check_for_unit(cx, arg);
            }
        }
    }

    fn check_stmt(&mut self, cx: &EarlyContext, stmt: &Stmt) {
        if let StmtKind::Local(ref local) = stmt.node {
            if local.pat.node == PatKind::Wild {
                return;
            }
            if let Some(ref expr) = local.init {
                check_for_unit(cx, expr);
            }
        }
    }
}

fn check_for_unit(cx: &EarlyContext, expr: &Expr) {
    match is_unit_expr(expr) {
        Some((span, UnitCause::SemiColon)) => span_note_and_lint(
            cx,
            UNIT_EXPR,
            expr.span,
            "This expression evaluates to the Unit type ()",
            span,
            "Consider removing the trailing semicolon",
        ),
        Some((_span, UnitCause::EmptyBlock)) => span_lint(
            cx,
            UNIT_EXPR,
            expr.span,
            "This expression evaluates to the Unit type ()",
        ),
        None => (),
    }
}

fn is_unit_expr(expr: &Expr) -> Option<(Span, UnitCause)> {
    match expr.node {
        ExprKind::Block(ref block) => match check_last_stmt_in_block(block) {
            Some(UnitCause::SemiColon) =>
                Some((block.stmts[block.stmts.len() - 1].span, UnitCause::SemiColon)),
            Some(UnitCause::EmptyBlock) =>
                Some((block.span, UnitCause::EmptyBlock)),
            None => None
        }
        ExprKind::If(_, ref then, ref else_) => {
            let check_then = check_last_stmt_in_block(then);
            if let Some(ref else_) = *else_ {
                let check_else = is_unit_expr(else_);
                if let Some(ref expr_else) = check_else {
                    return Some(*expr_else);
                }
            }
            match check_then {
                Some(c) => Some((expr.span, c)),
                None => None,
            }
        },
        ExprKind::Match(ref _pattern, ref arms) => {
            for arm in arms {
                if let Some(r) = is_unit_expr(&arm.body) {
                    return Some(r);
                }
            }
            None
        },
        _ => None,
    }
}

fn check_last_stmt_in_block(block: &Block) -> Option<UnitCause> {
    if block.stmts.is_empty() { return Some(UnitCause::EmptyBlock); }
    let final_stmt = &block.stmts[block.stmts.len() - 1];


    // Made a choice here to risk false positives on divergent macro invocations
    // like `panic!()`
    match final_stmt.node {
        StmtKind::Expr(_) => None,
        StmtKind::Semi(ref expr) => match expr.node {
            ExprKind::Break(_, _) | ExprKind::Continue(_) | ExprKind::Ret(_) => None,
            _ => Some(UnitCause::SemiColon),
        },
        _ => Some(UnitCause::SemiColon), // not sure what's happening here
    }
}
