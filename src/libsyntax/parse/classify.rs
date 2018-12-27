//! Routines the parser uses to classify AST nodes

// Predicates on exprs and stmts that the pretty-printer and parser use

use ast;

/// Does this expression require a semicolon to be treated
/// as a statement? The negation of this: 'can this expression
/// be used as a statement without a semicolon' -- is used
/// as an early-bail-out in the parser so that, for instance,
///     if true {...} else {...}
///      |x| 5
/// isn't parsed as (if true {...} else {...} | x) | 5
pub fn expr_requires_semi_to_be_stmt(e: &ast::Expr) -> bool {
    match e.node {
        ast::ExprKind::If(..) |
        ast::ExprKind::IfLet(..) |
        ast::ExprKind::Match(..) |
        ast::ExprKind::Block(..) |
        ast::ExprKind::While(..) |
        ast::ExprKind::WhileLet(..) |
        ast::ExprKind::Loop(..) |
        ast::ExprKind::ForLoop(..) |
        ast::ExprKind::TryBlock(..) => false,
        _ => true,
    }
}

/// this statement requires a semicolon after it.
/// note that in one case (`stmt_semi`), we've already
/// seen the semicolon, and thus don't need another.
pub fn stmt_ends_with_semi(stmt: &ast::StmtKind) -> bool {
    match *stmt {
        ast::StmtKind::Local(_) => true,
        ast::StmtKind::Expr(ref e) => expr_requires_semi_to_be_stmt(e),
        ast::StmtKind::Item(_) |
        ast::StmtKind::Semi(..) |
        ast::StmtKind::Mac(..) => false,
    }
}
