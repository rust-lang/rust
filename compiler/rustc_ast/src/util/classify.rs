//! Routines the parser uses to classify AST nodes

// Predicates on exprs and stmts that the pretty-printer and parser use

use crate::ast;

/// Does this expression require a semicolon to be treated
/// as a statement? The negation of this: 'can this expression
/// be used as a statement without a semicolon' -- is used
/// as an early-bail-out in the parser so that, for instance,
///     if true {...} else {...}
///      |x| 5
/// isn't parsed as (if true {...} else {...} | x) | 5
pub fn expr_requires_semi_to_be_stmt(e: &ast::Expr) -> bool {
    !matches!(
        e.kind,
        ast::ExprKind::If(..)
            | ast::ExprKind::Match(..)
            | ast::ExprKind::Block(..)
            | ast::ExprKind::While(..)
            | ast::ExprKind::Loop(..)
            | ast::ExprKind::ForLoop(..)
            | ast::ExprKind::TryBlock(..)
    )
}
