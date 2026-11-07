//! Utility functions for comparison operators.

#![deny(clippy::missing_docs_in_private_items)]

use rustc_hir::{BinOpKind, Expr};

#[derive(PartialEq, Eq, Debug, Copy, Clone)]
/// Represents a normalized comparison operator.
pub enum Rel {
    /// `<`
    Lt,
    /// `<=`
    Le,
    /// `==`
    Eq,
    /// `!=`
    Ne,
}

/// Put the expression in the form  `lhs < rhs`, `lhs <= rhs`, `lhs == rhs` or
/// `lhs != rhs`.
pub fn normalize_comparison<'a>(
    op: BinOpKind,
    lhs: &'a Expr<'a>,
    rhs: &'a Expr<'a>,
) -> Option<(Rel, &'a Expr<'a>, &'a Expr<'a>)> {
    match op {
        BinOpKind::Lt => Some((Rel::Lt, lhs, rhs)),
        BinOpKind::Le => Some((Rel::Le, lhs, rhs)),
        BinOpKind::Gt => Some((Rel::Lt, rhs, lhs)),
        BinOpKind::Ge => Some((Rel::Le, rhs, lhs)),
        BinOpKind::Eq => Some((Rel::Eq, rhs, lhs)),
        BinOpKind::Ne => Some((Rel::Ne, rhs, lhs)),
        _ => None,
    }
}
