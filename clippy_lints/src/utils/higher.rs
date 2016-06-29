//! This module contains functions for retrieve the original AST from lowered `hir`.

use rustc::hir;
use syntax::ast;

/// Convert a hir binary operator to the corresponding `ast` type.
pub fn binop(op: hir::BinOp_) -> ast::BinOpKind {
    match op {
        hir::BiEq => ast::BinOpKind::Eq,
        hir::BiGe => ast::BinOpKind::Ge,
        hir::BiGt => ast::BinOpKind::Gt,
        hir::BiLe => ast::BinOpKind::Le,
        hir::BiLt => ast::BinOpKind::Lt,
        hir::BiNe => ast::BinOpKind::Ne,
        hir::BiOr => ast::BinOpKind::Or,
        hir::BiAdd => ast::BinOpKind::Add,
        hir::BiAnd => ast::BinOpKind::And,
        hir::BiBitAnd => ast::BinOpKind::BitAnd,
        hir::BiBitOr => ast::BinOpKind::BitOr,
        hir::BiBitXor => ast::BinOpKind::BitXor,
        hir::BiDiv => ast::BinOpKind::Div,
        hir::BiMul => ast::BinOpKind::Mul,
        hir::BiRem => ast::BinOpKind::Rem,
        hir::BiShl => ast::BinOpKind::Shl,
        hir::BiShr => ast::BinOpKind::Shr,
        hir::BiSub => ast::BinOpKind::Sub,
    }
}
