//! Logic for validating block expressions i.e. `ast::BlockExpr`.

use crate::{
    ast::{self, AstNode, AttrsOwner},
    SyntaxError,
    SyntaxKind::*,
};

pub(crate) fn validate_block_expr(expr: ast::BlockExpr, errors: &mut Vec<SyntaxError>) {
    if let Some(parent) = expr.syntax().parent() {
        match parent.kind() {
            FN_DEF | EXPR_STMT | BLOCK => return,
            _ => {}
        }
    }
    if let Some(block) = expr.block() {
        errors.extend(block.attrs().map(|attr| {
            SyntaxError::new(
                "A block in this position cannot accept inner attributes",
                attr.syntax().text_range(),
            )
        }))
    }
}
