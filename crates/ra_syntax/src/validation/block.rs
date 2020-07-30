//! Logic for validating block expressions i.e. `ast::BlockExpr`.

use crate::{
    ast::{self, AstNode, AttrsOwner},
    SyntaxError,
    SyntaxKind::*,
};

pub(crate) fn validate_block_expr(block: ast::BlockExpr, errors: &mut Vec<SyntaxError>) {
    if let Some(parent) = block.syntax().parent() {
        match parent.kind() {
            FN | EXPR_STMT | BLOCK_EXPR => return,
            _ => {}
        }
    }
    errors.extend(block.attrs().map(|attr| {
        SyntaxError::new(
            "A block in this position cannot accept inner attributes",
            attr.syntax().text_range(),
        )
    }))
}
