//! Logic for validating block expressions i.e. `ast::BlockExpr`.

use crate::{
    ast::{self, AstNode, HasAttrs},
    SyntaxError,
    SyntaxKind::*,
};

pub(crate) fn validate_block_expr(block: ast::BlockExpr, errors: &mut Vec<SyntaxError>) {
    if let Some(parent) = block.syntax().parent() {
        match parent.kind() {
            FN | EXPR_STMT | STMT_LIST => return,
            _ => {}
        }
    }
    if let Some(stmt_list) = block.stmt_list() {
        errors.extend(stmt_list.attrs().filter(|attr| attr.kind().is_inner()).map(|attr| {
            SyntaxError::new(
                "A block in this position cannot accept inner attributes",
                attr.syntax().text_range(),
            )
        }));
    }
}
