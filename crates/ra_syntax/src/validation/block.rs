use crate::{SyntaxKind::*,
    ast::{self, AttrsOwner, AstNode},
    SyntaxError,
    SyntaxErrorKind::*,
};

pub(crate) fn validate_block_node(node: &ast::Block, errors: &mut Vec<SyntaxError>) {
    if let Some(parent) = node.syntax().parent() {
        match parent.kind() {
            FN_DEF => return,
            BLOCK_EXPR => match parent.parent().map(|v| v.kind()) {
                Some(EXPR_STMT) | Some(BLOCK) => return,
                _ => {}
            },
            _ => {}
        }
    }
    errors
        .extend(node.attrs().map(|attr| SyntaxError::new(InvalidBlockAttr, attr.syntax().range())))
}
