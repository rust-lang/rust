mod byte;
mod byte_string;
mod char;
mod string;
mod block;

use crate::{
    SourceFile, SyntaxError, AstNode, SyntaxNode,
    SyntaxKind::{L_CURLY, R_CURLY, BYTE, BYTE_STRING, STRING, CHAR},
    ast,
    algo::visit::{visitor_ctx, VisitorCtx},
};

pub(crate) fn validate(file: &SourceFile) -> Vec<SyntaxError> {
    let mut errors = Vec::new();
    for node in file.syntax().descendants() {
        let _ = visitor_ctx(&mut errors)
            .visit::<ast::Literal, _>(validate_literal)
            .visit::<ast::Block, _>(block::validate_block_node)
            .accept(node);
    }
    errors
}

// FIXME: kill duplication
fn validate_literal(literal: &ast::Literal, acc: &mut Vec<SyntaxError>) {
    match literal.token().kind() {
        BYTE => byte::validate_byte_node(literal.token(), acc),
        BYTE_STRING => byte_string::validate_byte_string_node(literal.token(), acc),
        STRING => string::validate_string_node(literal.token(), acc),
        CHAR => char::validate_char_node(literal.token(), acc),
        _ => (),
    }
}

pub(crate) fn validate_block_structure(root: &SyntaxNode) {
    let mut stack = Vec::new();
    for node in root.descendants() {
        match node.kind() {
            L_CURLY => stack.push(node),
            R_CURLY => {
                if let Some(pair) = stack.pop() {
                    assert_eq!(
                        node.parent(),
                        pair.parent(),
                        "\nunpaired curleys:\n{}\n{}\n",
                        root.text(),
                        root.debug_dump(),
                    );
                    assert!(
                        node.next_sibling().is_none() && pair.prev_sibling().is_none(),
                        "\nfloating curlys at {:?}\nfile:\n{}\nerror:\n{}\n",
                        node,
                        root.text(),
                        node.text(),
                    );
                }
            }
            _ => (),
        }
    }
}
