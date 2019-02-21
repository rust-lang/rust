mod byte;
mod byte_string;
mod char;
mod string;
mod block;

use crate::{
    SourceFile, SyntaxError, AstNode, SyntaxNode,
    SyntaxKind::{L_CURLY, R_CURLY},
    ast,
    algo::visit::{visitor_ctx, VisitorCtx},
};

pub(crate) fn validate(file: &SourceFile) -> Vec<SyntaxError> {
    let mut errors = Vec::new();
    for node in file.syntax().descendants() {
        let _ = visitor_ctx(&mut errors)
            .visit::<ast::Byte, _>(byte::validate_byte_node)
            .visit::<ast::ByteString, _>(byte_string::validate_byte_string_node)
            .visit::<ast::Char, _>(char::validate_char_node)
            .visit::<ast::String, _>(string::validate_string_node)
            .visit::<ast::Block, _>(block::validate_block_node)
            .accept(node);
    }
    errors
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
