//! This crate provides some utilities for indenting rust code.

use std::iter::successors;

use syntax::{
    ast::{self, AstToken},
    SmolStr, SyntaxNode, SyntaxToken,
};

/// If the node is on the beginning of the line, calculate indent.
pub fn leading_indent(node: &SyntaxNode) -> Option<SmolStr> {
    for token in prev_tokens(node.first_token()?) {
        if let Some(ws) = ast::Whitespace::cast(token.clone()) {
            let ws_text = ws.text();
            if let Some(pos) = ws_text.rfind('\n') {
                return Some(ws_text[pos + 1..].into());
            }
        }
        if token.text().contains('\n') {
            break;
        }
    }
    None
}

fn prev_tokens(token: SyntaxToken) -> impl Iterator<Item = SyntaxToken> {
    successors(token.prev_token(), |token| token.prev_token())
}
