//! This crate provides some utilities for indenting rust code.

use std::iter::successors;

use itertools::Itertools;
use ra_syntax::{
    ast::{self, AstNode, AstToken},
    SmolStr, SyntaxKind,
    SyntaxKind::*,
    SyntaxNode, SyntaxToken, T,
};

pub fn reindent(text: &str, indent: &str) -> String {
    let indent = format!("\n{}", indent);
    text.lines().intersperse(&indent).collect()
}

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

pub fn extract_trivial_expression(expr: &ast::BlockExpr) -> Option<ast::Expr> {
    let block = expr.block()?;
    let expr = block.expr()?;
    let non_trivial_children = block.syntax().children().filter(|it| match it.kind() {
        WHITESPACE | T!['{'] | T!['}'] => false,
        _ => it != expr.syntax(),
    });
    if non_trivial_children.count() > 0 {
        return None;
    }
    Some(expr)
}

pub fn compute_ws(left: SyntaxKind, right: SyntaxKind) -> &'static str {
    match left {
        T!['('] | T!['['] => return "",
        T!['{'] => {
            if let USE_TREE = right {
                return "";
            }
        }
        _ => (),
    }
    match right {
        T![')'] | T![']'] => return "",
        T!['}'] => {
            if let USE_TREE = left {
                return "";
            }
        }
        T![.] => return "",
        _ => (),
    }
    " "
}
