//! This crate provides some utilities for indenting rust code.
//!
use itertools::Itertools;
use ra_syntax::{
    AstNode,
    SyntaxNode, SyntaxKind::*, SyntaxToken, SyntaxKind,
    ast,
    algo::generate,
};

pub fn reindent(text: &str, indent: &str) -> String {
    let indent = format!("\n{}", indent);
    text.lines().intersperse(&indent).collect()
}

/// If the node is on the beginning of the line, calculate indent.
pub fn leading_indent(node: &SyntaxNode) -> Option<&str> {
    for token in prev_tokens(node.first_token()?) {
        if let Some(ws) = ast::Whitespace::cast(token) {
            let ws_text = ws.text();
            if let Some(pos) = ws_text.rfind('\n') {
                return Some(&ws_text[pos + 1..]);
            }
        }
        if token.text().contains('\n') {
            break;
        }
    }
    None
}

fn prev_tokens(token: SyntaxToken) -> impl Iterator<Item = SyntaxToken> {
    generate(token.prev_token(), |&token| token.prev_token())
}

pub fn extract_trivial_expression(block: &ast::Block) -> Option<&ast::Expr> {
    let expr = block.expr()?;
    if expr.syntax().text().contains('\n') {
        return None;
    }
    let non_trivial_children = block.syntax().children().filter(|it| match it.kind() {
        WHITESPACE | L_CURLY | R_CURLY => false,
        _ => it != &expr.syntax(),
    });
    if non_trivial_children.count() > 0 {
        return None;
    }
    Some(expr)
}

pub fn compute_ws(left: SyntaxKind, right: SyntaxKind) -> &'static str {
    match left {
        L_PAREN | L_BRACK => return "",
        L_CURLY => {
            if let USE_TREE = right {
                return "";
            }
        }
        _ => (),
    }
    match right {
        R_PAREN | R_BRACK => return "",
        R_CURLY => {
            if let USE_TREE = left {
                return "";
            }
        }
        DOT => return "",
        _ => (),
    }
    " "
}
