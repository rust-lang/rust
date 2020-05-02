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

pub fn unwrap_trivial_block(block: ast::BlockExpr) -> ast::Expr {
    extract_trivial_expression(&block)
        .filter(|expr| !expr.syntax().text().contains_char('\n'))
        .unwrap_or_else(|| block.into())
}

pub fn extract_trivial_expression(block: &ast::BlockExpr) -> Option<ast::Expr> {
    let has_anything_else = |thing: &SyntaxNode| -> bool {
        let mut non_trivial_children =
            block.syntax().children_with_tokens().filter(|it| match it.kind() {
                WHITESPACE | T!['{'] | T!['}'] => false,
                _ => it.as_node() != Some(thing),
            });
        non_trivial_children.next().is_some()
    };

    if let Some(expr) = block.expr() {
        if has_anything_else(expr.syntax()) {
            return None;
        }
        return Some(expr);
    }
    // Unwrap `{ continue; }`
    let (stmt,) = block.statements().next_tuple()?;
    if let ast::Stmt::ExprStmt(expr_stmt) = stmt {
        if has_anything_else(expr_stmt.syntax()) {
            return None;
        }
        let expr = expr_stmt.expr()?;
        match expr.syntax().kind() {
            CONTINUE_EXPR | BREAK_EXPR | RETURN_EXPR => return Some(expr),
            _ => (),
        }
    }
    None
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
