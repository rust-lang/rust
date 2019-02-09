//! This crate provides some utilities for indenting rust code.
//!
use itertools::Itertools;
use ra_syntax::{
    AstNode,
    SyntaxNode, SyntaxKind::*,
    ast::{self, AstToken},
    algo::generate,
};

pub fn reindent(text: &str, indent: &str) -> String {
    let indent = format!("\n{}", indent);
    text.lines().intersperse(&indent).collect()
}

/// If the node is on the beginning of the line, calculate indent.
pub fn leading_indent(node: &SyntaxNode) -> Option<&str> {
    for leaf in prev_leaves(node) {
        if let Some(ws) = ast::Whitespace::cast(leaf) {
            let ws_text = ws.text();
            if let Some(pos) = ws_text.rfind('\n') {
                return Some(&ws_text[pos + 1..]);
            }
        }
        if leaf.leaf_text().unwrap().contains('\n') {
            break;
        }
    }
    None
}

fn prev_leaves(node: &SyntaxNode) -> impl Iterator<Item = &SyntaxNode> {
    generate(prev_leaf(node), |&node| prev_leaf(node))
}

fn prev_leaf(node: &SyntaxNode) -> Option<&SyntaxNode> {
    generate(node.ancestors().find_map(SyntaxNode::prev_sibling), |it| it.last_child()).last()
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

pub fn compute_ws(left: &SyntaxNode, right: &SyntaxNode) -> &'static str {
    match left.kind() {
        L_PAREN | L_BRACK => return "",
        L_CURLY => {
            if let USE_TREE = right.kind() {
                return "";
            }
        }
        _ => (),
    }
    match right.kind() {
        R_PAREN | R_BRACK => return "",
        R_CURLY => {
            if let USE_TREE = left.kind() {
                return "";
            }
        }
        DOT => return "",
        _ => (),
    }
    " "
}
