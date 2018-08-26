use libsyntax2::{
    File, TextRange, SyntaxNodeRef,
    SyntaxKind::*,
    algo::{find_leaf_at_offset, find_covering_node, ancestors, Direction, siblings},
};

pub fn extend_selection(file: &File, range: TextRange) -> Option<TextRange> {
    let syntax = file.syntax();
    extend(syntax.borrowed(), range)
}

pub(crate) fn extend(root: SyntaxNodeRef, range: TextRange) -> Option<TextRange> {
    if range.is_empty() {
        let offset = range.start();
        let mut leaves = find_leaf_at_offset(root, offset);
        if let Some(leaf) = leaves.clone().find(|node| node.kind() != WHITESPACE) {
            return Some(leaf.range());
        }
        let ws = leaves.next()?;
        let ws_text = ws.leaf_text().unwrap();
        let range = TextRange::from_to(offset, ws.range().end()) - ws.range().start();
        let ws_suffix = &ws_text.as_str()[range];
        if ws_text.contains("\n") && !ws_suffix.contains("\n") {
            if let Some(node) = ws.next_sibling() {
                return Some(node.range());
            }
        }
        return Some(ws.range());
    };
    let node = find_covering_node(root, range);
    if node.kind() == COMMENT && range == node.range() {
        if let Some(range) = extend_comments(node) {
            return Some(range);
        }
    }

    match ancestors(node).skip_while(|n| n.range() == range).next() {
        None => None,
        Some(parent) => Some(parent.range()),
    }
}

fn extend_comments(node: SyntaxNodeRef) -> Option<TextRange> {
    let left = adj_comments(node, Direction::Backward);
    let right = adj_comments(node, Direction::Forward);
    if left != right {
        Some(TextRange::from_to(
            left.range().start(),
            right.range().end(),
        ))
    } else {
        None
    }
}

fn adj_comments(node: SyntaxNodeRef, dir: Direction) -> SyntaxNodeRef {
    let mut res = node;
    for node in siblings(node, dir) {
        match node.kind() {
            COMMENT => res = node,
            WHITESPACE if !node.leaf_text().unwrap().as_str().contains("\n\n") => (),
            _ => break
        }
    }
    res
}
