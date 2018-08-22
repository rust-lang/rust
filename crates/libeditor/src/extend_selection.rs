use libsyntax2::{
    ParsedFile, TextRange, SyntaxNodeRef,
    SyntaxKind::WHITESPACE,
    algo::{find_leaf_at_offset, find_covering_node, ancestors},
};

pub fn extend_selection(file: &ParsedFile, range: TextRange) -> Option<TextRange> {
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

    match ancestors(node).skip_while(|n| n.range() == range).next() {
        None => None,
        Some(parent) => Some(parent.range()),
    }
}
