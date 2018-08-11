use libsyntax2::{
    ast, AstNode,
    TextRange, SyntaxNodeRef,
    SyntaxKind::WHITESPACE,
    algo::{find_leaf_at_offset, find_covering_node, ancestors},
};

pub fn extend_selection(file: &ast::File, range: TextRange) -> Option<TextRange> {
    let syntax = file.syntax();
    extend(syntax.as_ref(), range)
}

pub(crate) fn extend(root: SyntaxNodeRef, range: TextRange) -> Option<TextRange> {
    if range.is_empty() {
        let offset = range.start();
        let mut leaves = find_leaf_at_offset(root, offset);
        if let Some(leaf) = leaves.clone().find(|node| node.kind() != WHITESPACE) {
            return Some(leaf.range());
        }
        let ws = leaves.next()?;
//        let ws_suffix = file.text().slice(
//            TextRange::from_to(offset, ws.range().end())
//        );
//        if ws.text().contains("\n") && !ws_suffix.contains("\n") {
//            if let Some(line_end) = file.text()
//                .slice(TextSuffix::from(ws.range().end()))
//                .find("\n")
//            {
//                let range = TextRange::from_len(ws.range().end(), line_end);
//                return Some(find_covering_node(file.root(), range).range());
//            }
//        }
        return Some(ws.range());
    };
    let node = find_covering_node(root, range);

    match ancestors(node).skip_while(|n| n.range() == range).next() {
        None => None,
        Some(parent) => Some(parent.range()),
    }
}
