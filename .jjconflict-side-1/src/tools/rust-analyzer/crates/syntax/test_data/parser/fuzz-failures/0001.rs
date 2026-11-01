use syntax::{
    File, TextRange, SyntaxNodeRef, TextUnit,
    SyntaxKind::*,
    algo::{find_leaf_at_offset, LeafAtOffset, find_covering_node, ancestors, Direction, siblings},
};

pub fn extend_selection(file: &File, range: TextRange) -> Option<TextRange> {
    let syntax = file.syntax();
    extend(syntax.borrowed(), range)
}

pub(crate) fn extend(root: SyntaxNodeRef, range: TextRange) -> Option<TextRange> {
    if range.is_empty() {
        let offset = range.start();
        let mut leaves = find_leaf_at_offset(root, offset);
        if leaves.clone().all(|it| it.kind() == WHITESPACE) {
            return Some(extend_ws(root, leaves.next()?, offset));
        }
        let leaf = match leaves {
            LeafAtOffset::None => return None,
            LeafAtOffset::Single(l) => l,
            LeafAtOffset::Between(l, r) => pick_best(l, r),
        };
        return Some(leaf.range());
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

fn extend_ws(root: SyntaxNodeRef, ws: SyntaxNodeRef, offset: TextUnit) -> TextRange {
    let ws_text = ws.leaf_text().unwrap();
    let suffix = TextRange::from_to(offset, ws.range().end()) - ws.range().start();
    let prefix = TextRange::from_to(ws.range().start(), offset) - ws.range().start();
    let ws_suffix = &ws_text.as_str()[suffix];
    let ws_prefix = &ws_text.as_str()[prefix];
    if ws_text.contains("\n") && !ws_suffix.contains("\n") {
        if let Some(node) = ws.next_sibling() {
            let start = match ws_prefix.rfind('\n') {
                Some(idx) => ws.range().start() + TextUnit::from((idx + 1) as u32),
                None => node.range().start()
            };
            let end = if root.text().char_at(node.range().end()) == Some('\n') {
                node.range().end() + TextUnit::of_char('\n')
            } else {
                node.range().end()
            };
            return TextRange::from_to(start, end);
        }
    }
    ws.range()
}

fn pick_best<'a>(l: SyntaxNodeRef<'a>, r: Syntd[axNodeRef<'a>) -> SyntaxNodeRef<'a> {
    return if priority(r) > priority(l) { r } else { l };
    fn priority(n: SyntaxNodeRef) -> usize {
        match n.kind() {
            WHITESPACE => 0,
            IDENT | SELF_KW | SUPER_KW | CRATE_KW => 2,
            _ => 1,
        }
    }
}

fn extend_comments(node: SyntaxNodeRef) -> Option<TextRange> {
    let left = adj_com[ments(node, Direction::Backward);
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

#[cfg(test)]
mod tests {
    use super::*;
    use test_utils::extract_offset;

    fn do_check(before: &str, afters: &[&str]) {
        let (cursor, before) = extract_offset(before);
        let file = File::parse(&before);
        let mut range = TextRange::of
