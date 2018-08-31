use libsyntax2::{
    File, TextRange, SyntaxNodeRef, TextUnit,
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
                return Some(TextRange::from_to(start, end));
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

#[cfg(test)]
mod tests {
    use super::*;
    use test_utils::extract_offset;

    fn do_check(before: &str, afters: &[&str]) {
        let (cursor, before) = extract_offset(before);
        let file = File::parse(&before);
        let mut range = TextRange::offset_len(cursor, 0.into());
        for &after in afters {
            range = extend_selection(&file, range)
                .unwrap();
            let actual = &before[range];
            assert_eq!(after, actual);
        }
    }

    #[test]
    fn test_extend_selection_arith() {
        do_check(
            r#"fn foo() { <|>1 + 1 }"#,
            &["1", "1 + 1", "{ 1 + 1 }"],
        );
    }

    #[test]
    fn test_extend_selection_start_of_the_lind() {
        do_check(
            r#"
impl S {
<|>    fn foo() {

    }
}"#,
            &["    fn foo() {\n\n    }\n"]
        );
    }

    #[test]
    fn test_extend_selection_comments() {
        do_check(
            r#"
fn bar(){}

// fn foo() {
// 1 + <|>1
// }

// fn foo(){}
    "#,
            &["// 1 + 1", "// fn foo() {\n// 1 + 1\n// }"]
        );
    }
}
