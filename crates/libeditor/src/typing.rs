use libsyntax2::{
    TextUnit, TextRange, SyntaxNodeRef,
    ast,
    algo::{
        walk::preorder,
        find_covering_node,
    },
};

use {ActionResult, EditBuilder};

pub fn join_lines(file: &ast::ParsedFile, range: TextRange) -> ActionResult {
    let range = if range.is_empty() {
        let text = file.syntax().text();
        let text = &text[TextRange::from_to(range.start(), TextUnit::of_str(&text))];
        let pos = text.bytes().take_while(|&b| b != b'\n').count();
        if pos == text.len() {
            return ActionResult {
                edit: EditBuilder::new().finish(),
                cursor_position: None
            };
        }
        let pos: TextUnit = (pos as u32).into();
        TextRange::offset_len(
            range.start() + pos,
            TextUnit::of_char('\n'),
        )
    } else {
        range
    };
    let node = find_covering_node(file.syntax(), range);
    let mut edit = EditBuilder::new();
    for node in preorder(node) {
        let text = match node.leaf_text() {
            Some(text) => text,
            None => continue,
        };
        let range = match intersect(range, node.range()) {
            Some(range) => range,
            None => continue,
        } - node.range().start();
        for (pos, _) in text[range].bytes().enumerate().filter(|&(_, b)| b == b'\n') {
            let pos: TextUnit = (pos as u32).into();
            let off = node.range().start() + range.start() + pos;
            remove_newline(&mut edit, node, text.as_str(), off);
        }
    }

    ActionResult {
        edit: edit.finish(),
        cursor_position: None,
    }
}

fn intersect(r1: TextRange, r2: TextRange) -> Option<TextRange> {
    let start = r1.start().max(r2.start());
    let end = r1.end().min(r2.end());
    if start <= end {
        Some(TextRange::from_to(start, end))
    } else {
        None
    }
}

fn remove_newline(
    edit: &mut EditBuilder,
    node: SyntaxNodeRef,
    node_text: &str,
    offset: TextUnit,
) {
    let suff = &node_text[TextRange::from_to(
        offset - node.range().start() + TextUnit::of_char('\n'),
        TextUnit::of_str(node_text),
    )];
    let spaces = suff.bytes().take_while(|&b| b == b' ').count();

    edit.replace(
        TextRange::offset_len(offset, ((spaces + 1) as u32).into()),
        " ".to_string(),
    );
}
