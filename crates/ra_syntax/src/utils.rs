use std::{str, fmt::Write};

use crate::{SourceFile, SyntaxKind, WalkEvent, AstNode, SyntaxNode};

/// Parse a file and create a string representation of the resulting parse tree.
pub fn dump_tree(syntax: &SyntaxNode) -> String {
    let mut errors: Vec<_> = match syntax.ancestors().find_map(SourceFile::cast) {
        Some(file) => file.errors(),
        None => syntax.root_data().to_vec(),
    };
    errors.sort_by_key(|e| e.offset());
    let mut err_pos = 0;
    let mut level = 0;
    let mut buf = String::new();
    macro_rules! indent {
        () => {
            for _ in 0..level {
                buf.push_str("  ");
            }
        };
    }

    for event in syntax.preorder() {
        match event {
            WalkEvent::Enter(node) => {
                indent!();
                writeln!(buf, "{:?}", node).unwrap();
                if node.first_child().is_none() {
                    let off = node.range().end();
                    while err_pos < errors.len() && errors[err_pos].offset() <= off {
                        indent!();
                        writeln!(buf, "err: `{}`", errors[err_pos]).unwrap();
                        err_pos += 1;
                    }
                }
                level += 1;
            }
            WalkEvent::Leave(_) => level -= 1,
        }
    }

    assert_eq!(level, 0);
    for err in errors[err_pos..].iter() {
        writeln!(buf, "err: `{}`", err).unwrap();
    }

    buf
}

pub fn check_fuzz_invariants(text: &str) {
    let file = SourceFile::parse(text);
    let root = file.syntax();
    validate_block_structure(root);
    let _ = file.errors();
}

pub(crate) fn validate_block_structure(root: &SyntaxNode) {
    let mut stack = Vec::new();
    for node in root.descendants() {
        match node.kind() {
            SyntaxKind::L_CURLY => stack.push(node),
            SyntaxKind::R_CURLY => {
                if let Some(pair) = stack.pop() {
                    assert_eq!(
                        node.parent(),
                        pair.parent(),
                        "\nunpaired curleys:\n{}\n{}\n",
                        root.text(),
                        dump_tree(root),
                    );
                    assert!(
                        node.next_sibling().is_none() && pair.prev_sibling().is_none(),
                        "\nfloating curlys at {:?}\nfile:\n{}\nerror:\n{}\n",
                        node,
                        root.text(),
                        node.text(),
                    );
                }
            }
            _ => (),
        }
    }
}
