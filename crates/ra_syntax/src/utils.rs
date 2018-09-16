use std::fmt::Write;
use {
    algo::walk::{preorder, walk, WalkEvent},
    SyntaxKind, File, SyntaxNodeRef, TreeRoot,
};

/// Parse a file and create a string representation of the resulting parse tree.
pub fn dump_tree(syntax: SyntaxNodeRef) -> String {
    let mut errors: Vec<_> = syntax.root.syntax_root().errors.iter().cloned().collect();
    errors.sort_by_key(|e| e.offset);
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

    for event in walk(syntax) {
        match event {
            WalkEvent::Enter(node) => {
                indent!();
                writeln!(buf, "{:?}", node).unwrap();
                if node.first_child().is_none() {
                    let off = node.range().end();
                    while err_pos < errors.len() && errors[err_pos].offset <= off {
                        indent!();
                        writeln!(buf, "err: `{}`", errors[err_pos].msg).unwrap();
                        err_pos += 1;
                    }
                }
                level += 1;
            }
            WalkEvent::Exit(_) => level -= 1,
        }
    }

    assert_eq!(level, 0);
    for err in errors[err_pos..].iter() {
        writeln!(buf, "err: `{}`", err.msg).unwrap();
    }

    return buf;
}

pub fn check_fuzz_invariants(text: &str) {
    let file = File::parse(text);
    let root = file.syntax();
    validate_block_structure(root);
    let _ = file.ast();
    let _ = file.errors();
}

pub(crate) fn validate_block_structure(root: SyntaxNodeRef) {
    let mut stack = Vec::new();
    for node in preorder(root) {
        match node.kind() {
            SyntaxKind::L_CURLY => {
                stack.push(node)
            }
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
