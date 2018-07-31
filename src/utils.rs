use std::fmt::Write;
use {
    algo::walk::{walk, WalkEvent},
    SyntaxNode,
};

/// Parse a file and create a string representation of the resulting parse tree.
pub fn dump_tree(syntax: &SyntaxNode) -> String {
    let syntax = syntax.as_ref();
    let mut errors: Vec<_> = syntax.root.errors.iter().cloned().collect();
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
                        writeln!(buf, "err: `{}`", errors[err_pos].message).unwrap();
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
        writeln!(buf, "err: `{}`", err.message).unwrap();
    }

    return buf;
}
