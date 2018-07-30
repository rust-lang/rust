use std::{collections::BTreeSet, fmt::Write};
use {SyntaxError, SyntaxNode, SyntaxNodeRef};

/// Parse a file and create a string representation of the resulting parse tree.
pub fn dump_tree(syntax: &SyntaxNode) -> String {
    let syntax = syntax.as_ref();
    let mut errors: BTreeSet<_> = syntax.root.errors.iter().cloned().collect();
    let mut result = String::new();
    go(syntax, &mut result, 0, &mut errors);
    return result;

    fn go(
        node: SyntaxNodeRef,
        buff: &mut String,
        level: usize,
        errors: &mut BTreeSet<SyntaxError>,
    ) {
        buff.push_str(&String::from("  ").repeat(level));
        write!(buff, "{:?}\n", node).unwrap();
        let my_errors: Vec<_> = errors
            .iter()
            .filter(|e| e.offset == node.range().start())
            .cloned()
            .collect();
        for err in my_errors {
            errors.remove(&err);
            buff.push_str(&String::from("  ").repeat(level));
            write!(buff, "err: `{}`\n", err.message).unwrap();
        }

        for child in node.children() {
            go(child, buff, level + 1, errors)
        }

        let my_errors: Vec<_> = errors
            .iter()
            .filter(|e| e.offset == node.range().end())
            .cloned()
            .collect();
        for err in my_errors {
            errors.remove(&err);
            buff.push_str(&String::from("  ").repeat(level));
            write!(buff, "err: `{}`\n", err.message).unwrap();
        }
    }
}
