extern crate file;
extern crate libsyntax2;
extern crate testutils;

use std::fmt::Write;

use libsyntax2::{tokenize, parse, Node, File};
use testutils::dir_tests;

#[test]
fn parser_tests() {
    dir_tests(
        &["parser/ok", "parser/err"],
        |text| {
            let tokens = tokenize(text);
            let file = parse(text.to_string(), &tokens);
            dump_tree(&file)
        }
    )
}

fn dump_tree(file: &File) -> String {
    let mut result = String::new();
    go(file.root(), &mut result, 0);
    return result;

    fn go(node: Node, buff: &mut String, level: usize) {
        buff.push_str(&String::from("  ").repeat(level));
        write!(buff, "{:?}\n", node).unwrap();
        let my_errors = node.errors().filter(|e| e.after_child().is_none());
        let parent_errors = node.parent().into_iter()
            .flat_map(|n| n.errors())
            .filter(|e| e.after_child() == Some(node));

        for err in my_errors {
            buff.push_str(&String::from("  ").repeat(level));
            write!(buff, "err: `{}`\n", err.message()).unwrap();
        }

        for child in node.children() {
            go(child, buff, level + 1)
        }

        for err in parent_errors {
            buff.push_str(&String::from("  ").repeat(level));
            write!(buff, "err: `{}`\n", err.message()).unwrap();
        }
    }
}
