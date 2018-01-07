extern crate file;
extern crate libsyntax2;
extern crate testutils;

use std::path::{Path};
use std::fmt::Write;

use libsyntax2::{tokenize, parse, Node, File};
use testutils::{collect_tests, assert_equal_text};

#[test]
fn parser_tests() {
    for test_case in collect_tests(&["parser/ok", "parser/err"]) {
        parser_test_case(&test_case);
    }
}

fn parser_test_case(path: &Path) {
    let actual = {
        let text = file::get_text(path).unwrap();
        let tokens = tokenize(&text);
        let file = parse(text, &tokens);
        dump_tree(&file)
    };
    let expected_path = path.with_extension("txt");
    let expected = file::get_text(&expected_path).expect(
        &format!("Can't read {}", expected_path.display())
    );
    let expected = expected.as_str();
    let actual = actual.as_str();
    assert_equal_text(expected, actual, &expected_path);
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
