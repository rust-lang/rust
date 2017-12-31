extern crate file;
#[macro_use(assert_diff)]
extern crate difference;
extern crate libsyntax2;

use std::path::{PathBuf, Path};
use std::fs::read_dir;
use std::fmt::Write;

use libsyntax2::{tokenize, Token, Node, File, FileBuilder};

#[test]
fn parser_tests() {
    for test_case in parser_test_cases() {
        parser_test_case(&test_case);
    }
}

fn parser_test_dir() -> PathBuf {
    let dir = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(dir).join("tests/data/parser")
}

fn parser_test_cases() -> Vec<PathBuf> {
    let mut acc = Vec::new();
    let dir = parser_test_dir();
    for file in read_dir(&dir).unwrap() {
        let file = file.unwrap();
        let path = file.path();
        if path.extension().unwrap_or_default() == "rs" {
            acc.push(path);
        }
    }
    acc.sort();
    acc
}

fn parser_test_case(path: &Path) {
    let actual = {
        let text = file::get_text(path).unwrap();
        let tokens = tokenize(&text);
        let file = parse(text, &tokens);
        dump_tree(&file)
    };
    let expected = file::get_text(&path.with_extension("txt")).unwrap();
    let expected = expected.as_str();
    let actual = actual.as_str();
    if expected == actual {
        return
    }
    if expected.trim() == actual.trim() {
        panic!("Whitespace difference!")
    }
    assert_diff!(expected, actual, "\n", 0)
}

fn dump_tree(file: &File) -> String {
    let mut result = String::new();
    go(file.root(), &mut result, 0);
    return result;

    fn go(node: Node, buff: &mut String, level: usize) {
        buff.push_str(&String::from("  ").repeat(level));
        write!(buff, "{:?}\n", node);
        for child in node.children() {
            go(child, buff, level + 1)
        }
    }
}

fn parse(text: String, tokens: &[Token]) -> File {
    let mut builder = FileBuilder::new(text);
    builder.start_internal(libsyntax2::syntax_kinds::FILE);
    builder.finish_internal();
    builder.finish()
}