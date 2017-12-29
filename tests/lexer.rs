extern crate file;
#[macro_use(assert_diff)]
extern crate difference;
extern crate libsyntax2;

use std::path::{PathBuf, Path};
use std::fs::read_dir;
use std::fmt::Write;

use libsyntax2::{Token, next_token};

#[test]
fn lexer_tests() {
    for test_case in lexer_test_cases() {
        lexer_test_case(&test_case);
    }
}

fn lexer_test_dir() -> PathBuf {
    let dir = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(dir).join("tests/data/lexer")
}

fn lexer_test_cases() -> Vec<PathBuf> {
    let mut acc = Vec::new();
    let dir = lexer_test_dir();
    for file in read_dir(&dir).unwrap() {
        let file = file.unwrap();
        let path = file.path();
        if path.extension().unwrap_or_default() == "rs" {
            acc.push(path);
        }
    }
    acc
}

fn lexer_test_case(path: &Path) {
    let actual = {
        let text = file::get_text(path).unwrap();
        let tokens = tokenize(&text);
        dump_tokens(&tokens)
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

fn tokenize(text: &str) -> Vec<Token> {
    let mut text = text;
    let mut acc = Vec::new();
    while !text.is_empty() {
        let token = next_token(text);
        acc.push(token);
        let len: u32 = token.len.into();
        text = &text[len as usize..];
    }
    acc
}

fn dump_tokens(tokens: &[Token]) -> String {
    let mut acc = String::new();
    for token in tokens {
        write!(acc, "{:?} {}\n", token.kind, token.len).unwrap()
    }
    acc
}