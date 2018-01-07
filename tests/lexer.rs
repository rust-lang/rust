extern crate file;
extern crate libsyntax2;
extern crate testutils;

use std::path::{Path};
use std::fmt::Write;

use libsyntax2::{Token, tokenize};
use testutils::{assert_equal_text, collect_tests};

#[test]
fn lexer_tests() {
    for test_case in collect_tests(&["lexer"]) {
        lexer_test_case(&test_case);
    }
}

fn lexer_test_case(path: &Path) {
    let actual = {
        let text = file::get_text(path).unwrap();
        let tokens = tokenize(&text);
        dump_tokens(&tokens, &text)
    };
    let path = path.with_extension("txt");
    let expected = file::get_text(&path).unwrap();
    let expected = expected.as_str();
    let actual = actual.as_str();
    assert_equal_text(expected, actual, &path)
}

fn dump_tokens(tokens: &[Token], text: &str) -> String {
    let mut acc = String::new();
    let mut offset = 0;
    for token in tokens {
        let len: u32 = token.len.into();
        let len = len as usize;
        let token_text = &text[offset..offset + len];
        offset += len;
        write!(acc, "{:?} {} {:?}\n", token.kind, token.len, token_text).unwrap()
    }
    acc
}