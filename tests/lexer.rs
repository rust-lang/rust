extern crate libsyntax2;
extern crate testutils;

use std::fmt::Write;

use libsyntax2::{tokenize, Token};
use testutils::dir_tests;

#[test]
fn lexer_tests() {
    dir_tests(&["lexer"], |text| {
        let tokens = tokenize(text);
        dump_tokens(&tokens, text)
    })
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
