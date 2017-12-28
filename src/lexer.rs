use {Token, TextUnit};
use syntax_kinds::*;

pub fn next_token(text: &str) -> Token {
    let c = text.chars().next().unwrap();
    Token {
        kind: IDENT,
        len: TextUnit::len_of_char(c),
    }
}