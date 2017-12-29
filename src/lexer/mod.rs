use {Token, SyntaxKind};
use syntax_kinds::*;

mod ptr;
use self::ptr::Ptr;

mod classes;
use self::classes::*;

pub fn next_token(text: &str) -> Token {
    assert!(!text.is_empty());
    let mut ptr = Ptr::new(text);
    let c = ptr.bump().unwrap();
    let kind = next_token_inner(c, &mut ptr);
    let len = ptr.into_len();
    Token { kind, len }
}

fn next_token_inner(c: char, ptr: &mut Ptr) -> SyntaxKind {
    // Note: r as in r" or r#" is part of a raw string literal,
    // b as in b' is part of a byte literal.
    // They are not identifiers, and are handled further down.
    let ident_start = is_ident_start(c) && !string_literal_start(c, ptr.next(), ptr.nnext());
    if ident_start {
        ptr.bump_while(is_ident_continue);
        return IDENT;
    }

    if is_whitespace(c) {
        ptr.bump_while(is_whitespace);
        return WHITESPACE;
    }

    return ERROR
}

fn string_literal_start(c: char, c1: Option<char>, c2: Option<char>) -> bool {
    match (c, c1, c2) {
        ('r', Some('"'), _) |
        ('r', Some('#'), _) |
        ('b', Some('"'), _) |
        ('b', Some('\''), _) |
        ('b', Some('r'), Some('"')) |
        ('b', Some('r'), Some('#')) => true,
        _ => false
    }
}
