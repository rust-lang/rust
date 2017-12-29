use unicode_xid::UnicodeXID;

use {Token, SyntaxKind};
use syntax_kinds::*;

mod ptr;
use self::ptr::Ptr;

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
    let ident_start = ident_start(c) && !string_literal_start(c, ptr.next(), ptr.nnext());
    if ident_start {
        loop {
            match ptr.next() {
                Some(c) if ident_continue(c) => {
                    ptr.bump();
                },
                _ => break,
            }
        }
        IDENT
    } else {
        WHITESPACE
    }
}

fn ident_start(c: char) -> bool {
    (c >= 'a' && c <= 'z')
        || (c >= 'A' && c <= 'Z')
        || c == '_'
        || (c > '\x7f' && UnicodeXID::is_xid_start(c))
}

fn ident_continue(c: char) -> bool {
    (c >= 'a' && c <= 'z')
        || (c >= 'A' && c <= 'Z')
        || (c >= '0' && c <= '9')
        || c == '_'
        || (c > '\x7f' && UnicodeXID::is_xid_continue(c))
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
