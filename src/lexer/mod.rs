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
        return scan_ident(c, ptr);
    }

    if is_whitespace(c) {
        ptr.bump_while(is_whitespace);
        return WHITESPACE;
    }

    if is_dec_digit(c) {
        return scan_number(c, ptr);
    }

    ERROR
}

fn scan_ident(c: char, ptr: &mut Ptr) -> SyntaxKind {
    let is_single_letter = match ptr.next() {
        None => true,
        Some(c) if !is_ident_continue(c) => true,
        _ => false,
    };
    if is_single_letter {
        return if c == '_' { UNDERSCORE } else { IDENT };
    }
    ptr.bump_while(is_ident_continue);
    IDENT
}

fn scan_number(c: char, ptr: &mut Ptr) -> SyntaxKind {
    if c == '0' {
        match ptr.next().unwrap_or('\0') {
            'b' | 'o' => {
                ptr.bump();
                scan_digits(ptr, false);
            }
            'x' => {
                ptr.bump();
                scan_digits(ptr, true);
            }
            '0'...'9' | '_' | '.' | 'e' | 'E' => {
                scan_digits(ptr, true);
            }
            _ => return INT_NUMBER,
        }
    } else {
        scan_digits(ptr, false);
    }

    // might be a float, but don't be greedy if this is actually an
    // integer literal followed by field/method access or a range pattern
    // (`0..2` and `12.foo()`)
    if ptr.next_is('.') && !(ptr.nnext_is('.') || ptr.nnext_is_p(is_ident_start)) {
        // might have stuff after the ., and if it does, it needs to start
        // with a number
        ptr.bump();
        scan_digits(ptr, false);
        scan_float_exponent(ptr);
        return FLOAT_NUMBER;
    }
    // it might be a float if it has an exponent
    if ptr.next_is('e') || ptr.next_is('E') {
        scan_float_exponent(ptr);
        return FLOAT_NUMBER;
    }
    INT_NUMBER
}

fn scan_digits(ptr: &mut Ptr, allow_hex: bool) {
    while let Some(c) = ptr.next() {
        match c {
            '_' | '0'...'9' => {
                ptr.bump();
            }
            'a'...'f' | 'A' ... 'F' if allow_hex => {
                ptr.bump();
            }
            _ => return
        }
    }
}

fn scan_float_exponent(ptr: &mut Ptr) {
    if ptr.next_is('e') || ptr.next_is('E') {
        ptr.bump();
        if ptr.next_is('-') || ptr.next_is('+') {
            ptr.bump();
        }
        scan_digits(ptr, false);
    }
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
