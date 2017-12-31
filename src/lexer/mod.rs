use {Token, SyntaxKind};
use syntax_kinds::*;

mod ptr;
use self::ptr::Ptr;

mod classes;
use self::classes::*;

mod numbers;
use self::numbers::scan_number;

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
        let kind = scan_number(c, ptr);
        scan_literal_suffix(ptr);
        return kind;
    }

    // One-byte tokens.
    match c {
        ';' => return SEMI,
        ',' => return COMMA,
        '(' => return L_PAREN,
        ')' => return R_PAREN,
        '{' => return L_CURLY,
        '}' => return R_CURLY,
        '[' => return L_BRACK,
        ']' => return R_BRACK,
        '<' => return L_ANGLE,
        '>' => return R_ANGLE,
        '@' => return AT,
        '#' => return POUND,
        '~' => return TILDE,
        '?' => return QUESTION,
        '$' => return DOLLAR,

        // Multi-byte tokens.
        '.' => return match (ptr.next(), ptr.nnext()) {
            (Some('.'), Some('.')) => {
                ptr.bump();
                ptr.bump();
                DOTDOTDOT
            },
            (Some('.'), Some('=')) => {
                ptr.bump();
                ptr.bump();
                DOTDOTEQ
            },
            (Some('.'), _) => {
                ptr.bump();
                DOTDOT
            },
            _ => DOT
        },
        ':' => return match ptr.next() {
            Some(':') => {
                ptr.bump();
                COLONCOLON
            }
            _ => COLON
        },
        '=' => return match ptr.next() {
            Some('=') => {
                ptr.bump();
                EQEQ
            }
            Some('>') => {
                ptr.bump();
                FAT_ARROW
            }
            _ => EQ,
        },
        '!' => return match ptr.next() {
            Some('=') => {
                ptr.bump();
                NEQ
            }
            _ => NOT,
        },

        // '\'' => scan_char_or_lifetime(ptr),
        _ => (),
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

fn scan_char_or_lifetime(ptr: &mut Ptr) -> SyntaxKind {
    // Either a character constant 'a' OR a lifetime name 'abc
    let c = match ptr.bump() {
        Some(c) => c,
        None => return CHAR, // TODO: error reporting is upper in the stack
    };

    // If the character is an ident start not followed by another single
    // quote, then this is a lifetime name:
    if is_ident_start(c) && !ptr.next_is('\'') {
        while ptr.next_is_p(is_ident_continue) {
            ptr.bump();
        }

        // lifetimes shouldn't end with a single quote
        // if we find one, then this is an invalid character literal
        if ptr.next_is('\'') {
            ptr.bump();
            return CHAR;
        }
        return LIFETIME;
    }
    scan_char_or_byte(ptr);
    if !ptr.next_is('\'') {
        return CHAR; // TODO: error reporting
    }
    ptr.bump();
    scan_literal_suffix(ptr);
    CHAR
}

fn scan_literal_suffix(ptr: &mut Ptr) {

}

fn scan_char_or_byte(ptr: &mut Ptr) {
    ptr.bump();
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
