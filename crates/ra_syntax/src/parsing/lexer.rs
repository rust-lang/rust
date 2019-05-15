mod classes;
mod comments;
mod numbers;
mod ptr;
mod strings;

use crate::{
    SyntaxKind::{self, *},
    TextUnit,
    T,
};

use self::{
    classes::*,
    comments::{scan_comment, scan_shebang},
    numbers::scan_number,
    ptr::Ptr,
    strings::{
        is_string_literal_start, scan_byte_char_or_string, scan_char, scan_raw_string, scan_string,
    },
};

/// A token of Rust source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Token {
    /// The kind of token.
    pub kind: SyntaxKind,
    /// The length of the token.
    pub len: TextUnit,
}

/// Break a string up into its component tokens
pub fn tokenize(text: &str) -> Vec<Token> {
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

/// Get the next token from a string
pub fn next_token(text: &str) -> Token {
    assert!(!text.is_empty());
    let mut ptr = Ptr::new(text);
    let c = ptr.bump().unwrap();
    let kind = next_token_inner(c, &mut ptr);
    let len = ptr.into_len();
    Token { kind, len }
}

fn next_token_inner(c: char, ptr: &mut Ptr) -> SyntaxKind {
    if is_whitespace(c) {
        ptr.bump_while(is_whitespace);
        return WHITESPACE;
    }

    match c {
        '#' => {
            if scan_shebang(ptr) {
                return SHEBANG;
            }
        }
        '/' => {
            if let Some(kind) = scan_comment(ptr) {
                return kind;
            }
        }
        _ => (),
    }

    let ident_start = is_ident_start(c) && !is_string_literal_start(c, ptr.current(), ptr.nth(1));
    if ident_start {
        return scan_ident(c, ptr);
    }

    if is_dec_digit(c) {
        let kind = scan_number(c, ptr);
        scan_literal_suffix(ptr);
        return kind;
    }

    // One-byte tokens.
    if let Some(kind) = SyntaxKind::from_char(c) {
        return kind;
    }

    match c {
        // Possiblily multi-byte tokens,
        // but we only produce single byte token now
        // T![...], T![..], T![..=], T![.]
        '.' => return T![.],
        // T![::] T![:]
        ':' => return T![:],
        // T![==] FATARROW T![=]
        '=' => return T![=],
        // T![!=] T![!]
        '!' => return T![!],
        // T![->] T![-]
        '-' => return T![-],

        // If the character is an ident start not followed by another single
        // quote, then this is a lifetime name:
        '\'' => {
            return if ptr.at_p(is_ident_start) && !ptr.at_str("''") {
                ptr.bump();
                while ptr.at_p(is_ident_continue) {
                    ptr.bump();
                }
                // lifetimes shouldn't end with a single quote
                // if we find one, then this is an invalid character literal
                if ptr.at('\'') {
                    ptr.bump();
                    return CHAR;
                }
                LIFETIME
            } else {
                scan_char(ptr);
                scan_literal_suffix(ptr);
                CHAR
            };
        }
        'b' => {
            let kind = scan_byte_char_or_string(ptr);
            scan_literal_suffix(ptr);
            return kind;
        }
        '"' => {
            scan_string(ptr);
            scan_literal_suffix(ptr);
            return STRING;
        }
        'r' => {
            scan_raw_string(ptr);
            scan_literal_suffix(ptr);
            return RAW_STRING;
        }
        _ => (),
    }
    ERROR
}

fn scan_ident(c: char, ptr: &mut Ptr) -> SyntaxKind {
    let is_raw = match (c, ptr.current()) {
        ('r', Some('#')) => {
            ptr.bump();
            true
        }
        ('_', None) => return T![_],
        ('_', Some(c)) if !is_ident_continue(c) => return T![_],
        _ => false,
    };
    ptr.bump_while(is_ident_continue);
    if !is_raw {
        if let Some(kind) = SyntaxKind::from_keyword(ptr.current_token_text()) {
            return kind;
        }
    }
    IDENT
}

fn scan_literal_suffix(ptr: &mut Ptr) {
    if ptr.at_p(is_ident_start) {
        ptr.bump();
    }
    ptr.bump_while(is_ident_continue);
}

pub fn classify_literal(text: &str) -> Option<Token> {
    let tkn = next_token(text);
    if !tkn.kind.is_literal() || tkn.len.to_usize() != text.len() {
        return None;
    }

    Some(tkn)
}
