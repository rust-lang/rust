use {Token, SyntaxKind};
use syntax_kinds::*;

mod ptr;
use self::ptr::Ptr;

mod classes;
use self::classes::*;

mod numbers;
use self::numbers::scan_number;

mod strings;
use self::strings::{is_string_literal_start, scan_char, scan_byte_char_or_string, scan_string, scan_raw_string};

mod comments;
use self::comments::{scan_shebang, scan_comment};

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
        '#' => if scan_shebang(ptr) {
            return SHEBANG;
        }
        '/' => if let Some(kind) = scan_comment(ptr) {
            return kind;
        }
        _ => (),
    }

    let ident_start = is_ident_start(c) && !is_string_literal_start(c, ptr.next(), ptr.nnext());
    if ident_start {
        return scan_ident(c, ptr);
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
        '&' => return AMPERSAND,
        '|' => return PIPE,
        '+' => return PLUS,
        '*' => return STAR,
        '/' => return SLASH,
        '^' => return CARET,
        '%' => return PERCENT,

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
        '-' => return if ptr.next_is('>') {
            ptr.bump();
            THIN_ARROW
        } else {
            MINUS
        },

        // If the character is an ident start not followed by another single
        // quote, then this is a lifetime name:
        '\'' => return if ptr.next_is_p(is_ident_start) && !ptr.nnext_is('\'') {
            ptr.bump();
            while ptr.next_is_p(is_ident_continue) {
                ptr.bump();
            }
            // lifetimes shouldn't end with a single quote
            // if we find one, then this is an invalid character literal
            if ptr.next_is('\'') {
                ptr.bump();
                return CHAR; // TODO: error reporting
            }
            LIFETIME
        } else {
            scan_char(ptr);
            scan_literal_suffix(ptr);
            CHAR
        },
        'b' => {
            let kind = scan_byte_char_or_string(ptr);
            scan_literal_suffix(ptr);
            return kind
        },
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

fn scan_literal_suffix(ptr: &mut Ptr) {
    if ptr.next_is_p(is_ident_start) {
        ptr.bump();
    }
    ptr.bump_while(is_ident_continue);
}
