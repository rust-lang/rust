//! This module provides a syntax highlighter for Rust code.
//! It is used by the `rustc --explain` command.
//!
//! The syntax highlighter uses `rustc_lexer`'s `tokenize`
//! function to parse the Rust code into a `Vec` of tokens.
//! The highlighter then highlights the tokens in the `Vec`,
//! and writes the highlighted output to the buffer.
use std::io::{self, Write};

use anstyle::{AnsiColor, Color, Effects, Style};
use rustc_lexer::{LiteralKind, strip_shebang, tokenize};

const PRIMITIVE_TYPES: &'static [&str] = &[
    "i8", "i16", "i32", "i64", "i128", "isize", // signed integers
    "u8", "u16", "u32", "u64", "u128", "usize", // unsigned integers
    "f32", "f64", // floating point
    "char", "bool", // others
];

const KEYWORDS: &'static [&str] = &[
    "static", "struct", "super", "trait", "true", "type", "unsafe", "use", "where", "while", "as",
    "async", "await", "break", "const", "continue", "crate", "dyn", "else", "enum", "extern",
    "false", "fn", "for", "if", "impl", "in", "let", "loop", "match", "mod", "move", "mut", "pub",
    "ref",
];

const STR_LITERAL_COLOR: AnsiColor = AnsiColor::Green;
const OTHER_LITERAL_COLOR: AnsiColor = AnsiColor::BrightRed;
const DERIVE_COLOR: AnsiColor = AnsiColor::BrightRed;
const KEYWORD_COLOR: AnsiColor = AnsiColor::BrightMagenta;
const TYPE_COLOR: AnsiColor = AnsiColor::Yellow;
const FUNCTION_COLOR: AnsiColor = AnsiColor::Blue;
const USE_COLOR: AnsiColor = AnsiColor::BrightMagenta;
const PRIMITIVE_TYPE_COLOR: AnsiColor = AnsiColor::Cyan;

/// Highlight a Rust code string and write the highlighted
/// output to the buffer. It serves as a wrapper around
/// `Highlighter::highlight_rustc_lexer`. It is passed to
/// `write_anstream_buf` in the `lib.rs` file.
pub fn highlight(code: &str, buf: &mut Vec<u8>) -> io::Result<()> {
    let mut highlighter = Highlighter::default();
    highlighter.highlight_rustc_lexer(code, buf)
}

/// A syntax highlighter for Rust code
/// It is used by the `rustc --explain` command.
#[derive(Default)]
pub struct Highlighter {
    /// Used to track if the previous token was a token
    /// that warrants the next token to be colored differently
    ///
    /// For example, the keyword `fn` requires the next token
    /// (the function name) to be colored differently.
    prev_was_special: bool,
    /// Used to track the length of tokens that have been
    /// written so far. This is used to find the original
    /// lexeme for a token from the code string.
    len_accum: usize,
}

impl Highlighter {
    /// Create a new highlighter
    pub fn new() -> Self {
        Self::default()
    }

    /// Highlight a Rust code string and write the highlighted
    /// output to the buffer.
    pub fn highlight_rustc_lexer(&mut self, code: &str, buf: &mut Vec<u8>) -> io::Result<()> {
        use rustc_lexer::TokenKind;

        // Remove shebang from code string
        let stripped_idx = strip_shebang(code).unwrap_or(0);
        let stripped_code = &code[stripped_idx..];
        self.len_accum = stripped_idx;
        let len_accum = &mut self.len_accum;
        let tokens = tokenize(stripped_code, rustc_lexer::FrontmatterAllowed::No);
        for token in tokens {
            let len = token.len as usize;
            // If the previous token was a special token, and this token is
            // not a whitespace token, then it should be colored differently
            let token_str = &code[*len_accum..*len_accum + len];
            if self.prev_was_special {
                if token_str != " " {
                    self.prev_was_special = false;
                }
                let style = Style::new().fg_color(Some(Color::Ansi(AnsiColor::Blue)));
                write!(buf, "{style}{token_str}{style:#}")?;
                *len_accum += len;
                continue;
            }
            match token.kind {
                TokenKind::Ident => {
                    let mut style = Style::new();
                    // Match if an identifier is a (well-known) keyword
                    if KEYWORDS.contains(&token_str) {
                        if token_str == "fn" {
                            self.prev_was_special = true;
                        }
                        style = style.fg_color(Some(Color::Ansi(KEYWORD_COLOR)));
                    }
                    // The `use` keyword is colored differently
                    if matches!(token_str, "use") {
                        style = style.fg_color(Some(Color::Ansi(USE_COLOR)));
                    }
                    // This heuristic test is to detect if the identifier is
                    // a function call. If it is, then the function identifier is
                    // colored differently.
                    if code[*len_accum..*len_accum + len + 1].ends_with('(') {
                        style = style.fg_color(Some(Color::Ansi(FUNCTION_COLOR)));
                    }
                    // The `derive` keyword is colored differently.
                    if token_str == "derive" {
                        style = style.fg_color(Some(Color::Ansi(DERIVE_COLOR)));
                    }
                    // This heuristic test is to detect if the identifier is
                    // a type. If it is, then the identifier is colored differently.
                    if matches!(token_str.chars().next().map(|c| c.is_uppercase()), Some(true)) {
                        style = style.fg_color(Some(Color::Ansi(TYPE_COLOR)));
                    }
                    // This if statement is to detect if the identifier is a primitive type.
                    if PRIMITIVE_TYPES.contains(&token_str) {
                        style = style.fg_color(Some(Color::Ansi(PRIMITIVE_TYPE_COLOR)));
                    }
                    write!(buf, "{style}{token_str}{style:#}")?;
                }

                // Color literals
                TokenKind::Literal { kind, suffix_start: _ } => {
                    // Strings -> Green
                    // Chars -> Green
                    // Raw strings -> Green
                    // C strings -> Green
                    // Byte Strings -> Green
                    // Other literals -> Bright Red (Orage-esque)
                    let style = match kind {
                        LiteralKind::Str { terminated: _ }
                        | LiteralKind::Char { terminated: _ }
                        | LiteralKind::RawStr { n_hashes: _ }
                        | LiteralKind::CStr { terminated: _ } => {
                            Style::new().fg_color(Some(Color::Ansi(STR_LITERAL_COLOR)))
                        }
                        _ => Style::new().fg_color(Some(Color::Ansi(OTHER_LITERAL_COLOR))),
                    };
                    write!(buf, "{style}{token_str}{style:#}")?;
                }
                _ => {
                    // All other tokens are dimmed
                    let style = Style::new()
                        .fg_color(Some(Color::Ansi(AnsiColor::BrightWhite)))
                        .effects(Effects::DIMMED);
                    write!(buf, "{style}{token_str}{style:#}")?;
                }
            }
            *len_accum += len;
        }
        Ok(())
    }
}
