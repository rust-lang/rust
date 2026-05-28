//! Simple hand-written ungrammar lexer
use crate::error::{Result, bail};

#[derive(Debug, Eq, PartialEq)]
pub(crate) enum TokenKind {
    Node(String),
    Token(String),
    Eq,
    Star,
    Pipe,
    QMark,
    Colon,
    LParen,
    RParen,
}

#[derive(Debug)]
pub(crate) struct Token {
    pub(crate) kind: TokenKind,
    pub(crate) loc: Location,
}

#[derive(Copy, Clone, Default, Debug)]
pub(crate) struct Location {
    pub(crate) line: usize,
    pub(crate) column: usize,
}

impl Location {
    fn advance(&mut self, text: &str) {
        match text.rfind('\n') {
            Some(idx) => {
                self.line += text.chars().filter(|&it| it == '\n').count();
                self.column = text[idx + 1..].chars().count();
            }
            None => self.column += text.chars().count(),
        }
    }
}

pub(crate) fn tokenize(mut input: &str) -> Result<Vec<Token>> {
    let mut res = Vec::new();
    let mut loc = Location::default();
    while !input.is_empty() {
        let old_input = input;
        skip_ws(&mut input);
        skip_comment(&mut input);
        if old_input.len() == input.len() {
            match advance(&mut input) {
                Ok(kind) => {
                    res.push(Token { kind, loc });
                }
                Err(err) => return Err(err.with_location(loc)),
            }
        }
        let consumed = old_input.len() - input.len();
        loc.advance(&old_input[..consumed]);
    }

    Ok(res)
}

fn skip_ws(input: &mut &str) {
    *input = input.trim_start_matches(is_whitespace)
}
fn skip_comment(input: &mut &str) {
    if input.starts_with("//") {
        let idx = input.find('\n').map_or(input.len(), |it| it + 1);
        *input = &input[idx..]
    }
}

fn advance(input: &mut &str) -> Result<TokenKind> {
    let mut chars = input.chars();
    let c = chars.next().unwrap();
    let res = match c {
        '=' => TokenKind::Eq,
        '*' => TokenKind::Star,
        '?' => TokenKind::QMark,
        '(' => TokenKind::LParen,
        ')' => TokenKind::RParen,
        '|' => TokenKind::Pipe,
        ':' => TokenKind::Colon,
        '\'' => {
            let mut buf = String::new();
            loop {
                match chars.next() {
                    None => bail!("unclosed token literal"),
                    Some('\\') => match chars.next() {
                        Some(c) if is_escapable(c) => buf.push(c),
                        _ => bail!("invalid escape in token literal"),
                    },
                    Some('\'') => break,
                    Some(c) => buf.push(c),
                }
            }
            TokenKind::Token(buf)
        }
        c if is_ident_char(c) => {
            let mut buf = String::new();
            buf.push(c);
            loop {
                match chars.clone().next() {
                    Some(c) if is_ident_char(c) => {
                        chars.next();
                        buf.push(c);
                    }
                    _ => break,
                }
            }
            TokenKind::Node(buf)
        }
        '\r' => bail!("unexpected `\\r`, only Unix-style line endings allowed"),
        c => bail!("unexpected character: `{}`", c),
    };

    *input = chars.as_str();
    Ok(res)
}

fn is_escapable(c: char) -> bool {
    matches!(c, '\\' | '\'')
}
fn is_whitespace(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\n')
}
fn is_ident_char(c: char) -> bool {
    matches!(c, 'a'..='z' | 'A'..='Z' | '_')
}
