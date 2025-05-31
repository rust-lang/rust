use crate::utils::{ErrAction, File, expect_action};
use core::ops::Range;
use core::slice;
use rustc_lexer::{self as lexer, FrontmatterAllowed};
use std::fs;
use std::path::{Path, PathBuf};
use walkdir::{DirEntry, WalkDir};

#[derive(Clone, Copy)]
pub enum Token<'a> {
    /// Matches any number of comments / doc comments.
    AnyComment,
    Ident(&'a str),
    CaptureIdent,
    LitStr,
    CaptureLitStr,
    Bang,
    CloseBrace,
    CloseBracket,
    CloseParen,
    /// This will consume the first colon even if the second doesn't exist.
    DoubleColon,
    Comma,
    Eq,
    Lifetime,
    Lt,
    Gt,
    OpenBrace,
    OpenBracket,
    OpenParen,
    Pound,
    Semi,
}

pub struct RustSearcher<'txt> {
    text: &'txt str,
    cursor: lexer::Cursor<'txt>,
    pos: u32,
    next_token: lexer::Token,
}
impl<'txt> RustSearcher<'txt> {
    #[must_use]
    #[expect(clippy::inconsistent_struct_constructor)]
    pub fn new(text: &'txt str) -> Self {
        let mut cursor = lexer::Cursor::new(text, FrontmatterAllowed::Yes);
        Self {
            text,
            pos: 0,
            next_token: cursor.advance_token(),
            cursor,
        }
    }

    #[must_use]
    pub fn peek_text(&self) -> &'txt str {
        &self.text[self.pos as usize..(self.pos + self.next_token.len) as usize]
    }

    #[must_use]
    pub fn peek_len(&self) -> u32 {
        self.next_token.len
    }

    #[must_use]
    pub fn peek(&self) -> lexer::TokenKind {
        self.next_token.kind
    }

    #[must_use]
    pub fn pos(&self) -> u32 {
        self.pos
    }

    #[must_use]
    pub fn at_end(&self) -> bool {
        self.next_token.kind == lexer::TokenKind::Eof
    }

    pub fn step(&mut self) {
        // `next_token.len` is zero for the eof marker.
        self.pos += self.next_token.len;
        self.next_token = self.cursor.advance_token();
    }

    /// Consumes the next token if it matches the requested value and captures the value if
    /// requested. Returns true if a token was matched.
    fn read_token(&mut self, token: Token<'_>, captures: &mut slice::IterMut<'_, &mut &'txt str>) -> bool {
        loop {
            match (token, self.next_token.kind) {
                (_, lexer::TokenKind::Whitespace)
                | (
                    Token::AnyComment,
                    lexer::TokenKind::BlockComment { terminated: true, .. } | lexer::TokenKind::LineComment { .. },
                ) => self.step(),
                (Token::AnyComment, _) => return true,
                (Token::Bang, lexer::TokenKind::Bang)
                | (Token::CloseBrace, lexer::TokenKind::CloseBrace)
                | (Token::CloseBracket, lexer::TokenKind::CloseBracket)
                | (Token::CloseParen, lexer::TokenKind::CloseParen)
                | (Token::Comma, lexer::TokenKind::Comma)
                | (Token::Eq, lexer::TokenKind::Eq)
                | (Token::Lifetime, lexer::TokenKind::Lifetime { .. })
                | (Token::Lt, lexer::TokenKind::Lt)
                | (Token::Gt, lexer::TokenKind::Gt)
                | (Token::OpenBrace, lexer::TokenKind::OpenBrace)
                | (Token::OpenBracket, lexer::TokenKind::OpenBracket)
                | (Token::OpenParen, lexer::TokenKind::OpenParen)
                | (Token::Pound, lexer::TokenKind::Pound)
                | (Token::Semi, lexer::TokenKind::Semi)
                | (
                    Token::LitStr,
                    lexer::TokenKind::Literal {
                        kind: lexer::LiteralKind::Str { terminated: true } | lexer::LiteralKind::RawStr { .. },
                        ..
                    },
                ) => {
                    self.step();
                    return true;
                },
                (Token::Ident(x), lexer::TokenKind::Ident) if x == self.peek_text() => {
                    self.step();
                    return true;
                },
                (Token::DoubleColon, lexer::TokenKind::Colon) => {
                    self.step();
                    if !self.at_end() && matches!(self.next_token.kind, lexer::TokenKind::Colon) {
                        self.step();
                        return true;
                    }
                    return false;
                },
                #[rustfmt::skip]
                (
                    Token::CaptureLitStr,
                    lexer::TokenKind::Literal {
                        kind:
                            lexer::LiteralKind::Str { terminated: true }
                            | lexer::LiteralKind::RawStr { n_hashes: Some(_) },
                        ..
                    },
                )
                | (Token::CaptureIdent, lexer::TokenKind::Ident) => {
                    **captures.next().unwrap() = self.peek_text();
                    self.step();
                    return true;
                },
                _ => return false,
            }
        }
    }

    #[must_use]
    pub fn find_token(&mut self, token: Token<'_>) -> bool {
        let mut capture = [].iter_mut();
        while !self.read_token(token, &mut capture) {
            self.step();
            if self.at_end() {
                return false;
            }
        }
        true
    }

    #[must_use]
    pub fn find_capture_token(&mut self, token: Token<'_>) -> Option<&'txt str> {
        let mut res = "";
        let mut capture = &mut res;
        let mut capture = slice::from_mut(&mut capture).iter_mut();
        while !self.read_token(token, &mut capture) {
            self.step();
            if self.at_end() {
                return None;
            }
        }
        Some(res)
    }

    #[must_use]
    pub fn match_tokens(&mut self, tokens: &[Token<'_>], captures: &mut [&mut &'txt str]) -> bool {
        let mut captures = captures.iter_mut();
        tokens.iter().all(|&t| self.read_token(t, &mut captures))
    }
}

pub struct Lint {
    pub name: String,
    pub group: String,
    pub module: String,
    pub path: PathBuf,
    pub declaration_range: Range<usize>,
}

pub struct DeprecatedLint {
    pub name: String,
    pub reason: String,
    pub version: String,
}

pub struct RenamedLint {
    pub old_name: String,
    pub new_name: String,
    pub version: String,
}

/// Finds all lint declarations (`declare_clippy_lint!`)
#[must_use]
pub fn find_lint_decls() -> Vec<Lint> {
    let mut lints = Vec::with_capacity(1000);
    let mut contents = String::new();
    for e in expect_action(fs::read_dir("."), ErrAction::Read, ".") {
        let e = expect_action(e, ErrAction::Read, ".");
        if !expect_action(e.file_type(), ErrAction::Read, ".").is_dir() {
            continue;
        }
        let Ok(mut name) = e.file_name().into_string() else {
            continue;
        };
        if name.starts_with("clippy_lints") && name != "clippy_lints_internal" {
            name.push_str("/src");
            for (file, module) in read_src_with_module(name.as_ref()) {
                parse_clippy_lint_decls(
                    file.path(),
                    File::open_read_to_cleared_string(file.path(), &mut contents),
                    &module,
                    &mut lints,
                );
            }
        }
    }
    lints.sort_by(|lhs, rhs| lhs.name.cmp(&rhs.name));
    lints
}

/// Reads the source files from the given root directory
fn read_src_with_module(src_root: &Path) -> impl use<'_> + Iterator<Item = (DirEntry, String)> {
    WalkDir::new(src_root).into_iter().filter_map(move |e| {
        let e = expect_action(e, ErrAction::Read, src_root);
        let path = e.path().as_os_str().as_encoded_bytes();
        if let Some(path) = path.strip_suffix(b".rs")
            && let Some(path) = path.get(src_root.as_os_str().len() + 1..)
        {
            if path == b"lib" {
                Some((e, String::new()))
            } else {
                let path = if let Some(path) = path.strip_suffix(b"mod")
                    && let Some(path) = path.strip_suffix(b"/").or_else(|| path.strip_suffix(b"\\"))
                {
                    path
                } else {
                    path
                };
                if let Ok(path) = str::from_utf8(path) {
                    let path = path.replace(['/', '\\'], "::");
                    Some((e, path))
                } else {
                    None
                }
            }
        } else {
            None
        }
    })
}

/// Parse a source file looking for `declare_clippy_lint` macro invocations.
fn parse_clippy_lint_decls(path: &Path, contents: &str, module: &str, lints: &mut Vec<Lint>) {
    #[allow(clippy::enum_glob_use)]
    use Token::*;
    #[rustfmt::skip]
    static DECL_TOKENS: &[Token<'_>] = &[
        // !{ /// docs
        Bang, OpenBrace, AnyComment,
        // #[clippy::version = "version"]
        Pound, OpenBracket, Ident("clippy"), DoubleColon, Ident("version"), Eq, LitStr, CloseBracket,
        // pub NAME, GROUP,
        Ident("pub"), CaptureIdent, Comma, AnyComment, CaptureIdent, Comma,
    ];

    let mut searcher = RustSearcher::new(contents);
    while searcher.find_token(Ident("declare_clippy_lint")) {
        let start = searcher.pos() as usize - "declare_clippy_lint".len();
        let (mut name, mut group) = ("", "");
        if searcher.match_tokens(DECL_TOKENS, &mut [&mut name, &mut group]) && searcher.find_token(CloseBrace) {
            lints.push(Lint {
                name: name.to_lowercase(),
                group: group.into(),
                module: module.into(),
                path: path.into(),
                declaration_range: start..searcher.pos() as usize,
            });
        }
    }
}

#[must_use]
pub fn read_deprecated_lints() -> (Vec<DeprecatedLint>, Vec<RenamedLint>) {
    #[allow(clippy::enum_glob_use)]
    use Token::*;
    #[rustfmt::skip]
    static DECL_TOKENS: &[Token<'_>] = &[
        // #[clippy::version = "version"]
        Pound, OpenBracket, Ident("clippy"), DoubleColon, Ident("version"), Eq, CaptureLitStr, CloseBracket,
        // ("first", "second"),
        OpenParen, CaptureLitStr, Comma, CaptureLitStr, CloseParen, Comma,
    ];
    #[rustfmt::skip]
    static DEPRECATED_TOKENS: &[Token<'_>] = &[
        // !{ DEPRECATED(DEPRECATED_VERSION) = [
        Bang, OpenBrace, Ident("DEPRECATED"), OpenParen, Ident("DEPRECATED_VERSION"), CloseParen, Eq, OpenBracket,
    ];
    #[rustfmt::skip]
    static RENAMED_TOKENS: &[Token<'_>] = &[
        // !{ RENAMED(RENAMED_VERSION) = [
        Bang, OpenBrace, Ident("RENAMED"), OpenParen, Ident("RENAMED_VERSION"), CloseParen, Eq, OpenBracket,
    ];

    let path = "clippy_lints/src/deprecated_lints.rs";
    let mut deprecated = Vec::with_capacity(30);
    let mut renamed = Vec::with_capacity(80);
    let mut contents = String::new();
    File::open_read_to_cleared_string(path, &mut contents);

    let mut searcher = RustSearcher::new(&contents);

    // First instance is the macro definition.
    assert!(
        searcher.find_token(Ident("declare_with_version")),
        "error reading deprecated lints"
    );

    if searcher.find_token(Ident("declare_with_version")) && searcher.match_tokens(DEPRECATED_TOKENS, &mut []) {
        let mut version = "";
        let mut name = "";
        let mut reason = "";
        while searcher.match_tokens(DECL_TOKENS, &mut [&mut version, &mut name, &mut reason]) {
            deprecated.push(DeprecatedLint {
                name: parse_str_single_line(path.as_ref(), name),
                reason: parse_str_single_line(path.as_ref(), reason),
                version: parse_str_single_line(path.as_ref(), version),
            });
        }
    } else {
        panic!("error reading deprecated lints");
    }

    if searcher.find_token(Ident("declare_with_version")) && searcher.match_tokens(RENAMED_TOKENS, &mut []) {
        let mut version = "";
        let mut old_name = "";
        let mut new_name = "";
        while searcher.match_tokens(DECL_TOKENS, &mut [&mut version, &mut old_name, &mut new_name]) {
            renamed.push(RenamedLint {
                old_name: parse_str_single_line(path.as_ref(), old_name),
                new_name: parse_str_single_line(path.as_ref(), new_name),
                version: parse_str_single_line(path.as_ref(), version),
            });
        }
    } else {
        panic!("error reading renamed lints");
    }

    deprecated.sort_by(|lhs, rhs| lhs.name.cmp(&rhs.name));
    renamed.sort_by(|lhs, rhs| lhs.old_name.cmp(&rhs.old_name));
    (deprecated, renamed)
}

/// Removes the line splices and surrounding quotes from a string literal
fn parse_str_lit(s: &str) -> String {
    let (s, is_raw) = if let Some(s) = s.strip_prefix("r") {
        (s.trim_matches('#'), true)
    } else {
        (s, false)
    };
    let s = s
        .strip_prefix('"')
        .and_then(|s| s.strip_suffix('"'))
        .unwrap_or_else(|| panic!("expected quoted string, found `{s}`"));

    if is_raw {
        s.into()
    } else {
        let mut res = String::with_capacity(s.len());
        rustc_literal_escaper::unescape_str(s, &mut |_, ch| {
            if let Ok(ch) = ch {
                res.push(ch);
            }
        });
        res
    }
}

fn parse_str_single_line(path: &Path, s: &str) -> String {
    let value = parse_str_lit(s);
    assert!(
        !value.contains('\n'),
        "error parsing `{}`: `{s}` should be a single line string",
        path.display(),
    );
    value
}
