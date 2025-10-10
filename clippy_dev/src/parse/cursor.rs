use core::slice;
use rustc_lexer::{self as lex, LiteralKind, Token, TokenKind};

/// A token pattern used for searching and matching by the [`Cursor`].
///
/// In the event that a pattern is a multi-token sequence, earlier tokens will be consumed
/// even if the pattern ultimately isn't matched. e.g. With the sequence `:*` matching
/// `DoubleColon` will consume the first `:` and then fail to match, leaving the cursor at
/// the `*`.
#[derive(Clone, Copy)]
pub enum Pat<'a> {
    /// Matches any number of comments and doc comments.
    AnyComment,
    Ident(&'a str),
    CaptureIdent,
    LitStr,
    CaptureLitStr,
    Bang,
    CloseBrace,
    CloseBracket,
    CloseParen,
    Comma,
    DoubleColon,
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

/// A unidirectional cursor over a token stream that is lexed on demand.
pub struct Cursor<'txt> {
    next_token: Token,
    pos: u32,
    inner: lex::Cursor<'txt>,
    text: &'txt str,
}
impl<'txt> Cursor<'txt> {
    #[must_use]
    pub fn new(text: &'txt str) -> Self {
        let mut inner = lex::Cursor::new(text, lex::FrontmatterAllowed::Yes);
        Self {
            next_token: inner.advance_token(),
            pos: 0,
            inner,
            text,
        }
    }

    /// Gets the text that makes up the next token in the stream, or the empty string if
    /// stream is exhausted.
    #[must_use]
    pub fn peek_text(&self) -> &'txt str {
        &self.text[self.pos as usize..(self.pos + self.next_token.len) as usize]
    }

    /// Gets the length of the next token in bytes, or zero if the stream is exhausted.
    #[must_use]
    pub fn peek_len(&self) -> u32 {
        self.next_token.len
    }

    /// Gets the next token in the stream, or [`TokenKind::Eof`] if the stream is
    /// exhausted.
    #[must_use]
    pub fn peek(&self) -> TokenKind {
        self.next_token.kind
    }

    /// Gets the offset of the next token in the source string, or the string's length if
    /// the stream is exhausted.
    #[must_use]
    pub fn pos(&self) -> u32 {
        self.pos
    }

    /// Gets whether the cursor has exhausted its input.
    #[must_use]
    pub fn at_end(&self) -> bool {
        self.next_token.kind == TokenKind::Eof
    }

    /// Advances the cursor to the next token. If the stream is exhausted this will set
    /// the next token to [`TokenKind::Eof`].
    pub fn step(&mut self) {
        // `next_token.len` is zero for the eof marker.
        self.pos += self.next_token.len;
        self.next_token = self.inner.advance_token();
    }

    /// Consumes tokens until the given pattern is either fully matched of fails to match.
    /// Returns whether the pattern was fully matched.
    ///
    /// For each capture made by the pattern one item will be taken from the capture
    /// sequence with the result placed inside.
    fn match_pat(&mut self, pat: Pat<'_>, captures: &mut slice::IterMut<'_, &mut &'txt str>) -> bool {
        loop {
            match (pat, self.next_token.kind) {
                #[rustfmt::skip] // rustfmt bug: https://github.com/rust-lang/rustfmt/issues/6697
                (_, TokenKind::Whitespace)
                | (
                    Pat::AnyComment,
                    TokenKind::BlockComment { terminated: true, .. } | TokenKind::LineComment { .. },
                ) => self.step(),
                (Pat::AnyComment, _) => return true,
                (Pat::Bang, TokenKind::Bang)
                | (Pat::CloseBrace, TokenKind::CloseBrace)
                | (Pat::CloseBracket, TokenKind::CloseBracket)
                | (Pat::CloseParen, TokenKind::CloseParen)
                | (Pat::Comma, TokenKind::Comma)
                | (Pat::Eq, TokenKind::Eq)
                | (Pat::Lifetime, TokenKind::Lifetime { .. })
                | (Pat::Lt, TokenKind::Lt)
                | (Pat::Gt, TokenKind::Gt)
                | (Pat::OpenBrace, TokenKind::OpenBrace)
                | (Pat::OpenBracket, TokenKind::OpenBracket)
                | (Pat::OpenParen, TokenKind::OpenParen)
                | (Pat::Pound, TokenKind::Pound)
                | (Pat::Semi, TokenKind::Semi)
                | (
                    Pat::LitStr,
                    TokenKind::Literal {
                        kind: LiteralKind::Str { terminated: true } | LiteralKind::RawStr { .. },
                        ..
                    },
                ) => {
                    self.step();
                    return true;
                },
                (Pat::Ident(x), TokenKind::Ident) if x == self.peek_text() => {
                    self.step();
                    return true;
                },
                (Pat::DoubleColon, TokenKind::Colon) => {
                    self.step();
                    if !self.at_end() && matches!(self.next_token.kind, TokenKind::Colon) {
                        self.step();
                        return true;
                    }
                    return false;
                },
                #[rustfmt::skip]
                (
                    Pat::CaptureLitStr,
                    TokenKind::Literal {
                        kind:
                            LiteralKind::Str { terminated: true }
                            | LiteralKind::RawStr { n_hashes: Some(_) },
                        ..
                    },
                )
                | (Pat::CaptureIdent, TokenKind::Ident) => {
                    **captures.next().unwrap() = self.peek_text();
                    self.step();
                    return true;
                },
                _ => return false,
            }
        }
    }

    /// Continually attempt to match the pattern on subsequent tokens until a match is
    /// found. Returns whether the pattern was successfully matched.
    ///
    /// Not generally suitable for multi-token patterns or patterns that can match
    /// nothing.
    #[must_use]
    pub fn find_pat(&mut self, token: Pat<'_>) -> bool {
        let mut capture = [].iter_mut();
        while !self.match_pat(token, &mut capture) {
            self.step();
            if self.at_end() {
                return false;
            }
        }
        true
    }

    /// The same as [`Self::find_pat`], but returns a capture as well.
    #[must_use]
    pub fn find_capture_pat(&mut self, token: Pat<'_>) -> Option<&'txt str> {
        let mut res = "";
        let mut capture = &mut res;
        let mut capture = slice::from_mut(&mut capture).iter_mut();
        while !self.match_pat(token, &mut capture) {
            self.step();
            if self.at_end() {
                return None;
            }
        }
        Some(res)
    }

    /// Attempts to match a sequence of patterns at the current position. Returns whether
    /// the match is successful.
    ///
    /// Captures will be written to the given slice in the order they're matched. If a
    /// capture is matched, but there are no more capture slots this will panic. If the
    /// match is completed without filling all the capture slots they will be left
    /// unmodified.
    ///
    /// If the match fails the cursor will be positioned at the first failing token.
    #[must_use]
    pub fn match_all(&mut self, tokens: &[Pat<'_>], captures: &mut [&mut &'txt str]) -> bool {
        let mut captures = captures.iter_mut();
        tokens.iter().all(|&t| self.match_pat(t, &mut captures))
    }
}
