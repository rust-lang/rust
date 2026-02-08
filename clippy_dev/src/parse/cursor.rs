use core::{ptr, slice};
use rustc_lexer::{self as lex, LiteralKind, Token, TokenKind};

/// A token pattern used for searching and matching by the [`Cursor`].
///
/// In the event that a pattern is a multi-token sequence, earlier tokens will be consumed
/// even if the pattern ultimately isn't matched. e.g. With the sequence `:*` matching
/// `DoubleColon` will consume the first `:` and then fail to match, leaving the cursor at
/// the `*`.
#[derive(Clone, Copy)]
pub enum Pat {
    /// Matches any number of comments and doc comments.
    AnyComments,
    CaptureDocLines,
    CaptureIdent,
    CaptureLifetime,
    CaptureLitStr,
    Bang,
    CloseBrace,
    CloseBracket,
    CloseParen,
    Comma,
    DoubleColon,
    Eq,
    FatArrow,
    Gt,
    Ident(IdentPat),
    Lifetime,
    LitStr,
    Lt,
    OpenBrace,
    OpenBracket,
    OpenParen,
    Pound,
    Semi,
}

macro_rules! ident_or_lit {
    ($ident:ident) => {
        stringify!($ident)
    };
    ($_ident:ident $lit:literal) => {
        $lit
    };
}
macro_rules! decl_ident_pats {
    ($($ident:ident $(= $s:literal)?,)*) => {
        #[allow(non_camel_case_types, clippy::upper_case_acronyms)]
        #[derive(Clone, Copy)]
        pub enum IdentPat { $($ident),* }
        impl IdentPat {
            pub fn as_str(self) -> &'static str {
                match self { $(Self::$ident => ident_or_lit!($ident $($s)?)),* }
            }
        }
    }
}
decl_ident_pats! {
    DEPRECATED,
    DEPRECATED_VERSION,
    RENAMED,
    RENAMED_VERSION,
    clippy,
    r#pub = "pub",
    version,
}

#[derive(Clone, Copy)]
pub struct Capture {
    pub pos: u32,
    pub len: u32,
}
impl Capture {
    pub const EMPTY: Self = Self { pos: 0, len: 0 };
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

    /// Gets the text of the captured token assuming it came from this cursor.
    #[must_use]
    pub fn get_text(&self, capture: Capture) -> &'txt str {
        &self.text[capture.pos as usize..(capture.pos + capture.len) as usize]
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
    fn match_impl(&mut self, pat: Pat, captures: &mut slice::IterMut<'_, Capture>) -> bool {
        loop {
            match (pat, self.next_token.kind) {
                #[rustfmt::skip] // rustfmt bug: https://github.com/rust-lang/rustfmt/issues/6697
                (_, TokenKind::Whitespace)
                | (
                    Pat::AnyComments,
                    TokenKind::BlockComment { terminated: true, .. } | TokenKind::LineComment { .. },
                ) => self.step(),
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
                (Pat::Ident(x), TokenKind::Ident) if x.as_str() == self.peek_text() => {
                    self.step();
                    return true;
                },
                (Pat::DoubleColon, TokenKind::Colon) => {
                    self.step();
                    if matches!(self.next_token.kind, TokenKind::Colon) {
                        self.step();
                        return true;
                    }
                    return false;
                },
                (Pat::FatArrow, TokenKind::Eq) => {
                    self.step();
                    if matches!(self.next_token.kind, TokenKind::Gt) {
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
                | (Pat::CaptureIdent, TokenKind::Ident)
                | (Pat::CaptureLifetime, TokenKind::Lifetime { .. }) => {
                    *captures.next().unwrap() = Capture { pos: self.pos, len: self.next_token.len };
                    self.step();
                    return true;
                },
                (Pat::CaptureDocLines, TokenKind::LineComment { doc_style: Some(_) }) => {
                    let pos = self.pos;
                    loop {
                        self.step();
                        if !matches!(
                            self.next_token.kind,
                            TokenKind::Whitespace | TokenKind::LineComment { doc_style: Some(_) }
                        ) {
                            break;
                        }
                    }
                    *captures.next().unwrap() = Capture {
                        pos,
                        len: self.pos - pos,
                    };
                    return true;
                },
                (Pat::CaptureDocLines, _) => {
                    *captures.next().unwrap() = Capture::EMPTY;
                    return true;
                },
                (Pat::AnyComments, _) => return true,
                _ => return false,
            }
        }
    }

    /// Consumes all tokens until the specified identifier is found and returns its
    /// position. Returns `None` if the identifier could not be found.
    ///
    /// The cursor will be positioned immediately after the identifier, or at the end if
    /// it is not.
    pub fn find_ident(&mut self, ident: &str) -> Option<u32> {
        loop {
            match self.next_token.kind {
                TokenKind::Ident if self.peek_text() == ident => {
                    let pos = self.pos;
                    self.step();
                    return Some(pos);
                },
                TokenKind::Eof => return None,
                _ => self.step(),
            }
        }
    }

    /// Consumes all tokens until the next identifier is found and captures it. Returns
    /// `None` if no identifier could be found.
    ///
    /// The cursor will be positioned immediately after the identifier, or at the end if
    /// it is not.
    pub fn find_any_ident(&mut self) -> Option<Capture> {
        loop {
            match self.next_token.kind {
                TokenKind::Ident => {
                    let res = Capture {
                        pos: self.pos,
                        len: self.next_token.len,
                    };
                    self.step();
                    return Some(res);
                },
                TokenKind::Eof => return None,
                _ => self.step(),
            }
        }
    }

    /// Consume the returns the position of the next non-whitespace token if it's the
    /// specified identifier. Returns `None` otherwise.
    pub fn match_ident(&mut self, s: &str) -> Option<u32> {
        loop {
            match self.next_token.kind {
                TokenKind::Ident if s == self.peek_text() => {
                    let pos = self.pos;
                    self.step();
                    return Some(pos);
                },
                TokenKind::Whitespace => self.step(),
                _ => return None,
            }
        }
    }

    /// Consumes and captures the next non-whitespace token if it's an identifier. Returns
    /// `None` otherwise.
    pub fn capture_ident(&mut self) -> Option<Capture> {
        loop {
            match self.next_token.kind {
                TokenKind::Ident => {
                    let pos = self.pos;
                    let len = self.next_token.len;
                    self.step();
                    return Some(Capture { pos, len });
                },
                TokenKind::Whitespace => self.step(),
                _ => return None,
            }
        }
    }

    /// Continually attempt to match the pattern on subsequent tokens until a match is
    /// found. Returns whether the pattern was successfully matched.
    ///
    /// Not generally suitable for multi-token patterns or patterns that can match
    /// nothing.
    #[must_use]
    pub fn find_pat(&mut self, pat: Pat) -> bool {
        let mut capture = [].iter_mut();
        while !self.match_impl(pat, &mut capture) {
            self.step();
            if self.at_end() {
                return false;
            }
        }
        true
    }

    /// Attempts to match a sequence of patterns at the current position. Returns whether
    /// all patterns were successfully matched.
    ///
    /// Captures will be written to the given slice in the order they're matched. If a
    /// capture is matched, but there are no more capture slots this will panic. If the
    /// match is completed without filling all the capture slots they will be left
    /// unmodified.
    ///
    /// If the match fails the cursor will be positioned at the first failing token.
    #[must_use]
    pub fn match_all(&mut self, pats: &[Pat], captures: &mut [Capture]) -> bool {
        let mut captures = captures.iter_mut();
        pats.iter().all(|&p| self.match_impl(p, &mut captures))
    }

    /// Attempts to match a sequence of patterns at the current position. Returns whether
    /// all patterns were successfully matched.
    ///
    /// Captures will be written to the given slice in the order they're matched. If a
    /// capture is matched, but there are no more capture slots this will panic. If the
    /// match is completed without filling all the capture slots they will be left
    /// unmodified.
    ///
    /// If the match fails the cursor will be positioned at the first failing token.
    #[must_use]
    pub fn opt_match_all(&mut self, pats: &[Pat], captures: &mut [Capture]) -> bool {
        let mut captures = captures.iter_mut();
        pats.iter()
            .try_for_each(|p| self.match_impl(*p, &mut captures).ok_or(p))
            .err()
            .is_none_or(|p| ptr::addr_eq(pats.as_ptr(), p))
    }

    /// Attempts to match a single pattern at the current position. Returns whether the
    /// pattern was successfully matched.
    ///
    /// If the pattern attempts to capture anything this will panic. If the match fails
    /// the cursor will be positioned at the first failing token.
    #[must_use]
    pub fn match_pat(&mut self, pat: Pat) -> bool {
        self.match_impl(pat, &mut [].iter_mut())
    }
}
