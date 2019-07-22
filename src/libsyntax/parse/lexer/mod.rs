use crate::parse::ParseSess;
use crate::parse::token::{self, Token, TokenKind};
use crate::symbol::{sym, Symbol};
use crate::parse::unescape;
use crate::parse::unescape_error_reporting::{emit_unescape_error, push_escaped_char};

use errors::{FatalError, Diagnostic, DiagnosticBuilder};
use syntax_pos::{BytePos, Pos, Span, NO_EXPANSION};
use rustc_lexer::Base;

use std::borrow::Cow;
use std::char;
use std::iter;
use std::convert::TryInto;
use rustc_data_structures::sync::Lrc;
use log::debug;

pub mod comments;
mod tokentrees;
mod unicode_chars;

#[derive(Clone, Debug)]
pub struct UnmatchedBrace {
    pub expected_delim: token::DelimToken,
    pub found_delim: token::DelimToken,
    pub found_span: Span,
    pub unclosed_span: Option<Span>,
    pub candidate_span: Option<Span>,
}

pub struct StringReader<'a> {
    crate sess: &'a ParseSess,
    /// The absolute offset within the source_map of the current character
    crate pos: BytePos,
    /// The current character (which has been read from self.pos)
    crate source_file: Lrc<syntax_pos::SourceFile>,
    /// Stop reading src at this index.
    crate end_src_index: usize,
    fatal_errs: Vec<DiagnosticBuilder<'a>>,
    // cache a direct reference to the source text, so that we don't have to
    // retrieve it via `self.source_file.src.as_ref().unwrap()` all the time.
    src: Lrc<String>,
    override_span: Option<Span>,
}

impl<'a> StringReader<'a> {
    pub fn new(sess: &'a ParseSess,
               source_file: Lrc<syntax_pos::SourceFile>,
               override_span: Option<Span>) -> Self {
        if source_file.src.is_none() {
            sess.span_diagnostic.bug(&format!("Cannot lex source_file without source: {}",
                                              source_file.name));
        }

        let src = (*source_file.src.as_ref().unwrap()).clone();

        StringReader {
            sess,
            pos: source_file.start_pos,
            source_file,
            end_src_index: src.len(),
            src,
            fatal_errs: Vec::new(),
            override_span,
        }
    }

    pub fn retokenize(sess: &'a ParseSess, mut span: Span) -> Self {
        let begin = sess.source_map().lookup_byte_offset(span.lo());
        let end = sess.source_map().lookup_byte_offset(span.hi());

        // Make the range zero-length if the span is invalid.
        if span.lo() > span.hi() || begin.sf.start_pos != end.sf.start_pos {
            span = span.shrink_to_lo();
        }

        let mut sr = StringReader::new(sess, begin.sf, None);

        // Seek the lexer to the right byte range.
        sr.end_src_index = sr.src_index(span.hi());

        sr
    }


    fn mk_sp(&self, lo: BytePos, hi: BytePos) -> Span {
        self.override_span.unwrap_or_else(|| Span::new(lo, hi, NO_EXPANSION))
    }

    fn unwrap_or_abort(&mut self, res: Result<Token, ()>) -> Token {
        match res {
            Ok(tok) => tok,
            Err(_) => {
                self.emit_fatal_errors();
                FatalError.raise();
            }
        }
    }

    /// Returns the next token, including trivia like whitespace or comments.
    ///
    /// `Err(())` means that some errors were encountered, which can be
    /// retrieved using `buffer_fatal_errors`.
    pub fn try_next_token(&mut self) -> Result<Token, ()> {
        assert!(self.fatal_errs.is_empty());

        let start_src_index = self.src_index(self.pos);
        let text: &str = &self.src[start_src_index..self.end_src_index];

        if text.is_empty() {
            let span = self.mk_sp(self.source_file.end_pos, self.source_file.end_pos);
            return Ok(Token::new(token::Eof, span));
        }

        {
            let is_beginning_of_file = self.pos == self.source_file.start_pos;
            if is_beginning_of_file {
                if let Some(shebang_len) = rustc_lexer::strip_shebang(text) {
                    let start = self.pos;
                    self.pos = self.pos + BytePos::from_usize(shebang_len);

                    let sym = self.symbol_from(start + BytePos::from_usize("#!".len()));
                    let kind = token::Shebang(sym);

                    let span = self.mk_sp(start, self.pos);
                    return Ok(Token::new(kind, span));
                }
            }
        }

        let token = rustc_lexer::first_token(text);

        let start = self.pos;
        self.pos = self.pos + BytePos::from_usize(token.len);

        debug!("try_next_token: {:?}({:?})", token.kind, self.str_from(start));

        // This could use `?`, but that makes code significantly (10-20%) slower.
        // https://github.com/rust-lang/rust/issues/37939
        let kind = match self.cook_lexer_token(token.kind, start) {
            Ok(it) => it,
            Err(err) => return Err(self.fatal_errs.push(err)),
        };

        let span = self.mk_sp(start, self.pos);
        Ok(Token::new(kind, span))
    }

    /// Returns the next token, including trivia like whitespace or comments.
    ///
    /// Aborts in case of an error.
    pub fn next_token(&mut self) -> Token {
        let res = self.try_next_token();
        self.unwrap_or_abort(res)
    }

    fn emit_fatal_errors(&mut self) {
        for err in &mut self.fatal_errs {
            err.emit();
        }

        self.fatal_errs.clear();
    }

    pub fn buffer_fatal_errors(&mut self) -> Vec<Diagnostic> {
        let mut buffer = Vec::new();

        for err in self.fatal_errs.drain(..) {
            err.buffer(&mut buffer);
        }

        buffer
    }

    /// Report a fatal lexical error with a given span.
    fn fatal_span(&self, sp: Span, m: &str) -> FatalError {
        self.sess.span_diagnostic.span_fatal(sp, m)
    }

    /// Report a lexical error with a given span.
    fn err_span(&self, sp: Span, m: &str) {
        self.sess.span_diagnostic.struct_span_err(sp, m).emit();
    }


    /// Report a fatal error spanning [`from_pos`, `to_pos`).
    fn fatal_span_(&self, from_pos: BytePos, to_pos: BytePos, m: &str) -> FatalError {
        self.fatal_span(self.mk_sp(from_pos, to_pos), m)
    }

    /// Report a lexical error spanning [`from_pos`, `to_pos`).
    fn err_span_(&self, from_pos: BytePos, to_pos: BytePos, m: &str) {
        self.err_span(self.mk_sp(from_pos, to_pos), m)
    }

    fn struct_span_fatal(&self, from_pos: BytePos, to_pos: BytePos, m: &str)
        -> DiagnosticBuilder<'a>
    {
        self.sess.span_diagnostic.struct_span_fatal(self.mk_sp(from_pos, to_pos), m)
    }

    fn struct_fatal_span_char(&self, from_pos: BytePos, to_pos: BytePos, m: &str, c: char)
        -> DiagnosticBuilder<'a>
    {
        let mut m = m.to_string();
        m.push_str(": ");
        push_escaped_char(&mut m, c);

        self.sess.span_diagnostic.struct_span_fatal(self.mk_sp(from_pos, to_pos), &m[..])
    }

    /// Turns simple `rustc_lexer::TokenKind` enum into a rich
    /// `libsyntax::TokenKind`. This turns strings into interned
    /// symbols and runs additional validation.
    fn cook_lexer_token(
        &self,
        token: rustc_lexer::TokenKind,
        start: BytePos,
    ) -> Result<TokenKind, DiagnosticBuilder<'a>> {
        let kind = match token {
            rustc_lexer::TokenKind::LineComment => {
                let string = self.str_from(start);
                // comments with only more "/"s are not doc comments
                let tok = if is_doc_comment(string) {
                    let mut idx = 0;
                    loop {
                        idx = match string[idx..].find('\r') {
                            None => break,
                            Some(it) => idx + it + 1
                        };
                        if string[idx..].chars().next() != Some('\n') {
                            self.err_span_(start + BytePos(idx as u32 - 1),
                                            start + BytePos(idx as u32),
                                            "bare CR not allowed in doc-comment");
                        }
                    }
                    token::DocComment(Symbol::intern(string))
                } else {
                    token::Comment
                };

                tok
            }
            rustc_lexer::TokenKind::BlockComment { terminated } => {
                let string = self.str_from(start);
                // block comments starting with "/**" or "/*!" are doc-comments
                // but comments with only "*"s between two "/"s are not
                let is_doc_comment = is_block_doc_comment(string);

                if !terminated {
                    let msg = if is_doc_comment {
                        "unterminated block doc-comment"
                    } else {
                        "unterminated block comment"
                    };
                    let last_bpos = self.pos;
                    self.fatal_span_(start, last_bpos, msg).raise();
                }

                let tok = if is_doc_comment {
                    let has_cr = string.contains('\r');
                    let string = if has_cr {
                        self.translate_crlf(start,
                                            string,
                                            "bare CR not allowed in block doc-comment")
                    } else {
                        string.into()
                    };
                    token::DocComment(Symbol::intern(&string[..]))
                } else {
                    token::Comment
                };

                tok
            }
            rustc_lexer::TokenKind::Whitespace => token::Whitespace,
            rustc_lexer::TokenKind::Ident | rustc_lexer::TokenKind::RawIdent => {
                let is_raw_ident = token == rustc_lexer::TokenKind::RawIdent;
                let mut ident_start = start;
                if is_raw_ident {
                    ident_start = ident_start + BytePos(2);
                }
                // FIXME: perform NFKC normalization here. (Issue #2253)
                let sym = self.symbol_from(ident_start);
                if is_raw_ident {
                    let span = self.mk_sp(start, self.pos);
                    if !sym.can_be_raw() {
                        self.err_span(span, &format!("`{}` cannot be a raw identifier", sym));
                    }
                    self.sess.raw_identifier_spans.borrow_mut().push(span);
                }
                token::Ident(sym, is_raw_ident)
            }
            rustc_lexer::TokenKind::Literal { kind, suffix_start } => {
                let suffix_start = start + BytePos(suffix_start as u32);
                let (kind, symbol) = self.cook_lexer_literal(start, suffix_start, kind);
                let suffix = if suffix_start < self.pos {
                    let string = self.str_from(suffix_start);
                    if string == "_" {
                        self.sess.span_diagnostic
                            .struct_span_warn(self.mk_sp(suffix_start, self.pos),
                                              "underscore literal suffix is not allowed")
                            .warn("this was previously accepted by the compiler but is \
                                   being phased out; it will become a hard error in \
                                   a future release!")
                            .note("for more information, see issue #42326 \
                                   <https://github.com/rust-lang/rust/issues/42326>")
                            .emit();
                        None
                    } else {
                        Some(Symbol::intern(string))
                    }
                } else {
                    None
                };
                token::Literal(token::Lit { kind, symbol, suffix })
            }
            rustc_lexer::TokenKind::Lifetime { starts_with_number } => {
                // Include the leading `'` in the real identifier, for macro
                // expansion purposes. See #12512 for the gory details of why
                // this is necessary.
                let lifetime_name = self.str_from(start);
                if starts_with_number {
                    self.err_span_(
                        start,
                        self.pos,
                        "lifetimes cannot start with a number",
                    );
                }
                let ident = Symbol::intern(lifetime_name);
                token::Lifetime(ident)
            }
            rustc_lexer::TokenKind::Semi => token::Semi,
            rustc_lexer::TokenKind::Comma => token::Comma,
            rustc_lexer::TokenKind::DotDotDot => token::DotDotDot,
            rustc_lexer::TokenKind::DotDotEq => token::DotDotEq,
            rustc_lexer::TokenKind::DotDot => token::DotDot,
            rustc_lexer::TokenKind::Dot => token::Dot,
            rustc_lexer::TokenKind::OpenParen => token::OpenDelim(token::Paren),
            rustc_lexer::TokenKind::CloseParen => token::CloseDelim(token::Paren),
            rustc_lexer::TokenKind::OpenBrace => token::OpenDelim(token::Brace),
            rustc_lexer::TokenKind::CloseBrace => token::CloseDelim(token::Brace),
            rustc_lexer::TokenKind::OpenBracket => token::OpenDelim(token::Bracket),
            rustc_lexer::TokenKind::CloseBracket => token::CloseDelim(token::Bracket),
            rustc_lexer::TokenKind::At => token::At,
            rustc_lexer::TokenKind::Pound => token::Pound,
            rustc_lexer::TokenKind::Tilde => token::Tilde,
            rustc_lexer::TokenKind::Question => token::Question,
            rustc_lexer::TokenKind::ColonColon => token::ModSep,
            rustc_lexer::TokenKind::Colon => token::Colon,
            rustc_lexer::TokenKind::Dollar => token::Dollar,
            rustc_lexer::TokenKind::EqEq => token::EqEq,
            rustc_lexer::TokenKind::Eq => token::Eq,
            rustc_lexer::TokenKind::FatArrow => token::FatArrow,
            rustc_lexer::TokenKind::Ne => token::Ne,
            rustc_lexer::TokenKind::Not => token::Not,
            rustc_lexer::TokenKind::Le => token::Le,
            rustc_lexer::TokenKind::LArrow => token::LArrow,
            rustc_lexer::TokenKind::Lt => token::Lt,
            rustc_lexer::TokenKind::ShlEq => token::BinOpEq(token::Shl),
            rustc_lexer::TokenKind::Shl => token::BinOp(token::Shl),
            rustc_lexer::TokenKind::Ge => token::Ge,
            rustc_lexer::TokenKind::Gt => token::Gt,
            rustc_lexer::TokenKind::ShrEq => token::BinOpEq(token::Shr),
            rustc_lexer::TokenKind::Shr => token::BinOp(token::Shr),
            rustc_lexer::TokenKind::RArrow => token::RArrow,
            rustc_lexer::TokenKind::Minus => token::BinOp(token::Minus),
            rustc_lexer::TokenKind::MinusEq => token::BinOpEq(token::Minus),
            rustc_lexer::TokenKind::And => token::BinOp(token::And),
            rustc_lexer::TokenKind::AndEq => token::BinOpEq(token::And),
            rustc_lexer::TokenKind::AndAnd => token::AndAnd,
            rustc_lexer::TokenKind::Or => token::BinOp(token::Or),
            rustc_lexer::TokenKind::OrEq => token::BinOpEq(token::Or),
            rustc_lexer::TokenKind::OrOr => token::OrOr,
            rustc_lexer::TokenKind::Plus => token::BinOp(token::Plus),
            rustc_lexer::TokenKind::PlusEq => token::BinOpEq(token::Plus),
            rustc_lexer::TokenKind::Star => token::BinOp(token::Star),
            rustc_lexer::TokenKind::StarEq => token::BinOpEq(token::Star),
            rustc_lexer::TokenKind::Slash => token::BinOp(token::Slash),
            rustc_lexer::TokenKind::SlashEq => token::BinOpEq(token::Slash),
            rustc_lexer::TokenKind::Caret => token::BinOp(token::Caret),
            rustc_lexer::TokenKind::CaretEq => token::BinOpEq(token::Caret),
            rustc_lexer::TokenKind::Percent => token::BinOp(token::Percent),
            rustc_lexer::TokenKind::PercentEq => token::BinOpEq(token::Percent),

            rustc_lexer::TokenKind::Unknown => {
                let c = self.str_from(start).chars().next().unwrap();
                let mut err = self.struct_fatal_span_char(start,
                                                          self.pos,
                                                          "unknown start of token",
                                                          c);
                unicode_chars::check_for_substitution(self, start, c, &mut err);
                return Err(err)
            }
        };
        Ok(kind)
    }

    fn cook_lexer_literal(
        &self,
        start: BytePos,
        suffix_start: BytePos,
        kind: rustc_lexer::LiteralKind
    ) -> (token::LitKind, Symbol) {
        match kind {
            rustc_lexer::LiteralKind::Char { terminated } => {
                if !terminated {
                    self.fatal_span_(start, suffix_start,
                                     "unterminated character literal".into())
                        .raise()
                }
                let content_start = start + BytePos(1);
                let content_end = suffix_start - BytePos(1);
                self.validate_char_escape(content_start, content_end);
                let id = self.symbol_from_to(content_start, content_end);
                (token::Char, id)
            },
            rustc_lexer::LiteralKind::Byte { terminated } => {
                if !terminated {
                    self.fatal_span_(start + BytePos(1), suffix_start,
                                     "unterminated byte constant".into())
                        .raise()
                }
                let content_start = start + BytePos(2);
                let content_end = suffix_start - BytePos(1);
                self.validate_byte_escape(content_start, content_end);
                let id = self.symbol_from_to(content_start, content_end);
                (token::Byte, id)
            },
            rustc_lexer::LiteralKind::Str { terminated } => {
                if !terminated {
                    self.fatal_span_(start, suffix_start,
                                     "unterminated double quote string".into())
                        .raise()
                }
                let content_start = start + BytePos(1);
                let content_end = suffix_start - BytePos(1);
                self.validate_str_escape(content_start, content_end);
                let id = self.symbol_from_to(content_start, content_end);
                (token::Str, id)
            }
            rustc_lexer::LiteralKind::ByteStr { terminated } => {
                if !terminated {
                    self.fatal_span_(start + BytePos(1), suffix_start,
                                     "unterminated double quote byte string".into())
                        .raise()
                }
                let content_start = start + BytePos(2);
                let content_end = suffix_start - BytePos(1);
                self.validate_byte_str_escape(content_start, content_end);
                let id = self.symbol_from_to(content_start, content_end);
                (token::ByteStr, id)
            }
            rustc_lexer::LiteralKind::RawStr { n_hashes, started, terminated } => {
                if !started {
                    self.report_non_started_raw_string(start);
                }
                if !terminated {
                    self.report_unterminated_raw_string(start, n_hashes)
                }
                let n_hashes: u16 = self.restrict_n_hashes(start, n_hashes);
                let n = u32::from(n_hashes);
                let content_start = start + BytePos(2 + n);
                let content_end = suffix_start - BytePos(1 + n);
                self.validate_raw_str_escape(content_start, content_end);
                let id = self.symbol_from_to(content_start, content_end);
                (token::StrRaw(n_hashes), id)
            }
            rustc_lexer::LiteralKind::RawByteStr { n_hashes, started, terminated } => {
                if !started {
                    self.report_non_started_raw_string(start);
                }
                if !terminated {
                    self.report_unterminated_raw_string(start, n_hashes)
                }
                let n_hashes: u16 = self.restrict_n_hashes(start, n_hashes);
                let n = u32::from(n_hashes);
                let content_start = start + BytePos(3 + n);
                let content_end = suffix_start - BytePos(1 + n);
                self.validate_raw_byte_str_escape(content_start, content_end);
                let id = self.symbol_from_to(content_start, content_end);
                (token::ByteStrRaw(n_hashes), id)
            }
            rustc_lexer::LiteralKind::Int { base, empty_int } => {
                if empty_int {
                    self.err_span_(start, suffix_start, "no valid digits found for number");
                    (token::Integer, sym::integer(0))
                } else {
                    self.validate_int_literal(base, start, suffix_start);
                    (token::Integer, self.symbol_from_to(start, suffix_start))
                }
            },
            rustc_lexer::LiteralKind::Float { base, empty_exponent } => {
                if empty_exponent {
                    let mut err = self.struct_span_fatal(
                        start, self.pos,
                        "expected at least one digit in exponent"
                    );
                    err.emit();
                }

                match base {
                    Base::Hexadecimal => {
                        self.err_span_(start, suffix_start,
                                       "hexadecimal float literal is not supported")
                    }
                    Base::Octal => {
                        self.err_span_(start, suffix_start,
                                       "octal float literal is not supported")
                    }
                    Base::Binary => {
                        self.err_span_(start, suffix_start,
                                       "binary float literal is not supported")
                    }
                    _ => ()
                }

                let id = self.symbol_from_to(start, suffix_start);
                (token::Float, id)
            },
        }
    }

    #[inline]
    fn src_index(&self, pos: BytePos) -> usize {
        (pos - self.source_file.start_pos).to_usize()
    }

    /// Slice of the source text from `start` up to but excluding `self.pos`,
    /// meaning the slice does not include the character `self.ch`.
    fn str_from(&self, start: BytePos) -> &str
    {
        self.str_from_to(start, self.pos)
    }

    /// Creates a Symbol from a given offset to the current offset.
    fn symbol_from(&self, start: BytePos) -> Symbol {
        debug!("taking an ident from {:?} to {:?}", start, self.pos);
        Symbol::intern(self.str_from(start))
    }

    /// As symbol_from, with an explicit endpoint.
    fn symbol_from_to(&self, start: BytePos, end: BytePos) -> Symbol {
        debug!("taking an ident from {:?} to {:?}", start, end);
        Symbol::intern(self.str_from_to(start, end))
    }

    /// Slice of the source text spanning from `start` up to but excluding `end`.
    fn str_from_to(&self, start: BytePos, end: BytePos) -> &str
    {
        &self.src[self.src_index(start)..self.src_index(end)]
    }

    /// Converts CRLF to LF in the given string, raising an error on bare CR.
    fn translate_crlf<'b>(&self, start: BytePos, s: &'b str, errmsg: &'b str) -> Cow<'b, str> {
        let mut chars = s.char_indices().peekable();
        while let Some((i, ch)) = chars.next() {
            if ch == '\r' {
                if let Some((lf_idx, '\n')) = chars.peek() {
                    return translate_crlf_(self, start, s, *lf_idx, chars, errmsg).into();
                }
                let pos = start + BytePos(i as u32);
                let end_pos = start + BytePos((i + ch.len_utf8()) as u32);
                self.err_span_(pos, end_pos, errmsg);
            }
        }
        return s.into();

        fn translate_crlf_(rdr: &StringReader<'_>,
                           start: BytePos,
                           s: &str,
                           mut j: usize,
                           mut chars: iter::Peekable<impl Iterator<Item = (usize, char)>>,
                           errmsg: &str)
                           -> String {
            let mut buf = String::with_capacity(s.len());
            // Skip first CR
            buf.push_str(&s[.. j - 1]);
            while let Some((i, ch)) = chars.next() {
                if ch == '\r' {
                    if j < i {
                        buf.push_str(&s[j..i]);
                    }
                    let next = i + ch.len_utf8();
                    j = next;
                    if chars.peek().map(|(_, ch)| *ch) != Some('\n') {
                        let pos = start + BytePos(i as u32);
                        let end_pos = start + BytePos(next as u32);
                        rdr.err_span_(pos, end_pos, errmsg);
                    }
                }
            }
            if j < s.len() {
                buf.push_str(&s[j..]);
            }
            buf
        }
    }

    fn report_non_started_raw_string(&self, start: BytePos) -> ! {
        let bad_char = self.str_from(start).chars().last().unwrap();
        self
            .struct_fatal_span_char(
                start,
                self.pos,
                "found invalid character; only `#` is allowed \
                 in raw string delimitation",
                bad_char,
            )
            .emit();
        FatalError.raise()
    }

    fn report_unterminated_raw_string(&self, start: BytePos, n_hashes: usize) -> ! {
        let mut err = self.struct_span_fatal(
            start, start,
            "unterminated raw string",
        );
        err.span_label(
            self.mk_sp(start, start),
            "unterminated raw string",
        );

        if n_hashes > 0 {
            err.note(&format!("this raw string should be terminated with `\"{}`",
                                "#".repeat(n_hashes as usize)));
        }

        err.emit();
        FatalError.raise()
    }

    fn restrict_n_hashes(&self, start: BytePos, n_hashes: usize) -> u16 {
        match n_hashes.try_into() {
            Ok(n_hashes) => n_hashes,
            Err(_) => {
                self.fatal_span_(start,
                                 self.pos,
                                 "too many `#` symbols: raw strings may be \
                                  delimited by up to 65535 `#` symbols").raise();
            }
        }
    }

    fn validate_char_escape(&self, content_start: BytePos, content_end: BytePos) {
        let lit = self.str_from_to(content_start, content_end);
        if let Err((off, err)) = unescape::unescape_char(lit) {
            emit_unescape_error(
                &self.sess.span_diagnostic,
                lit,
                self.mk_sp(content_start - BytePos(1), content_end + BytePos(1)),
                unescape::Mode::Char,
                0..off,
                err,
            )
        }
    }

    fn validate_byte_escape(&self, content_start: BytePos, content_end: BytePos) {
        let lit = self.str_from_to(content_start, content_end);
        if let Err((off, err)) = unescape::unescape_byte(lit) {
            emit_unescape_error(
                &self.sess.span_diagnostic,
                lit,
                self.mk_sp(content_start - BytePos(1), content_end + BytePos(1)),
                unescape::Mode::Byte,
                0..off,
                err,
            )
        }
    }

    fn validate_str_escape(&self, content_start: BytePos, content_end: BytePos) {
        let lit = self.str_from_to(content_start, content_end);
        unescape::unescape_str(lit, &mut |range, c| {
            if let Err(err) = c {
                emit_unescape_error(
                    &self.sess.span_diagnostic,
                    lit,
                    self.mk_sp(content_start - BytePos(1), content_end + BytePos(1)),
                    unescape::Mode::Str,
                    range,
                    err,
                )
            }
        })
    }

    fn validate_raw_str_escape(&self, content_start: BytePos, content_end: BytePos) {
        let lit = self.str_from_to(content_start, content_end);
        unescape::unescape_raw_str(lit, &mut |range, c| {
            if let Err(err) = c {
                emit_unescape_error(
                    &self.sess.span_diagnostic,
                    lit,
                    self.mk_sp(content_start - BytePos(1), content_end + BytePos(1)),
                    unescape::Mode::Str,
                    range,
                    err,
                )
            }
        })
    }

    fn validate_raw_byte_str_escape(&self, content_start: BytePos, content_end: BytePos) {
        let lit = self.str_from_to(content_start, content_end);
        unescape::unescape_raw_byte_str(lit, &mut |range, c| {
            if let Err(err) = c {
                emit_unescape_error(
                    &self.sess.span_diagnostic,
                    lit,
                    self.mk_sp(content_start - BytePos(1), content_end + BytePos(1)),
                    unescape::Mode::ByteStr,
                    range,
                    err,
                )
            }
        })
    }

    fn validate_byte_str_escape(&self, content_start: BytePos, content_end: BytePos) {
        let lit = self.str_from_to(content_start, content_end);
        unescape::unescape_byte_str(lit, &mut |range, c| {
            if let Err(err) = c {
                emit_unescape_error(
                    &self.sess.span_diagnostic,
                    lit,
                    self.mk_sp(content_start - BytePos(1), content_end + BytePos(1)),
                    unescape::Mode::ByteStr,
                    range,
                    err,
                )
            }
        })
    }

    fn validate_int_literal(&self, base: Base, content_start: BytePos, content_end: BytePos) {
        let base = match base {
            Base::Binary => 2,
            Base::Octal => 8,
            _ => return,
        };
        let s = self.str_from_to(content_start + BytePos(2), content_end);
        for (idx, c) in s.char_indices() {
            let idx = idx as u32;
            if c != '_' && c.to_digit(base).is_none() {
                let lo = content_start + BytePos(2 + idx);
                let hi = content_start + BytePos(2 + idx + c.len_utf8() as u32);
                self.err_span_(lo, hi,
                               &format!("invalid digit for a base {} literal", base));

            }
        }
    }
}

fn is_doc_comment(s: &str) -> bool {
    let res = (s.starts_with("///") && *s.as_bytes().get(3).unwrap_or(&b' ') != b'/') ||
              s.starts_with("//!");
    debug!("is {:?} a doc comment? {}", s, res);
    res
}

fn is_block_doc_comment(s: &str) -> bool {
    // Prevent `/**/` from being parsed as a doc comment
    let res = ((s.starts_with("/**") && *s.as_bytes().get(3).unwrap_or(&b' ') != b'*') ||
               s.starts_with("/*!")) && s.len() >= 5;
    debug!("is {:?} a doc comment? {}", s, res);
    res
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::ast::CrateConfig;
    use crate::symbol::Symbol;
    use crate::source_map::{SourceMap, FilePathMapping};
    use crate::feature_gate::UnstableFeatures;
    use crate::parse::token;
    use crate::diagnostics::plugin::ErrorMap;
    use crate::with_default_globals;
    use std::io;
    use std::path::PathBuf;
    use syntax_pos::{BytePos, Span, NO_EXPANSION, edition::Edition};
    use rustc_data_structures::fx::{FxHashSet, FxHashMap};
    use rustc_data_structures::sync::Lock;

    fn mk_sess(sm: Lrc<SourceMap>) -> ParseSess {
        let emitter = errors::emitter::EmitterWriter::new(Box::new(io::sink()),
                                                          Some(sm.clone()),
                                                          false,
                                                          false,
                                                          false);
        ParseSess {
            span_diagnostic: errors::Handler::with_emitter(true, None, Box::new(emitter)),
            unstable_features: UnstableFeatures::from_environment(),
            config: CrateConfig::default(),
            included_mod_stack: Lock::new(Vec::new()),
            source_map: sm,
            missing_fragment_specifiers: Lock::new(FxHashSet::default()),
            raw_identifier_spans: Lock::new(Vec::new()),
            registered_diagnostics: Lock::new(ErrorMap::new()),
            buffered_lints: Lock::new(vec![]),
            edition: Edition::from_session(),
            ambiguous_block_expr_parse: Lock::new(FxHashMap::default()),
            param_attr_spans: Lock::new(Vec::new()),
            let_chains_spans: Lock::new(Vec::new()),
            async_closure_spans: Lock::new(Vec::new()),
        }
    }

    // open a string reader for the given string
    fn setup<'a>(sm: &SourceMap,
                 sess: &'a ParseSess,
                 teststr: String)
                 -> StringReader<'a> {
        let sf = sm.new_source_file(PathBuf::from(teststr.clone()).into(), teststr);
        StringReader::new(sess, sf, None)
    }

    #[test]
    fn t1() {
        with_default_globals(|| {
            let sm = Lrc::new(SourceMap::new(FilePathMapping::empty()));
            let sh = mk_sess(sm.clone());
            let mut string_reader = setup(&sm,
                                        &sh,
                                        "/* my source file */ fn main() { println!(\"zebra\"); }\n"
                                            .to_string());
            assert_eq!(string_reader.next_token(), token::Comment);
            assert_eq!(string_reader.next_token(), token::Whitespace);
            let tok1 = string_reader.next_token();
            let tok2 = Token::new(
                mk_ident("fn"),
                Span::new(BytePos(21), BytePos(23), NO_EXPANSION),
            );
            assert_eq!(tok1.kind, tok2.kind);
            assert_eq!(tok1.span, tok2.span);
            assert_eq!(string_reader.next_token(), token::Whitespace);
            // read another token:
            let tok3 = string_reader.next_token();
            assert_eq!(string_reader.pos.clone(), BytePos(28));
            let tok4 = Token::new(
                mk_ident("main"),
                Span::new(BytePos(24), BytePos(28), NO_EXPANSION),
            );
            assert_eq!(tok3.kind, tok4.kind);
            assert_eq!(tok3.span, tok4.span);

            assert_eq!(string_reader.next_token(), token::OpenDelim(token::Paren));
            assert_eq!(string_reader.pos.clone(), BytePos(29))
        })
    }

    // check that the given reader produces the desired stream
    // of tokens (stop checking after exhausting the expected vec)
    fn check_tokenization(mut string_reader: StringReader<'_>, expected: Vec<TokenKind>) {
        for expected_tok in &expected {
            assert_eq!(&string_reader.next_token(), expected_tok);
        }
    }

    // make the identifier by looking up the string in the interner
    fn mk_ident(id: &str) -> TokenKind {
        token::Ident(Symbol::intern(id), false)
    }

    fn mk_lit(kind: token::LitKind, symbol: &str, suffix: Option<&str>) -> TokenKind {
        TokenKind::lit(kind, Symbol::intern(symbol), suffix.map(Symbol::intern))
    }

    #[test]
    fn doublecolonparsing() {
        with_default_globals(|| {
            let sm = Lrc::new(SourceMap::new(FilePathMapping::empty()));
            let sh = mk_sess(sm.clone());
            check_tokenization(setup(&sm, &sh, "a b".to_string()),
                            vec![mk_ident("a"), token::Whitespace, mk_ident("b")]);
        })
    }

    #[test]
    fn dcparsing_2() {
        with_default_globals(|| {
            let sm = Lrc::new(SourceMap::new(FilePathMapping::empty()));
            let sh = mk_sess(sm.clone());
            check_tokenization(setup(&sm, &sh, "a::b".to_string()),
                            vec![mk_ident("a"), token::ModSep, mk_ident("b")]);
        })
    }

    #[test]
    fn dcparsing_3() {
        with_default_globals(|| {
            let sm = Lrc::new(SourceMap::new(FilePathMapping::empty()));
            let sh = mk_sess(sm.clone());
            check_tokenization(setup(&sm, &sh, "a ::b".to_string()),
                            vec![mk_ident("a"), token::Whitespace, token::ModSep, mk_ident("b")]);
        })
    }

    #[test]
    fn dcparsing_4() {
        with_default_globals(|| {
            let sm = Lrc::new(SourceMap::new(FilePathMapping::empty()));
            let sh = mk_sess(sm.clone());
            check_tokenization(setup(&sm, &sh, "a:: b".to_string()),
                            vec![mk_ident("a"), token::ModSep, token::Whitespace, mk_ident("b")]);
        })
    }

    #[test]
    fn character_a() {
        with_default_globals(|| {
            let sm = Lrc::new(SourceMap::new(FilePathMapping::empty()));
            let sh = mk_sess(sm.clone());
            assert_eq!(setup(&sm, &sh, "'a'".to_string()).next_token(),
                       mk_lit(token::Char, "a", None));
        })
    }

    #[test]
    fn character_space() {
        with_default_globals(|| {
            let sm = Lrc::new(SourceMap::new(FilePathMapping::empty()));
            let sh = mk_sess(sm.clone());
            assert_eq!(setup(&sm, &sh, "' '".to_string()).next_token(),
                       mk_lit(token::Char, " ", None));
        })
    }

    #[test]
    fn character_escaped() {
        with_default_globals(|| {
            let sm = Lrc::new(SourceMap::new(FilePathMapping::empty()));
            let sh = mk_sess(sm.clone());
            assert_eq!(setup(&sm, &sh, "'\\n'".to_string()).next_token(),
                       mk_lit(token::Char, "\\n", None));
        })
    }

    #[test]
    fn lifetime_name() {
        with_default_globals(|| {
            let sm = Lrc::new(SourceMap::new(FilePathMapping::empty()));
            let sh = mk_sess(sm.clone());
            assert_eq!(setup(&sm, &sh, "'abc".to_string()).next_token(),
                       token::Lifetime(Symbol::intern("'abc")));
        })
    }

    #[test]
    fn raw_string() {
        with_default_globals(|| {
            let sm = Lrc::new(SourceMap::new(FilePathMapping::empty()));
            let sh = mk_sess(sm.clone());
            assert_eq!(setup(&sm, &sh, "r###\"\"#a\\b\x00c\"\"###".to_string()).next_token(),
                       mk_lit(token::StrRaw(3), "\"#a\\b\x00c\"", None));
        })
    }

    #[test]
    fn literal_suffixes() {
        with_default_globals(|| {
            let sm = Lrc::new(SourceMap::new(FilePathMapping::empty()));
            let sh = mk_sess(sm.clone());
            macro_rules! test {
                ($input: expr, $tok_type: ident, $tok_contents: expr) => {{
                    assert_eq!(setup(&sm, &sh, format!("{}suffix", $input)).next_token(),
                               mk_lit(token::$tok_type, $tok_contents, Some("suffix")));
                    // with a whitespace separator:
                    assert_eq!(setup(&sm, &sh, format!("{} suffix", $input)).next_token(),
                               mk_lit(token::$tok_type, $tok_contents, None));
                }}
            }

            test!("'a'", Char, "a");
            test!("b'a'", Byte, "a");
            test!("\"a\"", Str, "a");
            test!("b\"a\"", ByteStr, "a");
            test!("1234", Integer, "1234");
            test!("0b101", Integer, "0b101");
            test!("0xABC", Integer, "0xABC");
            test!("1.0", Float, "1.0");
            test!("1.0e10", Float, "1.0e10");

            assert_eq!(setup(&sm, &sh, "2us".to_string()).next_token(),
                       mk_lit(token::Integer, "2", Some("us")));
            assert_eq!(setup(&sm, &sh, "r###\"raw\"###suffix".to_string()).next_token(),
                       mk_lit(token::StrRaw(3), "raw", Some("suffix")));
            assert_eq!(setup(&sm, &sh, "br###\"raw\"###suffix".to_string()).next_token(),
                       mk_lit(token::ByteStrRaw(3), "raw", Some("suffix")));
        })
    }

    #[test]
    fn line_doc_comments() {
        assert!(is_doc_comment("///"));
        assert!(is_doc_comment("/// blah"));
        assert!(!is_doc_comment("////"));
    }

    #[test]
    fn nested_block_comments() {
        with_default_globals(|| {
            let sm = Lrc::new(SourceMap::new(FilePathMapping::empty()));
            let sh = mk_sess(sm.clone());
            let mut lexer = setup(&sm, &sh, "/* /* */ */'a'".to_string());
            assert_eq!(lexer.next_token(), token::Comment);
            assert_eq!(lexer.next_token(), mk_lit(token::Char, "a", None));
        })
    }

    #[test]
    fn crlf_comments() {
        with_default_globals(|| {
            let sm = Lrc::new(SourceMap::new(FilePathMapping::empty()));
            let sh = mk_sess(sm.clone());
            let mut lexer = setup(&sm, &sh, "// test\r\n/// test\r\n".to_string());
            let comment = lexer.next_token();
            assert_eq!(comment.kind, token::Comment);
            assert_eq!((comment.span.lo(), comment.span.hi()), (BytePos(0), BytePos(7)));
            assert_eq!(lexer.next_token(), token::Whitespace);
            assert_eq!(lexer.next_token(), token::DocComment(Symbol::intern("/// test")));
        })
    }
}
