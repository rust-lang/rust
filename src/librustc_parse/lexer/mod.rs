use rustc_ast::token::{self, Token, TokenKind};
use rustc_ast::util::comments;
use rustc_data_structures::sync::Lrc;
use rustc_errors::{error_code, Applicability, DiagnosticBuilder, FatalError};
use rustc_lexer::Base;
use rustc_lexer::{unescape, LexRawStrError, UnvalidatedRawStr, ValidatedRawStr};
use rustc_session::parse::ParseSess;
use rustc_span::symbol::{sym, Symbol};
use rustc_span::{BytePos, Pos, Span};

use log::debug;
use std::char;

mod tokentrees;
mod unescape_error_reporting;
mod unicode_chars;

use unescape_error_reporting::{emit_unescape_error, push_escaped_char};

#[derive(Clone, Debug)]
pub struct UnmatchedBrace {
    pub expected_delim: token::DelimToken,
    pub found_delim: Option<token::DelimToken>,
    pub found_span: Span,
    pub unclosed_span: Option<Span>,
    pub candidate_span: Option<Span>,
}

pub struct StringReader<'a> {
    sess: &'a ParseSess,
    /// Initial position, read-only.
    start_pos: BytePos,
    /// The absolute offset within the source_map of the current character.
    // FIXME(#64197): `pub` is needed by tests for now.
    pub pos: BytePos,
    /// Stop reading src at this index.
    end_src_index: usize,
    /// Source text to tokenize.
    src: Lrc<String>,
    override_span: Option<Span>,
}

impl<'a> StringReader<'a> {
    pub fn new(
        sess: &'a ParseSess,
        source_file: Lrc<rustc_span::SourceFile>,
        override_span: Option<Span>,
    ) -> Self {
        // Make sure external source is loaded first, before accessing it.
        // While this can't show up during normal parsing, `retokenize` may
        // be called with a source file from an external crate.
        sess.source_map().ensure_source_file_source_present(source_file.clone());

        // FIXME(eddyb) use `Lrc<str>` or similar to avoid cloning the `String`.
        let src = if let Some(src) = &source_file.src {
            src.clone()
        } else if let Some(src) = source_file.external_src.borrow().get_source() {
            src.clone()
        } else {
            sess.span_diagnostic
                .bug(&format!("cannot lex `source_file` without source: {}", source_file.name));
        };

        StringReader {
            sess,
            start_pos: source_file.start_pos,
            pos: source_file.start_pos,
            end_src_index: src.len(),
            src,
            override_span,
        }
    }

    pub fn retokenize(sess: &'a ParseSess, mut span: Span) -> Self {
        let begin = sess.source_map().lookup_byte_offset(span.lo());
        let end = sess.source_map().lookup_byte_offset(span.hi());

        // Make the range zero-length if the span is invalid.
        if begin.sf.start_pos != end.sf.start_pos {
            span = span.shrink_to_lo();
        }

        let mut sr = StringReader::new(sess, begin.sf, None);

        // Seek the lexer to the right byte range.
        sr.end_src_index = sr.src_index(span.hi());

        sr
    }

    fn mk_sp(&self, lo: BytePos, hi: BytePos) -> Span {
        self.override_span.unwrap_or_else(|| Span::with_root_ctxt(lo, hi))
    }

    /// Returns the next token, including trivia like whitespace or comments.
    pub fn next_token(&mut self) -> Token {
        let start_src_index = self.src_index(self.pos);
        let text: &str = &self.src[start_src_index..self.end_src_index];

        if text.is_empty() {
            let span = self.mk_sp(self.pos, self.pos);
            return Token::new(token::Eof, span);
        }

        {
            let is_beginning_of_file = self.pos == self.start_pos;
            if is_beginning_of_file {
                if let Some(shebang_len) = rustc_lexer::strip_shebang(text) {
                    let start = self.pos;
                    self.pos = self.pos + BytePos::from_usize(shebang_len);

                    let sym = self.symbol_from(start + BytePos::from_usize("#!".len()));
                    let kind = token::Shebang(sym);

                    let span = self.mk_sp(start, self.pos);
                    return Token::new(kind, span);
                }
            }
        }

        let token = rustc_lexer::first_token(text);

        let start = self.pos;
        self.pos = self.pos + BytePos::from_usize(token.len);

        debug!("try_next_token: {:?}({:?})", token.kind, self.str_from(start));

        // This could use `?`, but that makes code significantly (10-20%) slower.
        // https://github.com/rust-lang/rust/issues/37939
        let kind = self.cook_lexer_token(token.kind, start);

        let span = self.mk_sp(start, self.pos);
        Token::new(kind, span)
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

    fn struct_span_fatal(
        &self,
        from_pos: BytePos,
        to_pos: BytePos,
        m: &str,
    ) -> DiagnosticBuilder<'a> {
        self.sess.span_diagnostic.struct_span_fatal(self.mk_sp(from_pos, to_pos), m)
    }

    fn struct_fatal_span_char(
        &self,
        from_pos: BytePos,
        to_pos: BytePos,
        m: &str,
        c: char,
    ) -> DiagnosticBuilder<'a> {
        let mut m = m.to_string();
        m.push_str(": ");
        push_escaped_char(&mut m, c);

        self.sess.span_diagnostic.struct_span_fatal(self.mk_sp(from_pos, to_pos), &m[..])
    }

    /// Turns simple `rustc_lexer::TokenKind` enum into a rich
    /// `librustc_ast::TokenKind`. This turns strings into interned
    /// symbols and runs additional validation.
    fn cook_lexer_token(&self, token: rustc_lexer::TokenKind, start: BytePos) -> TokenKind {
        match token {
            rustc_lexer::TokenKind::LineComment => {
                let string = self.str_from(start);
                // comments with only more "/"s are not doc comments
                if comments::is_line_doc_comment(string) {
                    self.forbid_bare_cr(start, string, "bare CR not allowed in doc-comment");
                    token::DocComment(Symbol::intern(string))
                } else {
                    token::Comment
                }
            }
            rustc_lexer::TokenKind::BlockComment { terminated } => {
                let string = self.str_from(start);
                // block comments starting with "/**" or "/*!" are doc-comments
                // but comments with only "*"s between two "/"s are not
                let is_doc_comment = comments::is_block_doc_comment(string);

                if !terminated {
                    let msg = if is_doc_comment {
                        "unterminated block doc-comment"
                    } else {
                        "unterminated block comment"
                    };
                    let last_bpos = self.pos;
                    self.fatal_span_(start, last_bpos, msg).raise();
                }

                if is_doc_comment {
                    self.forbid_bare_cr(start, string, "bare CR not allowed in block doc-comment");
                    token::DocComment(Symbol::intern(string))
                } else {
                    token::Comment
                }
            }
            rustc_lexer::TokenKind::Whitespace => token::Whitespace,
            rustc_lexer::TokenKind::Ident | rustc_lexer::TokenKind::RawIdent => {
                let is_raw_ident = token == rustc_lexer::TokenKind::RawIdent;
                let mut ident_start = start;
                if is_raw_ident {
                    ident_start = ident_start + BytePos(2);
                }
                let sym = nfc_normalize(self.str_from(ident_start));
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
                        self.sess
                            .span_diagnostic
                            .struct_span_warn(
                                self.mk_sp(suffix_start, self.pos),
                                "underscore literal suffix is not allowed",
                            )
                            .warn(
                                "this was previously accepted by the compiler but is \
                                   being phased out; it will become a hard error in \
                                   a future release!",
                            )
                            .note(
                                "see issue #42326 \
                                 <https://github.com/rust-lang/rust/issues/42326> \
                                 for more information",
                            )
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
                    self.err_span_(start, self.pos, "lifetimes cannot start with a number");
                }
                let ident = Symbol::intern(lifetime_name);
                token::Lifetime(ident)
            }
            rustc_lexer::TokenKind::Semi => token::Semi,
            rustc_lexer::TokenKind::Comma => token::Comma,
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
            rustc_lexer::TokenKind::Colon => token::Colon,
            rustc_lexer::TokenKind::Dollar => token::Dollar,
            rustc_lexer::TokenKind::Eq => token::Eq,
            rustc_lexer::TokenKind::Not => token::Not,
            rustc_lexer::TokenKind::Lt => token::Lt,
            rustc_lexer::TokenKind::Gt => token::Gt,
            rustc_lexer::TokenKind::Minus => token::BinOp(token::Minus),
            rustc_lexer::TokenKind::And => token::BinOp(token::And),
            rustc_lexer::TokenKind::Or => token::BinOp(token::Or),
            rustc_lexer::TokenKind::Plus => token::BinOp(token::Plus),
            rustc_lexer::TokenKind::Star => token::BinOp(token::Star),
            rustc_lexer::TokenKind::Slash => token::BinOp(token::Slash),
            rustc_lexer::TokenKind::Caret => token::BinOp(token::Caret),
            rustc_lexer::TokenKind::Percent => token::BinOp(token::Percent),

            rustc_lexer::TokenKind::Unknown => {
                let c = self.str_from(start).chars().next().unwrap();
                let mut err =
                    self.struct_fatal_span_char(start, self.pos, "unknown start of token", c);
                // FIXME: the lexer could be used to turn the ASCII version of unicode homoglyphs,
                // instead of keeping a table in `check_for_substitution`into the token. Ideally,
                // this should be inside `rustc_lexer`. However, we should first remove compound
                // tokens like `<<` from `rustc_lexer`, and then add fancier error recovery to it,
                // as there will be less overall work to do this way.
                let token = unicode_chars::check_for_substitution(self, start, c, &mut err)
                    .unwrap_or_else(|| token::Unknown(self.symbol_from(start)));
                err.emit();
                token
            }
        }
    }

    fn cook_lexer_literal(
        &self,
        start: BytePos,
        suffix_start: BytePos,
        kind: rustc_lexer::LiteralKind,
    ) -> (token::LitKind, Symbol) {
        match kind {
            rustc_lexer::LiteralKind::Char { terminated } => {
                if !terminated {
                    self.fatal_span_(start, suffix_start, "unterminated character literal").raise()
                }
                let content_start = start + BytePos(1);
                let content_end = suffix_start - BytePos(1);
                self.validate_char_escape(content_start, content_end);
                let id = self.symbol_from_to(content_start, content_end);
                (token::Char, id)
            }
            rustc_lexer::LiteralKind::Byte { terminated } => {
                if !terminated {
                    self.fatal_span_(start + BytePos(1), suffix_start, "unterminated byte constant")
                        .raise()
                }
                let content_start = start + BytePos(2);
                let content_end = suffix_start - BytePos(1);
                self.validate_byte_escape(content_start, content_end);
                let id = self.symbol_from_to(content_start, content_end);
                (token::Byte, id)
            }
            rustc_lexer::LiteralKind::Str { terminated } => {
                if !terminated {
                    self.fatal_span_(start, suffix_start, "unterminated double quote string")
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
                    self.fatal_span_(
                        start + BytePos(1),
                        suffix_start,
                        "unterminated double quote byte string",
                    )
                    .raise()
                }
                let content_start = start + BytePos(2);
                let content_end = suffix_start - BytePos(1);
                self.validate_byte_str_escape(content_start, content_end);
                let id = self.symbol_from_to(content_start, content_end);
                (token::ByteStr, id)
            }
            rustc_lexer::LiteralKind::RawStr(unvalidated_raw_str) => {
                let valid_raw_str = self.validate_and_report_errors(start, unvalidated_raw_str);
                let n_hashes = valid_raw_str.num_hashes();
                let n = u32::from(n_hashes);

                let content_start = start + BytePos(2 + n);
                let content_end = suffix_start - BytePos(1 + n);
                self.validate_raw_str_escape(content_start, content_end);
                let id = self.symbol_from_to(content_start, content_end);
                (token::StrRaw(n_hashes), id)
            }
            rustc_lexer::LiteralKind::RawByteStr(unvalidated_raw_str) => {
                let validated_raw_str = self.validate_and_report_errors(start, unvalidated_raw_str);
                let n_hashes = validated_raw_str.num_hashes();
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
            }
            rustc_lexer::LiteralKind::Float { base, empty_exponent } => {
                if empty_exponent {
                    let mut err = self.struct_span_fatal(
                        start,
                        self.pos,
                        "expected at least one digit in exponent",
                    );
                    err.emit();
                }

                match base {
                    Base::Hexadecimal => self.err_span_(
                        start,
                        suffix_start,
                        "hexadecimal float literal is not supported",
                    ),
                    Base::Octal => {
                        self.err_span_(start, suffix_start, "octal float literal is not supported")
                    }
                    Base::Binary => {
                        self.err_span_(start, suffix_start, "binary float literal is not supported")
                    }
                    _ => (),
                }

                let id = self.symbol_from_to(start, suffix_start);
                (token::Float, id)
            }
        }
    }

    #[inline]
    fn src_index(&self, pos: BytePos) -> usize {
        (pos - self.start_pos).to_usize()
    }

    /// Slice of the source text from `start` up to but excluding `self.pos`,
    /// meaning the slice does not include the character `self.ch`.
    fn str_from(&self, start: BytePos) -> &str {
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
    fn str_from_to(&self, start: BytePos, end: BytePos) -> &str {
        &self.src[self.src_index(start)..self.src_index(end)]
    }

    fn forbid_bare_cr(&self, start: BytePos, s: &str, errmsg: &str) {
        let mut idx = 0;
        loop {
            idx = match s[idx..].find('\r') {
                None => break,
                Some(it) => idx + it + 1,
            };
            self.err_span_(start + BytePos(idx as u32 - 1), start + BytePos(idx as u32), errmsg);
        }
    }

    fn validate_and_report_errors(
        &self,
        start: BytePos,
        unvalidated_raw_str: UnvalidatedRawStr,
    ) -> ValidatedRawStr {
        match unvalidated_raw_str.validate() {
            Err(LexRawStrError::InvalidStarter) => self.report_non_started_raw_string(start),
            Err(LexRawStrError::NoTerminator { expected, found, possible_terminator_offset }) => {
                self.report_unterminated_raw_string(
                    start,
                    expected,
                    possible_terminator_offset,
                    found,
                )
            }
            Err(LexRawStrError::TooManyDelimiters) => self.report_too_many_hashes(start),
            Ok(valid) => valid,
        }
    }

    fn report_non_started_raw_string(&self, start: BytePos) -> ! {
        let bad_char = self.str_from(start).chars().last().unwrap();
        self.struct_fatal_span_char(
            start,
            self.pos,
            "found invalid character; only `#` is allowed \
                 in raw string delimitation",
            bad_char,
        )
        .emit();
        FatalError.raise()
    }

    fn report_unterminated_raw_string(
        &self,
        start: BytePos,
        n_hashes: usize,
        possible_offset: Option<usize>,
        found_terminators: usize,
    ) -> ! {
        let mut err = self.sess.span_diagnostic.struct_span_fatal_with_code(
            self.mk_sp(start, start),
            "unterminated raw string",
            error_code!(E0748),
        );

        err.span_label(self.mk_sp(start, start), "unterminated raw string");

        if n_hashes > 0 {
            err.note(&format!(
                "this raw string should be terminated with `\"{}`",
                "#".repeat(n_hashes)
            ));
        }

        if let Some(possible_offset) = possible_offset {
            let lo = start + BytePos(possible_offset as u32);
            let hi = lo + BytePos(found_terminators as u32);
            let span = self.mk_sp(lo, hi);
            err.span_suggestion(
                span,
                "consider terminating the string here",
                "#".repeat(n_hashes),
                Applicability::MaybeIncorrect,
            );
        }

        err.emit();
        FatalError.raise()
    }

    fn report_too_many_hashes(&self, start: BytePos) -> ! {
        self.fatal_span_(
            start,
            self.pos,
            "too many `#` symbols: raw strings may be delimited by up to 65535 `#` symbols",
        )
        .raise();
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
                self.err_span_(lo, hi, &format!("invalid digit for a base {} literal", base));
            }
        }
    }
}

pub fn nfc_normalize(string: &str) -> Symbol {
    use unicode_normalization::{is_nfc_quick, IsNormalized, UnicodeNormalization};
    match is_nfc_quick(string.chars()) {
        IsNormalized::Yes => Symbol::intern(string),
        _ => {
            let normalized_str: String = string.chars().nfc().collect();
            Symbol::intern(&normalized_str)
        }
    }
}
